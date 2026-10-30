"""Development/revision and frozen holdout phases in the regular experiment store."""
import hashlib
import json
import secrets
import tempfile
from pathlib import Path

from studio import storage as db
from studio.adapters import astra
from studio.adapters.simulation import make_simulator
from studio.authoring import AuthorCancelled, AuthorError, check_cancel
from studio.schemas import Scene
from studio.judges import state_judge as measure_task


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare(spec, prepared):
    from studio.author_providers import model_for, subscription_status
    from fastapi import HTTPException
    connection = subscription_status(spec.actor_provider)
    if not connection['configured']:
        raise HTTPException(409, connection.get('message') or 'Sign in to the selected subscription on the backend machine')
    if spec.scene.embodiment != 'franka-panda-v1':
        raise ValueError('LLM motion programs require the Franka Panda embodiment')
    if spec.horizon_steps is not None:
        raise ValueError('Astra uses bounded motion durations instead of horizon_steps')
    if len(prepared['context']) > 4:
        raise ValueError('Astra accepts up to four sampled demonstrations')
    if spec.episodes > 8:
        raise ValueError('Astra supports up to eight final holdout episodes per run')
    if spec.stress_suite == 'pen-v1' and (prepared['task'].get('id') != 'pen-into-open-pouch' or not any(o.id=='pen' for o in spec.scene.objects)):
        raise ValueError('The pen stress suite requires the original pen task and fixture')
    from studio.stress import CASES
    count = len(CASES) if spec.stress_suite == 'pen-v1' else spec.episodes
    excluded = {spec.scene.seed, *[d['scene']['seed'] for d in db.all_records('demo')]}
    for old in db.all_records('experiment'):
        excluded.update(old.get('holdout_seeds', []))
        excluded.update(range(old['scene']['seed'], old['scene']['seed'] + old.get('episodes', 1)))
    seeds = []
    while len(seeds) < count:
        seed = secrets.randbelow(2147483548)
        if seed not in excluded:
            seeds.append(seed)
            excluded.add(seed)
    prepared['episodes'] = count
    prepared['stress_cases'] = CASES if spec.stress_suite == 'pen-v1' else []
    from studio.adapters.astra_feedback import POLICY_SHA256
    prepared.update(mode='improve_program', max_attempts=spec.max_attempts,
        holdout_seeds=seeds, attempts=[], phase='queued', holdout_completed=0,
        holdout_passes=0, outcome=None, frozen_program=None,
        contract={'actor': model_for(spec.actor_provider), 'actor_provider': spec.actor_provider,
            'adapter_version': astra.VERSION, 'fixed_weights': True,
            'mutable': ['motion_program'], 'feedback_mode':spec.feedback_mode, 'max_checkpoints':spec.max_checkpoints,
            'feedback_policy_sha256':POLICY_SHA256 if spec.feedback_mode=='checkpoints' else None, 'context_sampling': 'up to 16 state/action pairs per demonstration',
            'observations': 'initial RGB and explicit simulator state; live relative motion targets',
            'task_sha256': digest(prepared['task']), 'scene_sha256': digest(prepared['scene']),
            'judge_sha256': digest(prepared['judge'])})
    return prepared


def persist_episode(run, artifact, phase, attempt, evaluate):
    # Artifact and episode are durable before any judge call can fail.
    episode = db.save('episode', {'experiment_id': run['id'], 'index':
        len([e for e in db.all_records('episode') if e['experiment_id'] == run['id']])+1,
        'phase': phase, 'attempt': attempt, 'scene': artifact['scene'],
        'artifact_id': db.write_artifact(artifact), 'duration': artifact['frames'][-1]['t'],
        'selected_demo_id': None, 'stress_case':artifact.get('stress_case'), 'feedback_trace':artifact.get('feedback_trace',[]), 'task':artifact['task'], 'program_sha256': digest(artifact['program'])})
    if artifact.get('execution_error'):
        verdict = db.save('evaluation', {'episode_id': episode['id'], 'judge': run['judge'],
            'status': 'error', 'outcomes': [], 'needs_review': True, 'kind': 'execution',
            'summary': artifact['execution_error']})
        state = {'status': 'error', 'outcomes': []}
    else:
        state = measure_task(artifact, {'require_review': True})
        verdict = evaluate(episode['id'], artifact, run['judge'])
    eligible = state['status'] == 'pass' and verdict['status'] == 'pass'
    episode = db.update('episode', episode['id'], evaluation_id=verdict['id'],
        initial_evaluation_id=verdict['id'], state_verdict=state, eligible=eligible)
    current = db.get('experiment', run['id'])
    db.update('experiment', run['id'], completed=current['completed']+1)
    return episode, verdict, state


def execute_episode(experiment, scene, task, program, event):
    provider = experiment.get('actor_provider', 'codex')
    if experiment.get('feedback_mode') != 'checkpoints':
        return astra.execute(scene,task,program,event,provider=provider)
    from studio.adapters.astra_feedback import Feedback
    feedback=Feedback(scene,task,event,experiment.get('max_checkpoints',2),
        progress=lambda message:db.update('experiment',experiment['id'],message=message),provider=provider)
    return astra.execute(scene,task,program,event,checkpoint=feedback,provider=provider)


def run(id, event):
    from studio.main import save_evaluation
    from studio.author_providers import model_name
    experiment = db.get('experiment', id)
    actor = model_name(experiment.get('actor_provider', 'codex'))
    attempts, history = [], []
    try:
        check_cancel(event)
        db.update('experiment', id, status='running', phase='preparing', message='Preparing initial cameras and sampled context')
        scene = Scene(**experiment['scene'])
        sim = make_simulator(scene)
        initial = {'scene': scene.model_dump(), 'xml': sim.source, 'frames': [sim.snapshot()]}
        demos = [{**d, 'artifact': db.read_artifact(d['artifact_id'])} for d in experiment['context']]
        frozen = None
        with tempfile.TemporaryDirectory(prefix='studio-astra-run-') as folder:
            with astra.Planner(Path(folder), experiment, initial, demos, event) as planner:
                for attempt in range(1, experiment['max_attempts']+1):
                    check_cancel(event)
                    db.update('experiment', id, phase='proposing', message=f'{actor} is proposing development attempt {attempt}')
                    item = {'number': attempt, 'status': 'proposing'}
                    attempts.append(item)
                    db.update('experiment', id, attempts=attempts)
                    proposal = planner.propose(history)
                    item.update(proposal_artifact_id=db.write_artifact(proposal), usage=proposal['usage'])
                    db.update('experiment', id, attempts=attempts)
                    try:
                        program, summary = astra.parse_proposal(proposal, scene, experiment['task'])
                    except ValueError:
                        item.update(status='invalid', summary='Invalid motion program; inspect the saved proposal for details.')
                        history.append({'attempt': attempt, 'proposal': proposal['reply'],
                            'error': 'Invalid program. Follow the motion schema, target names, finite workspace offsets and 30 second limit.'})
                        db.update('experiment', id, attempts=attempts)
                        continue
                    item.update(status='simulating', summary=summary, program=program.model_dump())
                    db.update('experiment', id, phase='development', attempts=attempts, message=f'Simulating development attempt {attempt}')
                    artifact = execute_episode(experiment,scene,experiment['task'],program,event)
                    item.update(status='judging')
                    db.update('experiment', id, phase='judging', attempts=attempts, message=f'Judging development attempt {attempt}')
                    episode, verdict, state = persist_episode(experiment, artifact, 'development', attempt, save_evaluation)
                    item.update(status='pass' if episode['eligible'] else 'fail', episode_id=episode['id'],
                        judge_status=verdict['status'], state_status=state['status'])
                    db.update('experiment', id, attempts=attempts)
                    check_cancel(event)
                    if verdict['status'] == 'error' and not artifact.get('execution_error'):
                        raise AuthorError('Judge failed. Evidence is saved; fix the judge connection before starting another run.')
                    history.append({'attempt': attempt, 'program': program.model_dump(),
                        'judge': {k: verdict[k] for k in ('status','outcomes','summary')}, 'state_checks': state,
                        'trajectory': [astra.observation(f) for f in artifact['frames'][::max(1,len(artifact['frames'])//24)]]})
                    if episode['eligible']:
                        frozen = program
                        db.update('experiment', id, frozen_program=program.model_dump(),
                            frozen_program_sha256=digest(program.model_dump()), frozen_at=db.now(),
                            frozen_from_episode_id=episode['id'])
                        break
        if frozen is None:
            db.update('experiment', id, status='completed', phase='finished', outcome='budget_exhausted',
                message='Development budget exhausted. No program qualified for final evaluation.', finished_at=db.now())
            return
        # The development planner is closed. Feedback controllers are fresh per episode; no holdout history enters development.
        passed = 0
        for index, seed in enumerate(experiment['holdout_seeds']):
            check_cancel(event)
            db.update('experiment', id, phase='holdout', message=f'Frozen program: final evaluation {index+1} of {len(experiment["holdout_seeds"])}')
            holdout_scene,holdout_task=scene.model_copy(update={'seed':seed}),experiment['task']
            if experiment.get('stress_cases'):
                from studio.stress import case_scene
                holdout_scene,holdout_task=case_scene(scene,experiment['task'],experiment['stress_cases'][index],seed)
            artifact = execute_episode(experiment,holdout_scene,holdout_task,frozen,event)
            if experiment.get('stress_cases'):
                artifact['stress_case']=experiment['stress_cases'][index]
            episode, verdict, state = persist_episode(experiment, artifact, 'holdout', None, save_evaluation)
            check_cancel(event)
            passed += int(episode['eligible'])
            db.update('experiment', id, holdout_completed=index+1, holdout_passes=passed)
            if verdict['status'] == 'error' and not artifact.get('execution_error'):
                raise AuthorError('Judge failed during final evaluation. Completed evidence and frozen program are saved.')
        db.update('experiment', id, status='completed', phase='finished',
            outcome='holdout_passed' if passed == len(experiment['holdout_seeds']) else 'holdout_failed',
            message=f'{passed}/{len(experiment["holdout_seeds"])} final evaluations passed both state and judge checks. Human review is separate.', finished_at=db.now())
    except AuthorCancelled:
        if attempts and attempts[-1]['status'] in {'proposing', 'simulating', 'judging'}:
            attempts[-1]['status'] = 'cancelled'
            db.update('experiment', id, attempts=attempts)
        db.update('experiment', id, status='cancelled', phase='finished', message='Cancelled. Saved proposals and recordings are preserved.', finished_at=db.now())
    except Exception as exc:
        if attempts and attempts[-1]['status'] in {'proposing', 'simulating', 'judging'}:
            attempts[-1]['status'] = 'error'
            db.update('experiment', id, attempts=attempts)
        message = str(exc) if isinstance(exc, AuthorError) else f'{type(exc).__name__}: {actor} execution failed. Saved evidence is preserved.'
        db.update('experiment', id, status='failed', phase='finished', error=message, finished_at=db.now())
