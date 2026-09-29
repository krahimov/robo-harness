import copy
import json
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from studio import astra_runs, storage, main, author_providers
from studio.adapters import astra
from studio.schemas import ExperimentCreate, JudgeSpec
from studio.workflows import pen_pouch_scene
from studio.task_api import get_task, bind_task


def program():
    return {'motions': [dict(target='pen', offset=[0,0,.12], yaw=0, closed=False, seconds=.5) for _ in range(4)]}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(author_providers, 'codex_status', lambda: {'configured': True})
    judge = storage.save('judge', {**JudgeSpec(name='Test judge', rubric='Measure all task outcomes.').model_dump(), 'version': 1})
    return judge


def prepare(judge, **kwargs):
    return main.prepare_experiment(ExperimentCreate(name='Pen trial', scene=pen_pouch_scene(),
        adapter='astra', demo_ids=[], judge_id=judge['id'], task_id='pen-into-open-pouch', episodes=3, **kwargs))


def frame():
    return {'t': 0, 'ee': [0,0,.2], 'euler_xyz': [3.14159,0,0], 'closed': False, 'objects': {}, 'cube': [0,0,.04], 'qpos': []}


def fake_runtime(monkeypatch, results, raw_replies=None, cancel_on_execute=None):
    history_seen, executions = [], []
    class Simulator:
        source = '<mujoco/>'
        def snapshot(self): return frame()
    monkeypatch.setattr(astra_runs, 'make_simulator', lambda scene: Simulator())
    class Planner:
        def __init__(self, folder, experiment, initial, demos, event):
            self.event = event
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def propose(self, history):
            history_seen.append(copy.deepcopy(history))
            raw = raw_replies[len(history_seen)-1] if raw_replies else json.dumps({'action':'build','program_json':json.dumps(program()),'summary':'Test motion'})
            return {'reply':raw, 'usage': {'input_tokens':10, 'output_tokens':5}}
    monkeypatch.setattr(astra, 'Planner', Planner)
    def execute(scene, task, proposed, event):
        executions.append((scene.seed, proposed.model_dump(), copy.deepcopy(task)))
        if cancel_on_execute: event.set()
        return {'scene':scene.model_dump(),'task':task,'xml':'<mujoco/>', 'frames':[frame(),{**frame(),'t':1}],
            'actions':[{}], 'frequency':1,'program':proposed.model_dump(), 'test_status':results[len(executions)-1],
            'execution_error':'Cancelled during simulation' if cancel_on_execute else None}
    monkeypatch.setattr(astra,'execute',execute)
    monkeypatch.setattr(astra_runs,'measure_task',lambda a,j:{'status':a['test_status'],'outcomes':[]})
    def evaluate(ep,artifact,judge):
        return storage.save('evaluation',{'episode_id':ep,'judge':judge,'status':artifact['test_status'],
            'outcomes':[],'summary':'Test verdict','needs_review':True})
    monkeypatch.setattr(main,'save_evaluation',evaluate)
    return history_seen, executions


def test_revision_then_freeze_separates_holdouts_and_preserves_failures(isolated,monkeypatch):
    history, executions = fake_runtime(monkeypatch,['fail','pass','pass','fail','pass'])
    prepared = prepare(isolated,max_attempts=3)
    task_before = copy.deepcopy(prepared['task'])
    run = storage.save('experiment',prepared)
    astra_runs.run(run['id'],threading.Event())
    result = main.experiment_detail(run['id'])
    assert result['outcome']=='holdout_failed'
    assert (result['holdout_passes'],result['holdout_completed'])==(2,3)
    assert len(history)==2 and len(history[1])==1
    assert len(result['attempts'])==2
    assert [e['phase'] for e in result['results']]==['development']*2+['holdout']*3
    assert result['results'][0]['evaluations'][0]['status']=='fail'
    assert all(t==task_before for _,_,t in executions)
    assert executions[0][0]==executions[1][0]==prepared['scene']['seed']
    assert [s for s,_,_ in executions[2:]]==prepared['holdout_seeds']
    assert len({e['program_sha256'] for e in result['results'][2:]})==1
    assert result['frozen_program_sha256']==astra_runs.digest(astra.Program(**program()).model_dump())
    assert result['contract']['task_sha256']==astra_runs.digest(task_before)
    for attempt in result['attempts']:
        assert storage.read_artifact(attempt['proposal_artifact_id'])['reply']
    # Later judgment cannot rewrite the frozen benchmark summary.
    ep=result['results'][-2]
    storage.save('evaluation',{'episode_id':ep['id'],'status':'pass'})
    assert main.experiment_detail(run['id'])['holdout_passes']==2


def test_budget_exhaustion_never_runs_holdouts(isolated,monkeypatch):
    history, executions=fake_runtime(monkeypatch,['fail','fail'])
    run=storage.save('experiment',prepare(isolated,max_attempts=2))
    astra_runs.run(run['id'],threading.Event())
    result=main.experiment_detail(run['id'])
    assert result['outcome']=='budget_exhausted' and result['frozen_program'] is None
    assert len(executions)==2 and result['holdout_completed']==0


def test_invalid_proposal_is_saved_and_consumes_budget(isolated,monkeypatch):
    history,executions=fake_runtime(monkeypatch,[],raw_replies=['{"action":"finish","program_json":"{}","summary":"done"}'])
    run=storage.save('experiment',prepare(isolated,max_attempts=1))
    astra_runs.run(run['id'],threading.Event())
    result=storage.get('experiment',run['id'])
    assert result['attempts'][0]['status']=='invalid'
    assert storage.read_artifact(result['attempts'][0]['proposal_artifact_id'])['reply']
    assert not executions and result['outcome']=='budget_exhausted'


def test_cancel_preserves_partial_episode_without_judging(isolated,monkeypatch):
    history, executions=fake_runtime(monkeypatch,['pass'],cancel_on_execute=True)
    monkeypatch.setattr(main,'save_evaluation',lambda *args:pytest.fail('Cannot judge partial cancelled rollout'))
    run=storage.save('experiment',prepare(isolated))
    astra_runs.run(run['id'],threading.Event())
    result=main.experiment_detail(run['id'])
    assert result['status']=='cancelled' and len(executions)==1
    assert result['results'][0]['evaluations'][0]['kind']=='execution'
    assert storage.read_artifact(result['results'][0]['artifact_id'])['frames']


def test_judge_error_stops_revisions_and_keeps_recording(isolated,monkeypatch):
    history,executions=fake_runtime(monkeypatch,['error'])
    run=storage.save('experiment',prepare(isolated))
    astra_runs.run(run['id'],threading.Event())
    result=main.experiment_detail(run['id'])
    assert result['status']=='failed' and len(history)==1
    assert len(result['results'])==1 and result['frozen_program'] is None


def test_final_seeds_are_unique_and_disjoint_from_prior_runs_and_context(isolated):
    first=storage.save('experiment',prepare(isolated))
    storage.save('demo',{'scene':{'seed':999}})
    second=prepare(isolated)
    assert len(set(second['holdout_seeds']))==3
    assert not set(second['holdout_seeds']) & {999, first['scene']['seed'], *first['holdout_seeds']}


def test_schema_and_target_boundaries_reject_model_scope_changes():
    for mutation in ({'scene':{}},{'task':{}},{'judge':{}},{'holdout_seeds':[1]}):
        with pytest.raises(ValidationError): astra.Program(**{**program(),**mutation})
    bad=program();bad['motions'][0]['offset'][0]=float('nan')
    with pytest.raises(ValidationError):astra.Program(**bad)
    bad=program();bad['motions']=bad['motions']*4
    for motion in bad['motions']:motion['seconds']=3
    with pytest.raises(ValidationError):astra.Program(**bad)
    bad=program();bad['motions'][0]['target']='unknown'
    scene=pen_pouch_scene()
    with pytest.raises(ValueError):astra.validate_targets(astra.Program(**bad),scene,bind_task(scene,get_task('pen-into-open-pouch')))


def test_api_validation_and_dispatch(isolated,monkeypatch):
    calls=[]
    monkeypatch.setattr(main,'schedule_experiment',lambda r:calls.append(r))
    client=TestClient(main.app)
    spec={'name':'Pen trial','scene':pen_pouch_scene().model_dump(),'adapter':'astra','demo_ids':[],
        'judge_id':isolated['id'],'task_id':'pen-into-open-pouch','episodes':3,'max_attempts':2}
    response=client.post('/api/experiments',json=spec)
    assert response.status_code==202,response.text
    assert calls[0]['mode']=='improve_program'
    assert client.post('/api/experiments',json={**spec,'max_attempts':99}).status_code==422
    assert client.post('/api/experiments',json={**spec,'episodes':24}).status_code==422
    assert client.post('/api/experiments',json={**spec,'horizon_steps':20}).status_code==422
    assert client.post('/api/experiments',json={**spec,'holdout_seeds':[4]}).status_code==422
    monkeypatch.setattr(author_providers,'codex_status',lambda:{'configured':False})
    assert client.post('/api/experiments',json=spec).status_code==409


def test_state_failure_cannot_be_overridden_by_judge_pass(isolated,monkeypatch):
    history,executions=fake_runtime(monkeypatch,['pass'])
    monkeypatch.setattr(astra_runs,'measure_task',lambda a,j:{'status':'fail','outcomes':[]})
    run=storage.save('experiment',prepare(isolated,max_attempts=1))
    astra_runs.run(run['id'],threading.Event())
    result=main.experiment_detail(run['id'])
    assert result['outcome']=='budget_exhausted'
    assert not result['results'][0]['eligible']


def test_executor_cancellation_retains_aligned_partial_recording(monkeypatch):
    event=threading.Event()
    class Simulator:
        source='<mujoco/>'
        frequency=15
        def snapshot(self):return {**frame(),'objects':{'pen':{'position':[-.2,-.1,.04]}}}
        def step(self,action):event.set();return {**self.snapshot(),'t':1/15}
    monkeypatch.setattr(astra,'make_simulator',lambda scene:Simulator())
    scene=pen_pouch_scene();task=bind_task(scene,get_task('pen-into-open-pouch'))
    artifact=astra.execute(scene,task,astra.Program(**program()),event)
    assert artifact['execution_error']=='Cancelled during simulation'
    assert len(artifact['frames'])==len(artifact['actions'])+1==2


def test_native_astra_executor_uses_contact_and_generic_goal_targets():
    from studio.adapters.franka import available
    if not available():pytest.skip('Menagerie assets not installed')
    from studio.workflows import pen_pouch_waypoints
    from studio.judges import state_judge
    scene=pen_pouch_scene();task=bind_task(scene,get_task('pen-into-open-pouch'))
    goal=astra.goals_for(scene,task)['goal_pen_inside']
    motions=[{'target':'goal_pen_inside','offset':[p[k]-g for k,g in zip(('x','y','z'),goal)],
        'yaw':p['euler_xyz'][2],'closed':p['closed'],'seconds':p['seconds']} for p in pen_pouch_waypoints(scene)]
    artifact=astra.execute(scene,task,astra.Program(motions=motions),threading.Event())
    assert artifact['execution_error'] is None
    assert state_judge(artifact,{'require_review':True})['status']=='pass'
    assert len(artifact['frames'])==len(artifact['actions'])+1
    assert artifact['model']['adapter_version']==astra.VERSION


def test_planner_prompt_samples_context_and_never_contains_holdout_seeds(isolated,monkeypatch,tmp_path):
    monkeypatch.setattr(astra,'rendered_frames',lambda *args:[{'image':'dGVzdA=='}])
    prepared=prepare(isolated)
    scene=pen_pouch_scene()
    initial={'scene':scene.model_dump(),'xml':'<mujoco/>','frames':[frame()]}
    demonstration={'id':'demo','name':'Example','artifact':{'frequency':15,
        'frames':[{**frame(),'t':i/15} for i in range(101)], 'actions':[{'x':i/1000} for i in range(100)]}}
    planner=astra.Planner(tmp_path,prepared,initial,[demonstration],threading.Event())
    prompts=[]
    monkeypatch.setattr(planner.session,'run',lambda prompt,images,event:(prompts.append(json.loads(prompt)) or '{}',{}))
    planner.propose([])
    prompt=prompts[0]
    assert 'holdout_seeds' not in prompt
    assert all(str(seed) not in json.dumps(prompt) for seed in prepared['holdout_seeds'])
    samples=prompt['demonstrations'][0]['samples']
    assert len(samples)==16
    assert samples[0]['action']=={'x':0} and samples[-1]['action']=={'x':.099}
    assert len(planner.images)==2


def test_stress_and_feedback_are_frozen_and_feedback_is_fresh_per_episode(isolated,monkeypatch):
    history,executions=fake_runtime(monkeypatch,['pass']*4)
    prepared=prepare(isolated,feedback_mode='checkpoints',max_checkpoints=2,max_attempts=1)
    assert prepared['contract']['feedback_policy_sha256']
    old_execute=astra.execute;feedback=[]
    def execute(scene,task,program,event,checkpoint=None):
        feedback.append(checkpoint)
        return old_execute(scene,task,program,event)
    monkeypatch.setattr(astra,'execute',execute)
    run=storage.save('experiment',prepared);astra_runs.run(run['id'],threading.Event())
    assert len(feedback)==4 and len({id(f) for f in feedback})==4
    assert all(f.count==0 and f.limit==2 for f in feedback)
    assert len(history)==1
    stress=prepare(isolated,stress_suite='pen-v1')
    assert stress['episodes']==len(stress['stress_cases'])==len(stress['holdout_seeds'])==8
