import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from studio import storage as db, task_api
from studio.schemas import Scene, JudgeSpec
from studio.adapters.simulation import make_simulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'integrations/runpod'))
from bundle import digest, fixture, source_files
from seed_trial import seed_trial
from run_trial import audit, Client


@pytest.fixture
def reference(tmp_path, monkeypatch):
    source = tmp_path/'source'
    monkeypatch.setattr(db, 'ROOT', source)
    scene = Scene()
    sim = make_simulator(scene)
    initial = sim.snapshot()
    action = {'x': -.2, 'y': 0, 'z': .2, 'closed': False}
    task = task_api.resolve_task(scene, 'block-to-tray')
    artifact = {'xml': sim.source, 'scene': scene.model_dump(), 'frames': [initial, sim.step(action)], 'actions': [action], 'frequency': sim.frequency, 'task': task}
    demo = db.save('demo', {'name': 'Frozen demo', 'version': 1, 'scene': scene.model_dump(), 'embodiment': scene.embodiment,
        'artifact_id': db.write_artifact(artifact), 'task': task, 'reference_image_ids': ['PRIVATE_PHOTO'], 'author_provider': 'codex'})
    judge = db.save('judge', {**JudgeSpec(name='State checks', rubric='Inspect the declared state outcomes.').model_dump(), 'version': 1})
    eval_scene = Scene(seed=2000, variation=.02).model_dump()
    task = task_api.resolve_task(eval_scene, 'block-to-tray')
    model = {'policy_sha256': 'expected-hash', 'source_commit': 'expected-commit', 'runtime_adapter_version': 'icrt-runtime-v2', 'action_reference': 'current_observation', 'device': 'cuda'}
    conditions = []
    for name, context in [('None', []), ('One', [demo])]:
        exp = db.save('experiment', {'name': name, 'scene': eval_scene, 'episodes': 1, 'horizon_steps': 1, 'instructions': 'Place the box.', 'task': task,
            'judge': judge, 'context': context, 'status': 'completed', 'demo_ids': [d['id'] for d in context]})
        trace = {'model': {'model': model, 'adapter_version': 'icrt-studio-v2'}}
        db.save('episode', {'experiment_id': exp['id'], 'artifact_id': db.write_artifact(trace)})
        conditions.append({'name': name, 'demo_ids': exp['demo_ids'], 'experiment_id': exp['id']})
    comparison = db.save('comparison', {'name': 'Reference', 'conditions': conditions, 'judge_id': judge['id'], 'adapter': 'icrt', 'scene': eval_scene,
        'episodes': 1, 'horizon_steps': 1, 'instructions': 'Place the box.', 'task_id': 'block-to-tray'})
    data = fixture(source, comparison['id'])
    return data, source, demo, artifact


def test_export_uses_frozen_context_and_omits_private_references(reference):
    data, source, demo, artifact = reference
    db.update('demo', demo['id'], name='Renamed after comparison')
    exported = fixture(source, data['source_comparison_id'])
    assert exported['demos'][0]['record']['name'] == 'Frozen demo'
    encoded = json.dumps(exported)
    assert 'PRIVATE_PHOTO' not in encoded and 'author_provider' not in encoded
    assert 'xml' not in exported['demos'][0]['artifact']
    assert exported['demos'][0]['artifact']['frames'] == artifact['frames']


def test_seed_preserves_trajectory_and_task_without_overwriting(reference, tmp_path):
    data, source, demo, artifact = reference
    path = tmp_path/'trial.json'
    path.write_text(json.dumps(data))
    target = tmp_path/'new'
    seed_trial(path, target)
    assert db.ROOT == source
    previous = db.ROOT
    try:
        db.ROOT = target
        restored = db.read_artifact(db.get('demo', demo['id'])['artifact_id'])
        assert restored['frames'] == artifact['frames']
        assert restored['actions'] == artifact['actions']
        assert restored['task'] == artifact['task']
        assert 'name="wrist"' in restored['xml']
    finally:
        db.ROOT = previous
    with pytest.raises(ValueError, match='never overwritten'):
        seed_trial(path, target)
    assert (target/'studio.sqlite').exists()


@pytest.mark.parametrize('change', ['trajectory', 'task', 'geometry'])
def test_invalid_restore_rolls_back_new_data(reference, tmp_path, change):
    data, source, _, _ = reference
    if change == 'trajectory':
        data['demos'][0]['artifact']['actions'][0]['x'] = .3
    elif change == 'task':
        data['task']['requirements'][0]['required'] = False
    else:
        item = data['demos'][0]
        item['artifact']['frames'][0]['qpos'].append(0)
        item['trajectory_sha256'] = digest(item['artifact'])
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(data))
    target = tmp_path/'invalid'
    with pytest.raises(ValueError):
        seed_trial(path, target)
    assert not target.exists() and db.ROOT == source


def test_source_allowlist_excludes_secrets_and_symlinks(tmp_path):
    for name in ['backend/studio/main.py', 'backend/studio/.env', 'integrations/icrt/.venv/secret.py', '.data/studio.sqlite', 'web/dist/index.html']:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test')
    (tmp_path/'backend/studio/leak.py').symlink_to(tmp_path/'.data/studio.sqlite')
    included = [p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)]
    assert 'backend/studio/main.py' in included
    assert 'backend/studio/.env' not in included
    assert not any('.venv' in p or '.data' in p or 'leak' in p for p in included)


@pytest.mark.parametrize('url', ['https://example.com/api', 'http://evil.example/api', 'http://token@localhost/api'])
def test_trial_client_requires_private_tunnel(url):
    with pytest.raises(ValueError):
        Client(url)


def test_cloud_origin_is_opt_in_and_does_not_allow_public_origins(tmp_path, monkeypatch):
    from studio.main import app
    monkeypatch.setattr(db, 'ROOT', tmp_path)
    with TestClient(app) as client:
        monkeypatch.delenv('STUDIO_UI_ORIGIN', raising=False)
        assert client.post('/api/scene/preview', json={}, headers={'Origin': 'http://127.0.0.1:8789'}).status_code == 403
        monkeypatch.setenv('STUDIO_UI_ORIGIN', 'http://127.0.0.1:8789')
        assert client.post('/api/scene/preview', json={}, headers={'Origin': 'http://127.0.0.1:8789'}).status_code == 200
        assert client.post('/api/scene/preview', json={}, headers={'Origin': 'https://example.com'}).status_code == 403


def test_audit_rejects_cpu_fallback_and_skipped_predictions(reference):
    data, _, _, _ = reference
    spec = data['spec']
    bundle = {'comparison': {**spec, 'seeds': [2000]}, 'experiments': []}
    artifacts = {}
    for i, c in enumerate(spec['conditions']):
        id = str(i)
        bundle['experiments'].append({'task': data['task'], 'judge': data['judge'], 'status': 'completed', 'demo_ids': c['demo_ids'],
            'results': [{'id': id, 'artifact_id': id, 'scene': {'seed': 2000}}]})
        artifacts[id] = {'actions': [{}], 'predictions': [{}], 'model': {'fixed_weights': True, 'gradient_updates': 0,
            'adapter_version': data['expected_model']['adapter_version'], 'model': {**data['expected_model'], 'device': 'cuda'},
            'context_sampling': [{'id': d} for d in c['demo_ids']]}}
    assert audit(bundle, artifacts, data)['valid']
    artifacts['0']['model']['model']['device'] = 'cpu'
    assert not audit(bundle, artifacts, data)['valid']
    assert audit(bundle, artifacts, data, require_cuda=False)['valid']
    artifacts['1']['predictions'] = []
    assert not audit(bundle, artifacts, data, require_cuda=False)['valid']
