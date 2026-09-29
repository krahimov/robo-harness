import time

import pytest
from fastapi.testclient import TestClient

from studio import storage, author_providers
from studio.comparisons import interval


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"id": "codex", "configured": False, "name": "Codex", "message": "Test fixture"})
    from studio.main import app
    with TestClient(app) as client:
        yield client


def setup(client):
    bootstrap = client.get('/api/bootstrap').json()
    demo = bootstrap['demos'][0]
    second = client.post('/api/demos', json={'name': 'Second start', 'scene': {'seed': 17, 'cube_x': -.29}}).json()
    return {'name': 'Context comparison', 'adapter': 'replay', 'judge_id': bootstrap['judges'][0]['id'],
        'scene': {'seed': 1000, 'variation': .04}, 'episodes': 2, 'horizon_steps': 190,
        'conditions': [{'name': 'One', 'demo_ids': [demo['id']]}, {'name': 'Two', 'demo_ids': [demo['id'], second['id']]}]}


def finish(client, spec):
    response = client.post('/api/comparisons', json=spec)
    assert response.status_code == 202, response.text
    id = response.json()['id']
    until = time.monotonic()+15
    while time.monotonic() < until:
        result = client.get('/api/comparisons/'+id).json()
        if result['status'] != 'running':
            assert result['status'] == 'completed', result
            return result
        time.sleep(.02)
    pytest.fail('Comparison did not complete')


def test_matched_seeds_horizon_and_persistent_original_evidence(client):
    spec = setup(client)
    result = finish(client, spec)
    assert result['seeds'] == [1000, 1001]
    for condition in result['conditions']:
        assert [a['seed'] for a in condition['attempts']] == [1000, 1001]
        run = client.get('/api/experiments/'+condition['experiment_id']).json()
        assert run['horizon_steps'] == 190
        assert run['judge']['id'] == spec['judge_id']
        for episode in run['results']:
            artifact = client.get('/api/artifacts/'+episode['artifact_id']).json()
            assert len(artifact['actions']) == 190
            assert episode['initial_evaluation_id'] == episode['evaluation_id']
    exported = client.get('/api/comparisons/'+result['id']+'/export').json()
    assert exported['comparison'] == result
    assert len(exported['experiments']) == 2
    assert client.get('/api/comparisons').json()[0]['id'] == result['id']


def test_comparison_rejects_seen_seeds_and_duplicate_context_before_scheduling(client):
    spec = setup(client)
    spec['scene']['seed'] = 42
    assert client.post('/api/comparisons', json=spec).status_code == 422
    assert not storage.all_records('experiment')
    spec['scene']['seed'] = 1000
    spec['conditions'][1]['demo_ids'] = spec['conditions'][0]['demo_ids']
    assert client.post('/api/comparisons', json=spec).status_code == 422
    assert not storage.all_records('comparison')


def test_replay_zero_demo_rejected_atomically_and_no_inference_is_claimed(client):
    spec = setup(client)
    spec['conditions'][1]['demo_ids'] = []
    result = client.post('/api/comparisons', json=spec)
    assert result.status_code == 422
    assert 'requires a demonstration' in result.text
    assert not storage.all_records('experiment')
    assert not storage.all_records('comparison')


def test_later_rejudge_does_not_change_original_comparison(client, monkeypatch):
    from studio import main
    result = finish(client, setup(client))
    condition = result['conditions'][0]
    episode = condition['attempts'][0]['episode_id']
    monkeypatch.setattr(main, 'evaluate', lambda *args: {'status': 'error', 'outcomes': [], 'needs_review': True, 'kind': 'state', 'summary': 'Fixture failure'})
    response = client.post('/api/episodes/'+episode+'/evaluate', json={'judge_id': storage.all_records('judge')[0]['id']})
    assert response.status_code == 201
    assert client.get('/api/comparisons/'+result['id']).json() == result


def test_unknowns_excluded_from_rates_and_paired_differences(client, monkeypatch):
    from studio import main
    original = main.evaluate
    calls = 0
    def partially_known(artifact, judge):
        nonlocal calls
        calls += 1
        result = original(artifact, judge)
        if calls == 1:
            result['status'] = 'unknown'
        return result
    monkeypatch.setattr(main, 'evaluate', partially_known)
    result = finish(client, setup(client))
    first, second = result['conditions']
    assert first['resolved'] == 1 and first['unresolved'] == 1
    assert second['paired']['count'] == 1
    assert second['recorded'] == 2


def test_wilson_interval_is_bounded_and_does_not_hide_small_samples():
    assert interval(0, 0) is None
    low, high = interval(1, 1)
    assert 0 < low < .5 and .999 < high <= 1
    low, high = interval(0, 1)
    assert low == 0 and .5 < high < 1


def test_all_links_visible_before_first_comparison_worker_starts(client, monkeypatch):
    from studio import main
    from studio.comparisons import comparison_detail
    schedule = main.schedule_experiment
    observed = []
    def inspect_before_scheduling(experiment):
        detail = comparison_detail(experiment['comparison_id'])
        assert len(detail['conditions']) == 2
        assert all(storage.get('experiment', c['experiment_id']) for c in detail['conditions'])
        observed.append(detail['id'])
        schedule(experiment)
    monkeypatch.setattr(main, 'schedule_experiment', inspect_before_scheduling)
    result = finish(client, setup(client))
    assert observed == [result['id'], result['id']]


def test_multi_operation_task_export_import_and_normal_horizon(client):
    task = client.get('/api/tasks/two-box-sort').json()
    response = client.post('/api/demos', json={'name': 'Sort both boxes', 'scene': task['suggested_scene'], 'task_id': task['id']})
    assert response.status_code == 201, response.text
    demo = response.json()
    exported = client.get('/api/demos/'+demo['id']+'/export').json()
    assert len(exported['actions']) > 180
    assert len(exported['task']['operations']) == 2
    imported = client.post('/api/demos/import', json=exported)
    assert imported.status_code == 201, imported.text
    imported = imported.json()
    assert imported['task']['requirements'] == exported['task']['requirements']
    assert imported['task_id']
    spec = {'name': 'Sorting run', 'demo_ids': [imported['id']], 'judge_id': storage.all_records('judge')[0]['id'],
        'scene': imported['scene'], 'task_id': imported['task_id'], 'episodes': 1, 'adapter': 'replay'}
    response = client.post('/api/experiments', json=spec)
    assert response.status_code == 202, response.text
    until = time.monotonic()+15
    while time.monotonic() < until:
        run = client.get('/api/experiments/'+response.json()['id']).json()
        if run['status'] not in {'queued', 'running'}:
            break
        time.sleep(.02)
    assert run['status'] == 'completed', run
    artifact = client.get('/api/artifacts/'+run['results'][0]['artifact_id']).json()
    assert len(artifact['actions']) == len(exported['actions'])
    assert run['results'][0]['evaluations'][0]['status'] == 'pass'
