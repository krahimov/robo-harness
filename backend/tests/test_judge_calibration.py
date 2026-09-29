import copy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studio import judge_lab_api as lab, storage as db
from studio.calibration import calibration, propose_rubric, review_queue


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "ROOT", tmp_path)
    judge = db.save("judge", {"name": "Inspector", "kind": "vlm", "provider": "codex", "model": "gpt-6-astra",
        "rubric": "Inspect the final state for placement. Review uncertainty.", "tools": ["inspect_state"], "require_review": True, "version": 1})
    task = {"id": "test-task", "version": 3, "requirements": [{"id": "placed", "required": True, "definition": "Inside human-defined bounds"},
        {"id": "released", "required": True, "definition": "Release the object"}]}
    def evaluation(status="pass", outcomes=None):
        artifact_id = db.write_artifact({"task": task, "frames": [{"t": 0}], "scene": {}})
        episode = db.save("episode", {"artifact_id": artifact_id, "experiment_id": "experiment", "index": 1})
        return db.save("evaluation", {"episode_id": episode["id"], "judge": copy.deepcopy(judge), "status": status,
            "outcomes": outcomes if outcomes is not None else [{"id": r["id"], "status": status, "explanation": "Model observation", "timestamps": [0]} for r in task["requirements"]],
            "needs_review": True, "summary": "Attempt inspected", "kind": "vlm"})
    app = FastAPI()
    app.include_router(lab.router)
    with TestClient(app) as client:
        yield client, judge, task, evaluation


def label(client, evaluation, outcome="placed", status="fail", note="Block crosses the boundary"):
    response = client.post(f"/api/evaluations/{evaluation['id']}/labels", json={"outcomes": [{"id": outcome, "status": status, "note": note}]})
    assert response.status_code == 201, response.text
    return response.json()


def test_labels_preserve_snapshots_and_latest_correction_controls_calibration(setup):
    client, judge, task, create = setup
    evaluation = create()
    original = copy.deepcopy(evaluation)
    first = label(client, evaluation)
    label(client, evaluation, "released", "unknown", "Occluded")
    metrics = calibration(judge["id"])
    assert metrics["label_count"] == 1 and metrics["agreement"] == 0
    assert metrics["false_passes"] == 1 and metrics["false_pass_rate"] == 1
    assert metrics["human_unknown_labels"] == 1
    second = label(client, evaluation, status="pass", note="Reviewed a clearer frame")
    metrics = calibration(judge["id"])
    assert metrics["label_count"] == 1 and metrics["agreement"] == 1
    assert metrics["false_passes"] == 0
    assert db.get("outcome_label", first["id"])["outcomes"][0]["status"] == "fail"
    assert second["task"] == task and second["judge"] == judge
    assert db.get("evaluation", evaluation["id"]) == original


def test_labels_cannot_invent_outcomes_and_error_judges_are_unknown(setup):
    client, judge, _, create = setup
    evaluation = create("error", [])
    response = client.post(f"/api/evaluations/{evaluation['id']}/labels", json={"outcomes": [{"id": "easier_task", "status": "pass"}]})
    assert response.status_code == 422
    label(client, evaluation, status="pass")
    result = calibration(judge["id"])
    assert result["judge_unknowns"] == 1 and result["agreement"] == 0


def test_sampling_reserves_passes_even_with_many_unknown_attempts(setup):
    client, judge, _, create = setup
    passing = create()
    for _ in range(8): create("unknown")
    queue = review_queue(judge["id"], 4)
    assert len(queue["items"]) == 4
    assert [item["evaluation"]["status"] for item in queue["items"]].count("pass") == 1
    assert queue["pass_slots_reserved"] == 1
    client.post(f"/api/evaluations/{passing['id']}/labels", json={"outcomes": [{"id": id, "status": "pass"} for id in ["placed", "released"]]})
    assert all(item["evaluation"]["id"] != passing["id"] for item in review_queue(judge["id"], 20)["items"])


def test_rubric_proposal_and_acceptance_never_rewrite_task_or_old_judge(setup, monkeypatch):
    client, judge, task, create = setup
    evaluation = create()
    correction = label(client, evaluation)
    monkeypatch.setattr(lab, "judge_connection", lambda _: {"configured": True})
    captured = {}
    def propose(saved_judge, evidence):
        captured.update(judge=saved_judge, evidence=evidence)
        return {"rubric": "Inspect block extents, not just its center. Keep the task boundaries unchanged.", "rationale": "A human corrected a false pass at the tray boundary."}
    monkeypatch.setattr(lab, "propose_rubric", propose)
    response = client.post(f"/api/judges/{judge['id']}/rubric-proposals")
    assert response.status_code == 201, response.text
    proposal = response.json()
    assert proposal["status"] == "draft"
    assert len(db.all_records("judge")) == 1
    assert proposal["evidence"][0]["label"]["label_id"] == correction["id"]
    assert captured["evidence"][0]["requirement"] == task["requirements"][0]
    response = client.post(f"/api/rubric-proposals/{proposal['id']}/accept", json={})
    assert response.status_code == 201, response.text
    version = response.json()
    assert version["version"] == 2 and version["parent_id"] == judge["id"]
    assert version["rubric"] == proposal["rubric"]
    assert db.get("judge", judge["id"]) == judge
    assert db.get("evaluation", evaluation["id"])["judge"] == judge
    assert calibration(version["id"])["label_count"] == 0
    assert client.post(f"/api/rubric-proposals/{proposal['id']}/accept", json={}).json()["id"] == version["id"]
    assert len(db.all_records("judge")) == 2


def test_rubric_generation_requires_a_resolved_correction(setup, monkeypatch):
    client, judge, _, create = setup
    monkeypatch.setattr(lab, "judge_connection", lambda _: {"configured": True})
    label(client, create(), status="pass")
    assert client.post(f"/api/judges/{judge['id']}/rubric-proposals").status_code == 409


def test_codex_proposal_receives_corrections_as_data_and_never_changes_the_task(setup):
    _, judge, task, _ = setup
    evidence = [{"requirement": task["requirements"][0], "label": {"status": "fail", "note": "Block edge was outside"}, "prediction": {"status": "pass"}}]
    original = copy.deepcopy(evidence)
    class Session:
        def __init__(self, folder, instructions):
            assert "thresholds" in instructions and "human-owned and fixed" in instructions
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def run(self, prompt, photos, event):
            assert json.loads(prompt)["correction_evidence"] == evidence
            assert photos == []
            return json.dumps({"rubric": "Inspect complete object extents at the final timestamp.", "rationale": "A center-only check missed the protruding edge."}), {}
    result = propose_rubric(judge, evidence, session_factory=Session)
    assert result["rubric"].startswith("Inspect complete")
    assert evidence == original
    assert db.all_records("rubric_proposal") == []  # Generation itself cannot activate or persist changes.


def test_api_proposal_failure_does_not_expose_endpoint_credentials(setup, monkeypatch):
    import httpx
    from studio.authoring import AuthorError
    _, judge, _, _ = setup
    monkeypatch.setenv("STUDIO_JUDGE_ENDPOINT", "https://private-credential@judge.invalid/v1/chat/completions")
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, *args, **kwargs):
            raise httpx.ConnectError("private-credential")
    monkeypatch.setattr(httpx, "Client", Client)
    with pytest.raises(AuthorError) as error:
        propose_rubric({**judge, "provider": "openai"}, [])
    assert "private-credential" not in str(error.value)
