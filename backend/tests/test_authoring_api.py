import base64
import copy
import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import authoring_api, author_providers, storage
from studio.authoring import MODEL, SceneProgram, compile_program
from studio.main import app
from studio.schemas import Scene


@pytest.fixture(autouse=True)
def isolated_authoring_store(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path / "authoring-data")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STUDIO_AUTHOR_API_KEY", raising=False)
    monkeypatch.setenv("STUDIO_AUTHOR_PROVIDER", "openai")
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"id": "codex", "name": "Codex subscription", "configured": False, "message": "Run codex login."})


@pytest.fixture
def client():
    with TestClient(app) as current:
        yield current


def upload_photo(client):
    stream = io.BytesIO()
    Image.new("RGBA", (36, 24), (20, 100, 50, 160)).save(stream, format="PNG")
    response = client.post("/api/references", json={"name": "Factory line.png",
        "data_url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")})
    assert response.status_code == 201, response.text
    return response.json()


def draft(client, **overrides):
    response = client.post("/api/scenes", json={"name": "Factory line", "prompt": "Move the box from the conveyor into the tray.", **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def eventually(check, seconds=4):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail("The authoring worker did not reach the expected state")


def settled_scene(client, scene_id, status):
    def check():
        result = client.get(f"/api/scenes/{scene_id}").json()
        if result["status"] in {"failed", "interrupted"} and result["status"] != status:
            pytest.fail(f"Unexpected authoring state: {result}")
        return result if result["status"] == status else None
    return eventually(check)


def wait_for_job_exit(scene_id):
    def done():
        with authoring_api.job_lock:
            return scene_id not in authoring_api.jobs
    eventually(done)


def compiled_result(tray_x=0.22):
    program = SceneProgram(name="Authored layout", description="Move the box to the specified tray.",
        assumptions=["Approximate scale inferred from the reference photo."], scene=Scene(tray_x=tray_x))
    spec, artifact, verification = compile_program(program)
    return {"spec": spec, "artifact": artifact, "verification": verification,
        "summary": "Compiled and recorded a box transfer.", "camera_available": False,
        "builds": 1, "usage": {"input_tokens": 12, "output_tokens": 24}}


def test_photo_upload_persists_and_serves_normalized_jpeg(client):
    reference = upload_photo(client)
    assert reference["mime_type"] == "image/jpeg"
    assert (reference["width"], reference["height"]) == (36, 24)
    assert reference == storage.get("reference", reference["id"])
    assert client.get("/api/references").json() == [reference]
    response = client.get(reference["url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-content-type-options"] == "nosniff"
    with Image.open(io.BytesIO(response.content)) as photo:
        assert photo.format == "JPEG"
        assert photo.mode == "RGB"
    assert "base64" not in json.dumps(reference)


@pytest.mark.parametrize("payload", [
    {"name": "Invalid", "data_url": "data:image/png;base64,PRIVATE_BYTES_DO_NOT_ECHO!"},
    {"name": "Invalid", "data_url": 123, "unexpected": "PRIVATE_BYTES_DO_NOT_ECHO"},
    {"name": "Invalid", "data_url": "https://example.test/PRIVATE_BYTES_DO_NOT_ECHO"},
])
def test_invalid_photo_payload_does_not_echo_uploaded_content(client, payload):
    response = client.post("/api/references", json=payload)
    assert response.status_code == 422
    assert "PRIVATE_BYTES_DO_NOT_ECHO" not in response.text
    assert client.get("/api/references").json() == []


def test_draft_keeps_references_when_astra_is_not_configured(client):
    reference = upload_photo(client)
    scene = draft(client, image_ids=[reference["id"]])
    status = client.get("/api/author/status").json()
    assert status["model"] == MODEL == "gpt-6-astra"
    assert status["configured"] is False
    assert status["max_images"] == 4
    response = client.post(f"/api/scenes/{scene['id']}/generate")
    assert response.status_code == 409
    assert "OPENAI_API_KEY" in response.json()["detail"]
    saved = client.get(f"/api/scenes/{scene['id']}").json()
    assert saved == scene
    assert saved["status"] == "draft"
    assert saved["image_ids"] == [reference["id"]]
    assert client.get("/api/scenes").json() == [scene]


def test_photo_only_draft_requires_operation_before_generation(client, monkeypatch):
    reference = upload_photo(client)
    scene = draft(client, prompt="", image_ids=[reference["id"]])
    assert scene["prompt"] == ""
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    response = client.post(f"/api/scenes/{scene['id']}/generate")
    assert response.status_code == 422
    assert "Describe the operation" in response.json()["detail"]
    assert client.get(f"/api/scenes/{scene['id']}").json()["status"] == "draft"


def test_unknown_reference_cannot_be_attached_or_loaded(client):
    response = client.post("/api/scenes", json={"name": "Missing reference", "prompt": "Build the scene", "image_ids": ["f" * 32]})
    assert response.status_code == 404
    assert client.get("/api/scenes").json() == []
    assert client.get("/api/references/" + "f" * 32 + "/image").status_code == 404
    assert client.get("/api/references/invalid/image").status_code == 404


def test_template_revision_records_once_and_exports_without_mutating_parent(client):
    response = client.post("/api/scenes/template")
    assert response.status_code == 201, response.text
    parent = response.json()
    assert parent["status"] == "ready"
    assert parent["source"] == "template"
    assert parent["model"] is None
    assert parent["verification"]["status"] == "pass"
    program = copy.deepcopy(parent["spec"])
    program["name"] = "Box moved farther left"
    program["scene"].update(cube_x=-0.28)
    program["waypoints"] = None  # Rebuild the operation for the new starting point.
    response = client.post(f"/api/scenes/{parent['id']}/preview", json=program)
    assert response.status_code == 201, response.text
    revised = response.json()
    assert revised["id"] != parent["id"]
    assert revised["parent_id"] == revised["root_id"] == parent["id"]
    assert revised["source"] == "manual"
    assert revised["status"] == "ready"
    assert revised["verification"]["status"] == "pass"
    assert client.get(f"/api/scenes/{parent['id']}").json() == parent
    artifact = client.get(f"/api/artifacts/{revised['artifact_id']}").json()
    assert artifact["task"] == parent["task"]
    assert artifact["frames"] and artifact["actions"]

    before = len(client.get("/api/bootstrap").json()["demos"])
    response = client.post(f"/api/scenes/{revised['id']}/record", json={"name": "Conveyor operation"})
    assert response.status_code == 201, response.text
    demo = response.json()
    assert demo["source"] == "manual"
    assert demo["source_scene_id"] == revised["id"]
    assert demo["author_model"] is None
    assert demo["reference_image_ids"] == []
    assert demo["scene"]["cube_x"] == -0.28
    repeated = client.post(f"/api/scenes/{revised['id']}/record", json={"name": "Duplicate click"})
    assert repeated.status_code == 201
    assert repeated.json() == demo
    assert len(client.get("/api/bootstrap").json()["demos"]) == before + 1
    assert client.get(f"/api/scenes/{parent['id']}").json() == parent

    exported = client.get(f"/api/scenes/{revised['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "robot-harness-scene-v1"
    assert bundle["scene_id"] == revised["id"]
    assert bundle["xml"] == artifact["xml"]
    assert bundle["actions"] == artifact["actions"]
    assert bundle["spec"] == revised["spec"]
    assert bundle["references"] == []
    assert client.get(f"/api/demos/{demo['id']}/export").json()["scene"] == revised["spec"]["scene"]


def test_mocked_astra_job_persists_progress_parent_context_and_real_simulation(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key-never-sent")
    reference = upload_photo(client)
    parent = client.post("/api/scenes/template").json()
    scene = draft(client, parent_id=parent["id"], image_ids=[reference["id"]])
    entered, release = threading.Event(), threading.Event()
    received = {}
    def fake_author(saved_draft, parent_program, progress, event, **options):
        received.update(draft=saved_draft, parent=parent_program)
        result = compiled_result(tray_x=parent_program["scene"]["tray_x"])
        progress("Compiled photo-guided scene candidate.")
        entered.set()
        assert release.wait(3), "Test failed to release the author worker"
        return result
    monkeypatch.setattr(authoring_api, "author_scene", fake_author)
    try:
        response = client.post(f"/api/scenes/{scene['id']}/generate")
        assert response.status_code == 202, response.text
        assert entered.wait(3)
        running = client.get(f"/api/scenes/{scene['id']}").json()
        assert running["status"] == "running"
        assert {"message": "Compiled photo-guided scene candidate."} in running["progress"]
        assert received["parent"] == parent["spec"]
        assert received["draft"]["image_ids"] == [reference["id"]]
    finally:
        release.set()
        wait_for_job_exit(scene["id"])
    ready = settled_scene(client, scene["id"], "ready")
    assert ready["source"] == "astra"
    assert ready["model"] == MODEL
    assert ready["usage"] == {"input_tokens": 12, "output_tokens": 24}
    assert ready["verification"]["status"] == "pass"
    assert "test-only-key-never-sent" not in json.dumps(ready)
    assert client.get(f"/api/artifacts/{ready['artifact_id']}").json()["frames"]
    assert client.get(f"/api/scenes/{parent['id']}").json() == parent
    demo = client.post(f"/api/scenes/{scene['id']}/record", json={"name": "Astra authored operation"}).json()
    assert demo["source"] == "scene_author"
    assert demo["author_model"] == MODEL
    assert demo["reference_image_ids"] == [reference["id"]]
    bundle = client.get(f"/api/scenes/{scene['id']}/export").json()
    assert bundle["references"] == [reference]
    assert "base64" not in json.dumps(bundle)


def test_codex_draft_keeps_provider_when_default_changes(client, monkeypatch):
    from studio import codex_author
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"id": "codex", "name": "Codex subscription", "configured": True, "message": "Signed in."})
    scene = draft(client, provider="codex")
    monkeypatch.setenv("STUDIO_AUTHOR_PROVIDER", "openai")
    received = []
    def fake_codex(saved_draft, parent_program, progress, event, **options):
        received.append(saved_draft["provider"])
        return compiled_result()
    monkeypatch.setattr(codex_author, "author_scene_codex", fake_codex)
    monkeypatch.setattr(authoring_api, "author_scene", lambda *args: pytest.fail("Do not fall back to the API"))
    assert client.post(f"/api/scenes/{scene['id']}/generate").status_code == 202
    ready = settled_scene(client, scene["id"], "ready")
    wait_for_job_exit(scene["id"])
    assert received == ["codex"] and ready["provider"] == "codex"
    demo = client.post(f"/api/scenes/{scene['id']}/record", json={"name": "Subscription example"}).json()
    assert demo["author_provider"] == "codex"
    assert client.get(f"/api/scenes/{scene['id']}/export").json()["provider"] == "codex"


def test_legacy_draft_uses_openai_even_when_default_is_codex(client, monkeypatch):
    monkeypatch.setenv("STUDIO_AUTHOR_PROVIDER", "codex")
    legacy = storage.save("scene", {"name": "Legacy", "prompt": "Place the block in the tray", "image_ids": [], "status": "draft", "progress": [], "source": "astra"})
    response = client.post(f"/api/scenes/{legacy['id']}/generate")
    assert response.status_code == 409
    assert "OPENAI_API_KEY" in response.json()["detail"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(authoring_api, "author_scene", lambda *args: compiled_result())
    assert client.post(f"/api/scenes/{legacy['id']}/generate").status_code == 202
    settled_scene(client, legacy["id"], "ready")
    wait_for_job_exit(legacy["id"])


def test_cancellation_blocks_retry_until_worker_stops_and_preserves_completed_scene(client, monkeypatch):
    monkeypatch.setenv("STUDIO_AUTHOR_API_KEY", "test-only-key-never-sent")
    scene = draft(client)
    entered, release = threading.Event(), threading.Event()
    def delayed_author(*args):
        result = compiled_result()
        entered.set()
        assert release.wait(3), "Test failed to release the author worker"
        # Deliberately ignore cancellation as a slow provider might, so the
        # API must discard this completed result after cancellation.
        return result
    monkeypatch.setattr(authoring_api, "author_scene", delayed_author)
    try:
        assert client.post(f"/api/scenes/{scene['id']}/generate").status_code == 202
        assert entered.wait(3)
        cancelled = client.post(f"/api/scenes/{scene['id']}/cancel").json()
        assert cancelled["status"] == "cancelled"
        retry = client.post(f"/api/scenes/{scene['id']}/generate")
        assert retry.status_code == 409
        assert "still stopping" in retry.json()["detail"]
    finally:
        release.set()
        wait_for_job_exit(scene["id"])
    stopped = client.get(f"/api/scenes/{scene['id']}").json()
    assert stopped["status"] == "cancelled"
    assert "artifact_id" not in stopped
    monkeypatch.setattr(authoring_api, "author_scene", lambda *args: compiled_result())
    assert client.post(f"/api/scenes/{scene['id']}/generate").status_code == 202
    ready = settled_scene(client, scene["id"], "ready")
    wait_for_job_exit(scene["id"])
    assert client.post(f"/api/scenes/{scene['id']}/cancel").json() == ready
    assert client.post(f"/api/scenes/{scene['id']}/generate").status_code == 409
    assert client.get(f"/api/scenes/{scene['id']}").json() == ready


@pytest.mark.parametrize("stale_status", ["queued", "running"])
def test_startup_interrupts_stale_jobs_and_preserves_draft(stale_status):
    with TestClient(app) as first:
        reference = upload_photo(first)
        scene = draft(first, image_ids=[reference["id"]])
    storage.update("scene", scene["id"], status=stale_status,
        progress=[{"message": "Job stopped when the server exited."}])
    with TestClient(app) as restarted:
        recovered = restarted.get(f"/api/scenes/{scene['id']}").json()
        assert recovered["status"] == "interrupted"
        assert recovered["prompt"] == scene["prompt"]
        assert recovered["image_ids"] == [reference["id"]]
        assert recovered["progress"] == [{"message": "Job stopped when the server exited."}]
        assert "restarted" in recovered["error"]
        assert restarted.get(reference["url"]).status_code == 200
        with authoring_api.job_lock:
            assert not authoring_api.jobs
