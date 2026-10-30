import base64
import io
import json
import subprocess
import threading

import httpx
import pytest
from PIL import Image

from studio import authoring, storage
from studio.authoring import AuthorError, AuthorCancelled, SceneProgram, author_scene
from studio.images import save_image


def program(**changes):
    return {"name": "Photo scene", "description": "Move the box into the tray", "assumptions": ["Estimated tabletop scale."], "scene": {}, **changes}


def call(args, number=1):
    return {"type": "function_call", "id": f"fc_{number}", "call_id": f"call_{number}", "name": "simulate_scene", "arguments": json.dumps(args)}


def message(text="The scene runs and placement passed."):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    monkeypatch.delenv("STUDIO_AUTHOR_API_KEY", raising=False)
    monkeypatch.setattr(authoring, "rendered_frames", lambda artifact, indexes: [{"t": artifact["frames"][i]["t"], "image": "test-jpeg"} for i in indexes])
    return {"name": "Factory operation", "prompt": "Use the photo to create a pick and place scene", "image_ids": []}


def factory_for(handler):
    return lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


def test_responses_sends_photos_exact_model_and_continues_tools(setup):
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), "orange").save(buf, format="PNG")
    photo = save_image("factory.png", "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    setup["image_ids"] = [photo["id"]]
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer unit-test-secret"
        assert payload["model"] == "gpt-6-astra"
        assert payload["store"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["parallel_tool_calls"] is False
        if len(requests) == 1:
            assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
            assert payload["tools"][0]["name"] == "simulate_scene"
            return httpx.Response(200, json={"output": [{"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "encrypted-test"}, call(program())], "usage": {"input_tokens": 12, "output_tokens": 7}})
        assert any(item.get("encrypted_content") == "encrypted-test" for item in payload["input"])
        tool_result = next(item for item in payload["input"] if item.get("type") == "function_call_output")
        assert tool_result["call_id"] == "call_1"
        report = json.loads(tool_result["output"])
        assert report["compiled"] and report["camera_available"]
        assert all(outcome["status"] == "pass" for outcome in report["checks"])
        assert any(part.get("type") == "input_image" for item in payload["input"][1:] for part in item.get("content", []) if isinstance(part, dict))
        return httpx.Response(200, json={"output": [message()], "usage": {"input_tokens": 20, "output_tokens": 8}})
    result = author_scene(setup, None, lambda _: None, threading.Event(), client_factory=factory_for(handler))
    assert len(requests) == 2
    assert result["verification"]["status"] == "pass"
    assert result["spec"]["waypoints"]
    assert result["artifact"]["frames"]
    assert result["usage"] == {"input_tokens": 32, "output_tokens": 15}
    assert "unit-test-secret" not in json.dumps(result)


def test_invalid_repair_retains_last_runnable_scene(setup):
    rounds = 0
    def handler(request):
        nonlocal rounds
        rounds += 1
        output = [call(program())] if rounds == 1 else [call(program(scene={"cube_x": 20}), 2)] if rounds == 2 else [message()]
        if rounds == 3:
            reports = [json.loads(item["output"]) for item in json.loads(request.content)["input"] if item.get("type") == "function_call_output"]
            assert reports[-1]["compiled"] is False
            assert reports[-1]["retained_candidate"] == 1
        return httpx.Response(200, json={"output": output})
    result = author_scene(setup, None, lambda _: None, threading.Event(), client_factory=factory_for(handler))
    assert result["spec"]["scene"]["cube_x"] == -.22
    assert "preserves runnable candidate 1" in result["summary"]


@pytest.mark.parametrize("status,message_text", [(401, "key was rejected"), (403, "cannot access"), (404, "No alternate model"), (429, "rate or billing limit"), (500, "HTTP 500")])
def test_provider_error_is_actionable_and_does_not_expose_body(setup, status, message_text):
    with pytest.raises(AuthorError, match=message_text) as error:
        author_scene(setup, None, lambda _: None, threading.Event(), client_factory=factory_for(lambda request: httpx.Response(status, json={"error": "unit-test-secret private factory contents"})))
    assert "unit-test-secret" not in str(error.value)
    assert "private factory" not in str(error.value)


def test_missing_key_cancellation_and_missing_build_are_not_success(setup, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(AuthorError, match="Set OPENAI_API_KEY"):
        author_scene(setup, None, lambda _: None, threading.Event())
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    event = threading.Event()
    event.set()
    with pytest.raises(AuthorCancelled):
        author_scene(setup, None, lambda _: None, event, client_factory=factory_for(lambda _: pytest.fail("Cancellation must skip the request")))
    with pytest.raises(AuthorError, match="did not build"):
        author_scene(setup, None, lambda _: None, threading.Event(), client_factory=factory_for(lambda _: httpx.Response(200, json={"output": [message("Done!")]})))


def test_camera_timeout_is_reported_without_discarding_recording(setup, monkeypatch):
    def unavailable(*args):
        raise subprocess.TimeoutExpired("renderer", 45)
    monkeypatch.setattr(authoring, "rendered_frames", unavailable)
    rounds = 0
    def handler(request):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            return httpx.Response(200, json={"output": [call(program())]})
        report = next(json.loads(item["output"]) for item in json.loads(request.content)["input"] if item.get("type") == "function_call_output")
        assert not report["camera_available"]
        return httpx.Response(200, json={"output": [message("State checks pass. Camera comparison unavailable.")]})
    result = author_scene(setup, None, lambda _: None, threading.Event(), client_factory=factory_for(handler))
    assert result["artifact"]["frames"]
    assert result["camera_available"] is False


def test_scene_program_rejects_executable_fields_and_unbounded_programs():
    with pytest.raises(ValueError):
        SceneProgram(**program(python="import os"))
    with pytest.raises(ValueError):
        SceneProgram(**program(assumptions=["x" * 601]))
    with pytest.raises(ValueError):
        SceneProgram(**program(waypoints=[{"x": 0, "y": 0, "z": .2, "seconds": 3}] * 11))


def test_unsupported_operation_is_declined_instead_of_substituted(setup):
    setup["prompt"] = "Pour water into the cup"
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "Other deformable solids, liquids" in payload["instructions"]
        assert [tool["name"] for tool in payload["tools"]] == ["simulate_scene", "decline_request"]
        decline = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "decline_request",
            "arguments": json.dumps({"reason": "Pouring needs a liquid simulation."})}
        return httpx.Response(200, json={"output": [decline]})
    with pytest.raises(AuthorError, match="declined.*liquid") as error:
        author_scene(setup, None, lambda message: None if "Reading" in message else pytest.fail(message), threading.Event(), client_factory=factory_for(handler))
    assert len(requests) == 1
    assert "unit-test-secret" not in str(error.value)
