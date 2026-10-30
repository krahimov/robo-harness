import base64
import io
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from studio import codex_author as author, author_providers, storage
from studio.authoring import AuthorCancelled, AuthorError
from studio.images import save_image


def build(**changes):
    program = {"name": "Factory scene", "description": "Pick the box and place it in the tray",
        "assumptions": ["Estimated tabletop scale."], "scene": {}, **changes}
    return json.dumps({"action": "build", "program_json": json.dumps(program), "summary": "Propose the layout."})


def finish(summary="The simulation placement checks passed."):
    return json.dumps({"action": "finish", "program_json": "", "summary": summary})


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False

    def __call__(self, folder, instructions):
        self.folder = folder
        self.instructions = instructions
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def run(self, prompt, photos, event):
        self.calls.append({"brief": json.loads(prompt), "photos": [(label, path.read_bytes()) for label, path in photos]})
        return next(self.responses), {"input_tokens": 11, "output_tokens": 7}


@pytest.fixture
def draft(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(author, "rendered_frames", lambda artifact, indexes: [
        {"t": artifact["frames"][i]["t"], "image": base64.b64encode(b"rendered-camera").decode()}
        for i in indexes])
    return {"name": "Factory operation", "prompt": "Use the photo to create a pick and place scene", "image_ids": []}


def test_codex_scene_loop_sends_references_and_actual_simulator_evidence(draft):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "orange").save(buffer, format="PNG")
    photo = save_image("factory.png", "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    draft["image_ids"] = [photo["id"]]
    session = FakeSession([build(), finish()])
    parent = {"name": "Previous scene"}
    result = author.author_scene_codex(draft, parent, lambda _: None, threading.Event(), session_factory=session)
    assert result["verification"]["status"] == "pass"
    assert result["artifact"]["frames"] and result["spec"]["waypoints"]
    assert result["usage"] == {"input_tokens": 22, "output_tokens": 14}
    assert len(session.calls[0]["photos"]) == 1
    assert len(session.calls[1]["photos"]) == 4
    assert session.calls[0]["brief"]["previous_scene"] == parent
    assert session.calls[1]["photos"][1][1] == b"rendered-camera"
    report = session.calls[1]["brief"]["history"][0]["simulation"]
    assert report["compiled"] and report["camera_available"]
    assert all(outcome["status"] == "pass" for outcome in report["checks"])
    assert "Scene program JSON schema" in session.instructions
    assert "cannot execute commands or read files" in session.instructions
    assert session.closed and not session.folder.exists()


def test_codex_invalid_revision_keeps_runnable_candidate(draft):
    session = FakeSession([build(), build(scene={"cube_x": 20}), finish()])
    result = author.author_scene_codex(draft, None, lambda _: None, threading.Event(), session_factory=session)
    assert result["builds"] == 1
    assert "preserves runnable candidate 1" in result["summary"]
    report = session.calls[-1]["brief"]["history"][-1]["simulation"]
    assert not report["compiled"] and report["retained_candidate"] == 1


def test_codex_does_not_call_executable_proposals(draft):
    session = FakeSession([build(python="import os"), finish()])
    with pytest.raises(AuthorError, match="did not build"):
        author.author_scene_codex(draft, None, lambda _: None, threading.Event(), session_factory=session)
    assert not session.calls[1]["brief"]["history"][0]["simulation"]["compiled"]


def test_codex_build_limit_is_enforced(draft):
    session = FakeSession([build()] * 4)
    with pytest.raises(AuthorError, match="three simulation attempts"):
        author.author_scene_codex(draft, None, lambda _: None, threading.Event(), session_factory=session)
    assert session.calls[-1]["brief"]["builds_remaining"] == 0
    assert len(session.calls[-1]["brief"]["history"]) == 3


@pytest.mark.parametrize("response", [finish(), "not JSON", '{"action":"shell","command":"pwd"}'])
def test_codex_missing_or_invalid_build_is_not_success(draft, response):
    with pytest.raises(AuthorError):
        author.author_scene_codex(draft, None, lambda _: None, threading.Event(), session_factory=FakeSession([response]))


def test_codex_camera_failure_preserves_recording_and_is_disclosed(draft, monkeypatch):
    def camera_failure(*_):
        raise subprocess.TimeoutExpired("renderer", 45)
    monkeypatch.setattr(author, "rendered_frames", camera_failure)
    session = FakeSession([build(), finish("State checks pass; visual inspection unavailable.")])
    result = author.author_scene_codex(draft, None, lambda _: None, threading.Event(), session_factory=session)
    assert result["artifact"]["frames"]
    assert not result["camera_available"]
    assert "numerical state checks only" in result["summary"]
    assert session.calls[-1]["photos"] == []


def test_codex_cancel_before_start_has_no_model_or_reference_access(draft):
    event = threading.Event()
    event.set()
    draft["image_ids"] = ["missing-reference"]
    with pytest.raises(AuthorCancelled):
        author.author_scene_codex(draft, None, lambda _: None, event,
            session_factory=lambda *_: pytest.fail("Cancelled jobs must not start Codex"))


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    executable = tmp_path / "codex-test"
    executable.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
folder = pathlib.Path(args[args.index('--cd') + 1])
scenario = json.loads((folder / 'scenario.json').read_text())
prompt = sys.stdin.read()
(folder / 'request.json').write_text(json.dumps({'args': args, 'prompt': prompt, 'env': dict(os.environ)}))
if scenario.get('wait'):
    (folder / 'started').write_text('yes')
    time.sleep(30)
if scenario.get('error'):
    print(json.dumps({'type':'error','message':scenario['error']}), flush=True)
    sys.exit(1)
if scenario.get('incomplete'):
    print(json.dumps({'type':'turn.started'}), flush=True)
    sys.exit(0)
reply = pathlib.Path(args[args.index('--output-last-message') + 1])
reply.write_text(json.dumps({'action':'finish','program_json':'','summary':'Ready'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':4,'output_tokens':3}}), flush=True)
""", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(author_providers, "codex_binary", lambda: str(executable))
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"configured": True})
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "scenario.json").write_text("{}")
    return folder


def test_cli_reuses_login_without_forwarding_backend_secrets(fake_cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-api-key")
    monkeypatch.setenv("CODEX_API_KEY", "do-not-send-codex-key")
    monkeypatch.setenv("STUDIO_JUDGE_TOKEN", "do-not-send-judge-token")
    monkeypatch.setenv("CODEX_HOME", "/existing/codex/state")
    photo = fake_cli / "reference.jpg"
    photo.write_bytes(b"normalized-photo")
    with author.CodexSession(fake_cli, "scene instructions") as session:
        reply, usage = session.run("Describe a factory", [("Factory photo", photo)], threading.Event())
    request = json.loads((fake_cli / "request.json").read_text())
    assert all(key not in request["env"] for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "STUDIO_JUDGE_TOKEN"))
    assert request["env"]["CODEX_HOME"] == "/existing/codex/state"
    assert request["env"]["HOME"] == os.environ["HOME"]
    args = request["args"]
    assert args[args.index("--model") + 1] == "gpt-6-astra"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert all(flag in args for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"))
    assert "project_doc_max_bytes=0" in args
    assert "features.shell_tool=false" in args and "features.plugins=false" in args
    assert 'forced_login_method="chatgpt"' in args
    assert args[args.index("--image") + 1] == str(photo)
    assert "Factory photo" in request["prompt"]
    assert json.loads(reply)["summary"] == "Ready"
    assert usage == {"input_tokens": 4, "output_tokens": 3}


@pytest.mark.parametrize("raw,message", [
    ("unit-test-secret model_not_found private factory details", "No alternate model"),
    ("unit-test-secret usage_limit private factory details", "Codex subscription reached"),
    ("unit-test-secret authentication private factory details", "fresh ChatGPT sign-in"),
    ("unit-test-secret unexpected argument private factory details", "Update Codex"),
    ("unit-test-secret private factory details", "could not complete"),
])
def test_cli_errors_are_safe_and_actionable(fake_cli, raw, message):
    (fake_cli / "scenario.json").write_text(json.dumps({"error": raw}))
    with author.CodexSession(fake_cli, "instructions") as session:
        with pytest.raises(AuthorError, match=message) as error:
            session.run("brief", [], threading.Event())
    assert "unit-test-secret" not in str(error.value)
    assert "private factory" not in str(error.value)


def test_cli_requires_completed_turn(fake_cli):
    (fake_cli / "scenario.json").write_text('{"incomplete":true}')
    with author.CodexSession(fake_cli, "instructions") as session:
        with pytest.raises(AuthorError, match="no completed scene response"):
            session.run("brief", [], threading.Event())


@pytest.mark.parametrize("cancel", [True, False])
def test_cli_cancellation_and_deadline_stop_process(fake_cli, monkeypatch, cancel):
    (fake_cli / "scenario.json").write_text('{"wait":true}')
    event = threading.Event()
    if cancel:
        timer = threading.Timer(0.4, event.set)
        timer.start()
    else:
        monkeypatch.setattr(author, "TURN_TIMEOUT_SECONDS", 0.4)
    try:
        with author.CodexSession(fake_cli, "instructions") as session:
            start = time.monotonic()
            with pytest.raises(AuthorCancelled if cancel else AuthorError):
                session.run("brief", [], event)
            assert session.process is None
            assert time.monotonic() - start < 5
    finally:
        if cancel:
            timer.cancel()


def test_cli_refuses_non_subscription_sign_in(fake_cli, monkeypatch):
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"configured": False})
    with pytest.raises(AuthorError, match="Sign in with ChatGPT"):
        with author.CodexSession(fake_cli, "instructions"):
            pytest.fail("A non-subscription login must not start authoring")


def test_codex_unsupported_operation_is_declined_without_a_build(draft):
    draft["prompt"] = "Pour water into the cup"
    session = FakeSession([json.dumps({"action": "decline", "program_json": "",
        "summary": "Pouring needs a liquid simulation."})])
    with pytest.raises(AuthorError, match="declined.*liquid"):
        author.author_scene_codex(draft, None, lambda message: None if "Reading" in message else pytest.fail(message),
            threading.Event(), session_factory=session)
    assert len(session.calls) == 1
    assert "Other deformable solids, liquids" in session.instructions
    assert 'action="decline"' in session.instructions
