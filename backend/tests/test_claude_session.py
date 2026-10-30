import base64
import json
import os
import subprocess
import threading
import time

import pytest

from studio import astra_runs, author_providers, claude_session, main, storage
from studio.adapters import astra
from studio.authoring import AuthorCancelled, AuthorError
from studio.codex_author import SceneReply, author_scene_codex
from studio.codex_judge import JudgeReply, codex_judge
from studio.schemas import ExperimentCreate, JudgeSpec
from studio.workflows import pen_pouch_scene


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    executable = tmp_path / "claude-test"
    executable.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
folder = pathlib.Path.cwd()
scenario = json.loads((folder / 'scenario.json').read_text())
stdin = sys.stdin.read()
(folder / 'request.json').write_text(json.dumps({'args': args, 'stdin': stdin, 'env': dict(os.environ)}))
if scenario.get('wait'):
    time.sleep(30)
print(json.dumps({'type': 'system', 'subtype': 'init', 'model': args[args.index('--model') + 1]}), flush=True)
if scenario.get('error'):
    print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': True, 'result': scenario['error']}), flush=True)
    sys.exit(1)
if scenario.get('silent'):
    sys.exit(0)
reply = scenario.get('reply', {'action': 'finish', 'program_json': '', 'summary': 'Ready'})
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': '',
    'structured_output': reply, 'usage': {'input_tokens': 4, 'cache_read_input_tokens': 6, 'output_tokens': 3}}), flush=True)
""", encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setattr(author_providers, "claude_binary", lambda: str(executable))
    monkeypatch.setattr(author_providers, "claude_status", lambda: {"id": "claude", "configured": True, "message": "Signed in."})
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "scenario.json").write_text("{}")
    return folder


def test_claude_uses_subscription_login_isolated_and_without_api_keys(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-send-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-openai-key")
    monkeypatch.setenv("STUDIO_JUDGE_TOKEN", "do-not-send-judge-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "subscription-oauth-token")
    photo = fake_claude / "reference.jpg"
    photo.write_bytes(b"\xff\xd8normalized-photo")
    with claude_session.ClaudeSession(fake_claude, "scene instructions") as session:
        reply, usage = session.run("Describe a factory", [("Factory photo", photo)], threading.Event())
    request = json.loads((fake_claude / "request.json").read_text())
    assert all(key not in request["env"] for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "STUDIO_JUDGE_TOKEN"))
    assert request["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "subscription-oauth-token"
    assert request["env"]["HOME"] == os.environ["HOME"]
    args = request["args"]
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert all(flag in args for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"))
    assert json.loads(args[args.index("--json-schema") + 1]) == SceneReply.model_json_schema()
    assert (fake_claude / "scene-author.md").read_text() == "scene instructions"
    assert "subscription-oauth-token" not in json.dumps(args)
    message = json.loads(request["stdin"])["message"]["content"]
    assert message[0] == {"type": "text", "text": "Describe a factory"}
    assert "Factory photo" in message[1]["text"]
    assert message[2]["source"] == {"type": "base64", "media_type": "image/jpeg",
        "data": base64.b64encode(b"\xff\xd8normalized-photo").decode()}
    assert json.loads(reply)["summary"] == "Ready"
    assert usage == {"input_tokens": 10, "output_tokens": 3}


def test_claude_reply_schema_follows_the_caller(fake_claude):
    with claude_session.ClaudeSession(fake_claude, "judge", reply_model=JudgeReply) as session:
        args = session.command()
    assert json.loads(args[args.index("--json-schema") + 1]) == JudgeReply.model_json_schema()


@pytest.mark.parametrize("raw,message", [
    ("unit-test-secret not_found_error model private factory details", "No alternate model"),
    ("unit-test-secret rate_limit private factory details", "Claude subscription reached"),
    ("unit-test-secret Not logged in · Please run /login private factory details", "fresh sign-in"),
    ("unit-test-secret private factory details", "could not complete"),
])
def test_claude_errors_are_safe_and_actionable(fake_claude, raw, message):
    (fake_claude / "scenario.json").write_text(json.dumps({"error": raw}))
    with claude_session.ClaudeSession(fake_claude, "instructions") as session:
        with pytest.raises(AuthorError, match=message) as error:
            session.run("brief", [], threading.Event())
    assert "unit-test-secret" not in str(error.value) and "private factory" not in str(error.value)


def test_claude_requires_a_result(fake_claude):
    (fake_claude / "scenario.json").write_text('{"silent": true}')
    with claude_session.ClaudeSession(fake_claude, "instructions") as session:
        with pytest.raises(AuthorError, match="could not complete"):
            session.run("brief", [], threading.Event())


@pytest.mark.parametrize("cancel", [True, False])
def test_claude_cancellation_and_deadline_stop_process(fake_claude, monkeypatch, cancel):
    (fake_claude / "scenario.json").write_text('{"wait": true}')
    event = threading.Event()
    timer = threading.Timer(0.4, event.set)
    if cancel:
        timer.start()
    else:
        monkeypatch.setattr(claude_session, "TURN_TIMEOUT_SECONDS", 0.4)
    try:
        with claude_session.ClaudeSession(fake_claude, "instructions") as session:
            start = time.monotonic()
            with pytest.raises(AuthorCancelled if cancel else AuthorError):
                session.run("brief", [], event)
            assert session.process is None and time.monotonic() - start < 5
    finally:
        timer.cancel()


def test_claude_refuses_without_subscription_sign_in(fake_claude, monkeypatch):
    monkeypatch.setattr(author_providers, "claude_status", lambda: {"configured": False})
    with pytest.raises(AuthorError, match="claude auth login"):
        with claude_session.ClaudeSession(fake_claude, "instructions"):
            pytest.fail("Claude must not start without subscription sign-in")


@pytest.mark.parametrize("state,configured", [
    ({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"}, True),
    ({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"}, True),
    ({"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty", "apiKeySource": "ANTHROPIC_API_KEY"}, False),
    ({"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "bedrock"}, False),
    ({"loggedIn": False, "authMethod": "none"}, False),
])
def test_claude_status_accepts_only_subscription_auth(real_claude_status, monkeypatch, state, configured):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-api-key")
    monkeypatch.setattr(author_providers, "claude_binary", lambda: "/test/claude")
    def run(command, **kwargs):
        assert command == ["/test/claude", "auth", "status", "--json"]
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, json.dumps(state), "")
    monkeypatch.setattr(author_providers.subprocess, "run", run)
    status = author_providers.claude_status()
    assert status["configured"] is configured
    assert "private-api-key" not in str(status)


def test_claude_status_without_cli_explains_setup(real_claude_status, monkeypatch):
    monkeypatch.setattr(author_providers, "claude_binary", lambda: None)
    status = author_providers.claude_status()
    assert not status["configured"] and "claude setup-token" in status["message"]


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def __call__(self, folder, instructions):
        self.instructions = instructions
        return self

    def __enter__(self): return self
    def __exit__(self, *_): pass

    def run(self, prompt, photos, event):
        self.calls.append(json.loads(prompt))
        return next(self.responses), {"input_tokens": 1, "output_tokens": 1}


def test_scene_author_selects_claude_session_and_names_the_model(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    chosen = []
    session = FakeSession([json.dumps({"action": "decline", "program_json": "", "summary": "Needs a hinged lid."})])
    monkeypatch.setattr(author_providers, "session_factory", lambda provider, reply_model=None: chosen.append(provider) or session)
    messages = []
    draft = {"name": "Box", "prompt": "Close the box lid", "image_ids": []}
    with pytest.raises(AuthorError, match="Claude Opus 5.5 declined"):
        author_scene_codex(draft, None, messages.append, threading.Event(), provider="claude")
    assert chosen == ["claude"] and "Claude Code subscription" in messages[0]


def test_claude_judge_pins_model_and_reports_provider(monkeypatch):
    artifact = {"task": {"requirements": [{"id": "placed", "required": True}]},
        "frames": [{"t": 0, "cube": [0, 0, .1]}, {"t": 1, "cube": [.2, 0, .05]}], "xml": "<mujoco/>"}
    judge = {"kind": "vlm", "provider": "claude", "model": "claude-opus-5-5", "rubric": "Inspect placement.",
        "tools": ["inspect_state"], "require_review": False}
    verdict = {"outcomes": [{"id": "placed", "status": "pass", "explanation": "Placed.", "timestamps": [1]}],
        "summary": "Placed.", "needs_review": False}
    session = FakeSession([json.dumps({"action": "finish", "tool": "none", "start": 0, "end": 0, "verdict_json": json.dumps(verdict)})])
    result = codex_judge(artifact, judge, session_factory=session)
    assert result["provider"] == "claude" and result["model"] == "claude-opus-5-5"
    with pytest.raises(ValueError, match="claude-opus-5-5"):
        codex_judge(artifact, {**judge, "model": "gpt-6-astra"}, session_factory=session)
    with pytest.raises(ValueError, match="Claude Code judges"):
        JudgeSpec(name="Opus judge", kind="vlm", rubric="Inspect all outcomes.", provider="claude", model="other")


@pytest.fixture
def judge_record(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    return storage.save("judge", {**JudgeSpec(name="Test judge", rubric="Measure all task outcomes.").model_dump(), "version": 1})


def prepare(judge, **kwargs):
    return main.prepare_experiment(ExperimentCreate(name="Pen trial", scene=pen_pouch_scene(), adapter="astra",
        demo_ids=[], judge_id=judge["id"], task_id="pen-into-open-pouch", episodes=2, **kwargs))


def test_experiment_actor_switch_records_the_selected_model(judge_record, monkeypatch):
    monkeypatch.setattr(author_providers, "codex_status", lambda: {"configured": False, "message": "Run codex login."})
    with pytest.raises(Exception) as missing:
        prepare(judge_record, actor_provider="claude")
    assert getattr(missing.value, "status_code", None) == 409
    monkeypatch.setattr(author_providers, "claude_status", lambda: {"configured": True, "message": "Signed in."})
    prepared = prepare(judge_record, actor_provider="claude")
    assert prepared["actor_provider"] == "claude"
    assert prepared["contract"]["actor"] == "claude-opus-5-5" and prepared["contract"]["actor_provider"] == "claude"
    with pytest.raises(Exception):
        prepare(judge_record, actor_provider="codex")


def test_planner_feedback_and_recording_use_the_selected_actor(monkeypatch, tmp_path):
    chosen = []
    class Session:
        def __init__(self, folder, instructions): pass
    monkeypatch.setattr(author_providers, "session_factory", lambda provider, reply_model=None: chosen.append(provider) or Session)
    monkeypatch.setattr(astra, "rendered_frames", lambda artifact, indexes, camera: [{"image": base64.b64encode(b"jpg").decode()}])
    frame = {"t": 0, "ee": [0, 0, .2], "euler_xyz": [3.14159, 0, 0], "closed": False, "objects": {}, "cube": [0, 0, .04]}
    class Simulator:
        source, frequency = "<mujoco/>", 5
        def snapshot(self): return dict(frame)
        def step(self, action): return {**frame, "ee": [action["x"], action["y"], action["z"]]}
    monkeypatch.setattr(astra, "make_simulator", lambda scene: Simulator())
    from studio.task_api import bind_task, get_task
    scene = pen_pouch_scene()
    task = bind_task(scene, get_task("pen-into-open-pouch"))
    initial = {"scene": scene.model_dump(), "xml": "<mujoco/>", "frames": [frame]}
    planner = astra.Planner(tmp_path, {"task": task, "instructions": "", "actor_provider": "claude"}, initial, [], threading.Event())
    assert chosen == ["claude"] and planner.provider == "claude"
    program = astra.Program(motions=[astra.Motion(target="gripper", offset=[0, 0, 0], yaw=0, closed=False, seconds=.2)] * 4)
    artifact = astra.execute(scene, task, program, threading.Event(), provider="claude")
    assert artifact["execution_error"] is None
    assert artifact["model"]["model"] == "claude-opus-5-5" and artifact["model"]["provider"] == "claude"
