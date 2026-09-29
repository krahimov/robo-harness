import base64
import json

import pytest

from studio.authoring import AuthorError
from studio.codex_judge import JudgeSession, codex_judge
from studio.judges import evidence_tool, validate_verdict


@pytest.fixture
def artifact():
    return {"task": {"requirements": [{"id": "placed", "required": True}, {"id": "style", "required": False}]},
        "frames": [{"t": 0, "cube": [0, 0, .1], "geoms": ["large"], "qpos": ["raw"]},
            {"t": 1, "cube": [.2, 0, .05], "geoms": ["large"], "qpos": ["raw"]}], "xml": "<mujoco/>"}


@pytest.fixture
def judge():
    return {"kind": "vlm", "provider": "codex", "model": "gpt-6-astra", "rubric": "Inspect placement and request review if unclear.",
        "tools": ["inspect_state", "replay_frames"], "require_review": False}


def verdict():
    return {"outcomes": [{"id": "placed", "status": "pass", "explanation": "Block placed.", "timestamps": [1]},
        {"id": "style", "status": "fail", "explanation": "Optional style criterion failed.", "timestamps": [1]}],
        "summary": "Placement passed, optional style did not.", "needs_review": False}


def reply(action="finish", tool="none", start=0, end=0, result=None):
    return json.dumps({"action": action, "tool": tool, "start": start, "end": end,
        "verdict_json": json.dumps(result or verdict()) if action == "finish" else ""})


def sessions(replies, calls):
    class FakeSession:
        def __init__(self, folder, instructions):
            calls.append({"instructions": instructions})
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def run(self, prompt, photos, event):
            calls.append({"brief": json.loads(prompt), "photos": [(label, path.read_bytes()) for label, path in photos]})
            return next(replies), {"input_tokens": 12, "output_tokens": 3}
    return FakeSession


def test_codex_judge_requests_state_and_camera_then_obeys_required_outcomes(artifact, judge, monkeypatch):
    from studio import judges
    monkeypatch.setattr(judges, "rendered_frames", lambda *args: [{"t": 1, "image": base64.b64encode(b"camera-image").decode()}])
    calls = []
    factory = sessions(iter([reply("inspect", "inspect_state", 0, 1), reply("inspect", "replay_frames", 0, 1), reply()]), calls)
    result = codex_judge(artifact, judge, session_factory=factory)
    assert result["status"] == "pass"  # Optional failure cannot redefine success.
    assert result["provider"] == "codex" and result["model"] == "gpt-6-astra"
    assert result["usage"] == {"input_tokens": 36, "output_tokens": 9}
    assert [item["tool"] for item in result["tool_trace"]] == ["inspect_state", "replay_frames"]
    assert calls[-1]["photos"][0][1] == b"camera-image"
    state = calls[2]["brief"]["history"][0]["evidence"]
    assert all("geoms" not in frame and "qpos" not in frame for frame in state["frames"])
    assert state["coverage"]["matched_frames"] == 2
    assert "human-owned" in calls[0]["instructions"]


def test_disabled_tool_or_no_tool_never_becomes_supported_pass(artifact, judge):
    judge["tools"] = ["replay_frames"]
    calls = []
    result = codex_judge(artifact, judge, session_factory=sessions(iter([reply("inspect", "inspect_state", 0, 1), reply()]), calls))
    assert result["status"] == "unknown" and result["needs_review"]
    assert all(item["status"] == "unknown" for item in result["outcomes"])
    assert result["tool_trace"] == [{"tool": "inspect_state", "ok": False}]
    assert result["summary"] != verdict()["summary"]


@pytest.mark.parametrize("change", ["duplicate", "nan", "boolean", "outside"])
def test_invalid_verdict_does_not_get_saved_as_success(artifact, judge, change):
    result = verdict()
    if change == "duplicate": result["outcomes"][1]["id"] = "placed"
    if change == "nan": result["outcomes"][0]["timestamps"] = [float("nan")]
    if change == "boolean": result["outcomes"][0]["timestamps"] = [True]
    if change == "outside": result["outcomes"][0]["timestamps"] = [2]
    with pytest.raises(ValueError):
        validate_verdict(result, artifact, judge, [{"ok": True}])


def test_model_is_exact_and_turn_loop_is_bounded(artifact, judge):
    with pytest.raises(ValueError, match="exact model"):
        codex_judge(artifact, {**judge, "model": "other"})
    calls = []
    with pytest.raises(AuthorError, match="turn limit"):
        codex_judge(artifact, judge, session_factory=sessions(iter([reply("inspect", "inspect_state", 0, 1)] * 6), calls))
    assert len(calls) == 7  # Instructions and exactly six model turns.


def test_judge_session_replaces_only_reply_schema(tmp_path):
    session = JudgeSession(tmp_path, "Judge instructions")
    schema = json.loads(session.schema.read_text())
    assert schema["properties"]["action"]["enum"] == ["inspect", "finish"]
    assert "program_json" not in schema["properties"]
    assert session.instructions.read_text() == "Judge instructions"


def test_state_statistics_include_an_event_between_returned_samples(judge):
    artifact = {"frames": [{"t": i / 100, "cube": [0, 0, -.1 if i == 1 else .05],
        "cube_speed": .9 if i == 1 else 0} for i in range(101)]}
    output, _, trace = evidence_tool(artifact, judge, "inspect_state", 0, 1)
    assert all(frame["t"] != .01 for frame in output["frames"])
    assert output["coverage"] == {"matched_frames": 101, "returned_frames": 40, "first_timestamp": 0, "last_timestamp": 1}
    assert output["statistics_over_all_matched_frames"]["cube"] == {"minimum_center_height": -.1, "maximum_speed": .9}
    assert trace["frames_inspected"] == 40
