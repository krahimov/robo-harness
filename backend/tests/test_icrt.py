import base64
import importlib.util
import io
import json
import math
import sys
import types
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio.adapters import icrt
from studio.adapters.models import build_model


def frame(index=0, embodiment="cartesian-gripper-v1"):
    return {"t": index / 20, "ee": [-.22, -.08, .25], "euler_xyz": [0, 0, 0],
        "gripper": 0., "closed": False, "embodiment": embodiment, "qpos": [index]}


def demonstration():
    return {"id": "demo1", "artifact": {"xml": "fixture", "frequency": 20,
        "frames": [frame(i) for i in range(181)],
        "actions": [{"x": -.2, "y": .1, "z": .15, "closed": True} for _ in range(180)]}}


def test_icrt_proprio_uses_only_pose_with_explicit_projection():
    sample = {**frame(), "cube": [999, 999, 999], "attached": True}
    assert icrt.pose(sample) == pytest.approx([.43, -.08, .25, math.pi, 0, 0, 0])
    sample["euler_xyz"] = [.1, .2, .3]
    converted = icrt.pose(sample, projected=False)
    restored, rotation = icrt.flange_to_tip(converted[:3], converted[3:6])
    assert restored == pytest.approx([.43, -.08, .25])
    assert rotation == pytest.approx([.1, .2, .3])


def test_context_resampling_is_bounded_and_preserves_order():
    demo = demonstration()["artifact"]
    indices = icrt.demonstration_indices(demo, 64)
    assert len(indices) == 64
    assert indices == sorted(set(indices))
    assert indices[0] == 0 and indices[-1] == 179
    full = icrt.demonstration_indices(demo, 256)
    assert len(full) == 135  # Nine seconds at the upstream 15 Hz rate.
    assert all(index < len(demo["actions"]) for index in full)


def test_invalid_context_alignment_is_rejected():
    demo = demonstration()["artifact"]
    demo["frames"].pop()
    with pytest.raises(ValueError, match="aligned"):
        icrt.demonstration_indices(demo, 64)


def test_changed_camera_mount_is_recorded_without_rewriting_old_context(adapter_setup):
    old=demonstration()
    actor=icrt.ICRTModel([old],{**frame(), 'wrist_camera_version':'side-mounted-pinch-v3'},adapter_setup.simulator)
    assert actor.metadata['context_wrist_camera_versions']==[{'id':'demo1','version':'legacy-unversioned'}]
    assert any('mounts differ' in x for x in actor.metadata['limitations'])
    assert 'wrist_camera_version' not in old['artifact']['frames'][0]
    actor.close()


def test_legacy_recordings_without_wrist_camera_fail_with_recovery_message():
    icrt.validate_camera_source('<mujoco><worldbody><camera name="overview"/><camera name="wrist"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="Re-record"):
        icrt.validate_camera_source('<mujoco><worldbody><camera name="overview"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="valid MuJoCo"):
        icrt.validate_camera_source("invalid")


@pytest.fixture
def adapter_setup(monkeypatch):
    requests = []
    cameras = []
    class Camera:
        def __init__(self):
            self.closed = False
            cameras.append(self)
        def render(self, xml, frames):
            return [{"side": "side-jpeg", "wrist": "wrist-jpeg"} for _ in frames]
        def close(self):
            self.closed = True
    def handler(request):
        payload = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, payload))
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "session123", "metadata": {"policy_sha256": "verifiedhash"}})
        if request.method == "DELETE":
            return httpx.Response(200, json={"closed": True})
        return httpx.Response(200, json={"action": [.48, -.06, .22, .2, .3, .4, .8]})
    client = httpx.Client
    monkeypatch.setattr(icrt, "CameraClient", Camera)
    monkeypatch.setattr(icrt.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    return types.SimpleNamespace(requests=requests, cameras=cameras, simulator=types.SimpleNamespace(source="fixture"))


def test_icrt_adapter_sends_sensorimotor_context_and_records_projection(adapter_setup):
    fixture = adapter_setup
    actor = icrt.ICRTModel([demonstration()], frame(), fixture.simulator)
    request = fixture.requests[0][2]
    assert request["protocol"] == "robot-harness-icrt-v1"
    demo = request["demonstrations"][0]
    assert set(demo["frames"][0]) == {"side", "wrist", "proprio"}
    assert demo["actions"][0] == pytest.approx([.45, .1, .15, math.pi, 0, 0, 1])
    result = actor.act(frame())
    assert "euler_xyz" not in result and result["closed"]
    assert result["x"] == pytest.approx(-.18)  # Four-centimeter intervention is recorded.
    assert actor.predictions[-1]["rotation_discarded"]
    assert actor.metadata["clipped_predictions"] == 1
    request_count = len(fixture.requests)
    assert actor.act(frame(1)) == result  # No extra prediction before the next 15 Hz tick.
    assert len(fixture.requests) == request_count
    actor.close()
    assert fixture.cameras[0].closed
    assert fixture.requests[-1][0] == "DELETE"


def test_native_icrt_keeps_orientation_and_refuses_mismatched_demos(adapter_setup):
    native = {**frame(embodiment="franka-panda-v1"), "robot_base": [-.65, 0, 0]}
    with pytest.raises(ValueError, match="embodiments must match"):
        icrt.ICRTModel([demonstration()], native, adapter_setup.simulator)
    actor = icrt.ICRTModel([], native, adapter_setup.simulator)
    action = actor.act(native)
    _, expected_rotation = icrt.flange_to_tip([0, 0, 0], [.2, .3, .4])
    assert action["euler_xyz"] == pytest.approx(expected_rotation)
    assert action["gripper"] == .8
    assert actor.metadata["projection"] is False
    assert actor.predictions[-1]["rotation_discarded"] is False
    actor.close()


def test_native_demo_preserves_continuous_gripper_targets(adapter_setup):
    native = {**frame(embodiment="franka-panda-v1"), "robot_base": [-.45, -.15, 0]}
    sample = demonstration()
    sample["artifact"]["frames"] = [{**f, "embodiment": "franka-panda-v1", "robot_base": native["robot_base"]}
        for f in sample["artifact"]["frames"]]
    for action in sample["artifact"]["actions"]:
        action.update(gripper=.6, euler_xyz=[.1, .2, .3])
    actor = icrt.ICRTModel([sample], native, adapter_setup.simulator)
    supplied = adapter_setup.requests[0][2]["demonstrations"][0]["actions"][0]
    position, rotation = icrt.flange_to_tip(supplied[:3], supplied[3:6])
    assert [*position, *rotation, supplied[6]] == pytest.approx([.25, .25, .15, .1, .2, .3, .6])
    actor.close()


def test_native_rounded_15hz_timestamps_make_exactly_150_predictions(adapter_setup):
    native = {**frame(embodiment="franka-panda-v1"), "robot_base": [-.45, -.15, 0]}
    actor = icrt.ICRTModel([], native, adapter_setup.simulator)
    for step in range(150):
        actor.act({**native, "t": round(step / 15, 5)})
    requests = [path for method, path, _ in adapter_setup.requests if path.endswith("/act")]
    assert len(requests) == len(actor.predictions) == 150
    assert actor.predictions[2]["t"] == .13333
    assert actor.metadata["adapter_version"] == "icrt-studio-v4"
    actor.close()


def test_model_factory_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="Unknown"):
        build_model("invented", [], {}, {}, "", None)


@pytest.fixture
def worker_client(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.demonstrations = []
        def metadata(self):
            return {"fixed_weights": True, "gradient_updates": 0, "supports_executed_action_feedback": True}
        def reset(self, demos):
            if any(len(demo["frames"]) != len(demo["actions"]) for demo in demos):
                raise ValueError()
            self.demonstrations = demos
        def act(self, observation, previous_action=None):
            self.previous_action = previous_action
            assert isinstance(observation["side"], Image.Image)
            assert isinstance(observation["wrist"], Image.Image)
            return [.5, 0, .2, .1, .2, .3, .7]
    monkeypatch.setitem(sys.modules, "runtime", types.SimpleNamespace(ICRTRuntime=FakeRuntime))
    path = Path(__file__).resolve().parents[2] / "integrations/icrt/worker.py"
    spec = importlib.util.spec_from_file_location("icrt_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("STUDIO_ICRT_TOKEN", raising=False)
    with TestClient(module.create_app(FakeRuntime)) as client:
        yield client


def camera_observation():
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "gray").save(output, format="JPEG")
    image = base64.b64encode(output.getvalue()).decode()
    return {"side": image, "wrist": image, "proprio": [.5, 0, .2, math.pi, 0, 0, 0]}


def test_worker_lifecycle_zero_demo_native_pose_and_session_isolation(worker_client):
    client = worker_client
    assert client.get("/health").json()["fixed_weights"]
    payload = {"protocol": "robot-harness-icrt-v1", "demonstrations": []}
    session = client.post("/sessions", json=payload).json()["session_id"]
    assert client.post("/sessions", json=payload).status_code == 409
    response = client.post(f"/sessions/{session}/act", json={"observation": camera_observation()})
    assert response.json()["action"] == [.5, 0, .2, .1, .2, .3, .7]
    assert client.delete(f"/sessions/{session}").status_code == 200
    assert client.post(f"/sessions/{session}/act", json={"observation": camera_observation()}).status_code == 404
    assert client.post("/sessions", json=payload).status_code == 200


def test_worker_rejects_bad_camera_and_hides_validation_inputs(worker_client):
    observation = camera_observation()
    observation["side"] = "private-photo-text"
    payload = {"protocol": "robot-harness-icrt-v1", "demonstrations": [{"id": "x", "frames": [observation], "actions": [[.5, 0, .2, 0, 0, 0, 0]]}]}
    response = worker_client.post("/sessions", json=payload)
    assert response.status_code == 422 and "private-photo" not in response.text
    observation["proprio"] = ["private-data"]
    response = worker_client.post("/sessions/unknown/act", json={"observation": observation})
    assert response.status_code == 422 and "private-data" not in response.text


def test_worker_authentication_does_not_return_token(worker_client, monkeypatch):
    monkeypatch.setenv("STUDIO_ICRT_TOKEN", "unit-test-private-token")
    response = worker_client.get("/health")
    assert response.status_code == 401 and "unit-test-private-token" not in response.text
    assert worker_client.get("/health", headers={"Authorization": "Bearer unit-test-private-token"}).status_code == 200


def test_adapter_feedback_uses_clipped_executed_command(adapter_setup):
    actor=icrt.ICRTModel([],frame(),adapter_setup.simulator)
    actor.executed_feedback=True  # Capability negotiated by a new worker.
    first=actor.act(frame())
    actor.act({**frame(), 't':1/15})
    calls=[body for method,path,body in adapter_setup.requests if path.endswith('/act')]
    assert 'previous_action' not in calls[0]
    assert calls[1]['previous_action'][:3]==pytest.approx([first['x']+.65,first['y'],first['z']])
    assert calls[1]['previous_action'][6]==float(first['closed'])
    assert calls[1]['previous_action'][:3]!=pytest.approx([.48,-.06,.22])
    assert actor.metadata['executed_feedback_steps']==1
    actor.close()


def test_worker_forwards_and_validates_executed_action(worker_client):
    c=worker_client;sid=c.post('/sessions',json={'protocol':'robot-harness-icrt-v1','demonstrations':[]}).json()['session_id']
    value=[.4,.1,.2,3.14,0,0,.7]
    r=c.post(f'/sessions/{sid}/act',json={'observation':camera_observation(),'previous_action':value})
    assert r.status_code==200 and c.app.state.runtime.previous_action==value
    assert c.post(f'/sessions/{sid}/act',json={'observation':camera_observation(),'previous_action':[.4]*6}).status_code==422
    assert c.post(f'/sessions/{sid}/act',json={'observation':camera_observation(),'previous_action':value[:-1]+[1.2]}).status_code==422


def test_worker_rejects_feedback_when_runtime_does_not_support_it(worker_client, monkeypatch):
    c=worker_client
    monkeypatch.setattr(c.app.state.runtime, 'metadata', lambda: {'supports_executed_action_feedback':False})
    sid=c.post('/sessions',json={'protocol':'robot-harness-icrt-v1','demonstrations':[]}).json()['session_id']
    request={'observation':camera_observation(),'previous_action':[.4,.1,.2,3.14,0,0,.7]}
    assert c.post(f'/sessions/{sid}/act',json=request).status_code==422
    assert c.post(f'/sessions/{sid}/act',json={'observation':camera_observation()}).status_code==200
