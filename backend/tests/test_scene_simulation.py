import copy
import json

import httpx
import pytest
from pydantic import ValidationError

from studio.adapters.simulation import Simulator, default_waypoints, record
from studio.judges import DEFAULT_RUBRIC, evaluate
from studio.schemas import Conveyor, JudgeSpec, Scene, SceneProp, TASK, Waypoint, task_for_scene


def judge():
    return JudgeSpec(name="Scene checks", rubric=DEFAULT_RUBRIC).model_dump()


def idle(sim, seconds):
    frame = sim.snapshot()
    for _ in range(round(seconds * sim.frequency)):
        frame = sim.step({"x": -0.22, "y": -0.08, "z": 0.28, "closed": False})
    return frame


def test_shifted_tray_binds_motion_and_judges_without_changing_requirements():
    original = copy.deepcopy(TASK)
    scene = Scene(tray_x=0.1, tray_y=0.2)
    artifact = record(scene, default_waypoints(scene))
    assert evaluate(artifact, judge())["status"] == "pass"
    assert artifact["task"] == task_for_scene(scene)
    assert artifact["task"]["requirements"][1:] == TASK["requirements"][1:]
    assert TASK == original
    # The old destination must fail if the authored tray moved elsewhere.
    artifact["scene"] = Scene().model_dump()
    outcomes = evaluate(artifact, judge())["outcomes"]
    assert next(o for o in outcomes if o["id"] == "inside_tray")["status"] == "fail"


@pytest.mark.parametrize("direction,speed", [("x", 0.025), ("y", 0.025), ("x", -0.025)])
def test_conveyor_moves_one_box_and_intercepts_it_with_recorded_actions(direction, speed):
    scene = Scene(conveyor=Conveyor(direction=direction, speed=speed))
    sim = Simulator(scene)
    initial = sim.snapshot()["cube"]
    final = idle(sim, 1)
    axis = 0 if direction == "x" else 1
    assert (final["cube"][axis] - initial[axis]) * (1 if speed > 0 else -1) > 0.01
    assert abs(final["cube"][2] - (scene.conveyor.height + 0.025)) < 0.002
    assert not final["attached"]
    waypoints = default_waypoints(scene)
    for point in waypoints:
        Waypoint(**point)
    artifact = record(scene, waypoints)
    assert any(frame["attached"] for frame in artifact["frames"])
    assert evaluate(artifact, judge())["status"] == "pass"


def test_static_prop_contacts_block_conveyor_motion_and_can_be_disabled():
    wall = SceneProp(name="machine_stop", position=[-0.12, -0.08, 0.07], size=[0.015, 0.06, 0.07])
    blocked = Simulator(Scene(conveyor=Conveyor(), props=[wall]))
    unblocked = Simulator(Scene(conveyor=Conveyor(), props=[wall.model_copy(update={"collidable": False})]))
    blocked_frame, unblocked_frame = idle(blocked, 5), idle(unblocked, 5)
    assert blocked_frame["cube"][0] < -0.15
    assert unblocked_frame["cube"][0] > -0.13
    assert blocked_frame["cube"][0] < unblocked_frame["cube"][0] - 0.025
    geometry = next(geom for geom in blocked_frame["geoms"] if geom["name"] == "prop_machine_stop")
    assert geometry["position"] == wall.position
    assert geometry["type"] == "box"


def test_cylinder_and_color_parameters_reach_recorded_geometry():
    scene = Scene(
        cube_color=[0.2, 0.3, 0.9, 1], table_color=[0.6, 0.5, 0.4, 1],
        props=[SceneProp(name="motor", shape="cylinder", position=[-0.45, 0.2, 0.1], size=[0.05, 0.05, 0.1])],
    )
    geoms = {geom["name"]: geom for geom in Simulator(scene).snapshot()["geoms"]}
    assert geoms["block"]["color"] == pytest.approx(scene.cube_color)
    assert geoms["table"]["color"] == pytest.approx(scene.table_color)
    assert geoms["prop_motor"]["type"] == "cylinder"
    assert geoms["prop_motor"]["size"][:2] == [0.05, 0.1]


@pytest.mark.parametrize("fields", [
    {"cube_color": [1, 0, float("nan"), 1]},
    {"table_color": [1, 0, 0]},
    {"tray_x": 0.8},
    {"conveyor": {"x": 0.5, "length": 0.8}},
    {"props": [{"name": "bad\"/>", "position": [0, 0, 0], "size": [0.1] * 3}]},
    {"props": [{"name": "motor", "position": [0, 0, 9], "size": [0.1] * 3}]},
    {"props": [{"name": "motor", "position": [0, 0, 0.1], "size": [0.1] * 3}] * 2},
    {"props": [{"name": f"part{i}", "position": [0, 0, 0.1], "size": [0.1] * 3} for i in range(25)]},
])
def test_scene_rejects_unbounded_or_ambiguous_geometry(fields):
    with pytest.raises(ValidationError):
        Scene(**fields)


def test_vlm_receives_scene_bound_requirement_snapshot(monkeypatch):
    scene = Scene(tray_x=0.1, tray_y=0.2)
    artifact = record(scene, default_waypoints(scene))
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        task = json.loads(payload["messages"][1]["content"])["task"]
        assert task["parameters"] == {"tray_x": 0.1, "tray_y": 0.2}
        assert "x -0.015–0.215 m" in task["requirements"][0]["definition"]
        result = {"outcomes": [{"id": requirement["id"], "status": "unknown", "explanation": "No evidence retrieved", "timestamps": []} for requirement in task["requirements"]]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    client_type = httpx.Client
    monkeypatch.setenv("STUDIO_JUDGE_ENDPOINT", "http://judge.test/chat/completions")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs))
    assert evaluate(artifact, {**judge(), "kind": "vlm", "model": "test"})["status"] == "unknown"
    assert len(requests) == 1
