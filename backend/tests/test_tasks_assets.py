import copy
import json
import numpy as np

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from studio import storage
from studio.assets import parse_obj
from studio.authoring import SceneProgram, compile_program
from studio.adapters.simulation import Simulator, task_waypoints, record
from studio.main import app
from studio.schemas import Scene, SceneObject, SceneProp
from studio.task_api import get_task, bind_task, measure_task

TETRA = '''v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
'''


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path / "tasks-assets")


def test_two_boxes_are_physically_picked_and_placed_sequentially():
    task = get_task("two-box-sort")
    scene = Scene.model_validate(task["suggested_scene"])
    task = bind_task(scene, task)
    artifact = record(scene, task_waypoints(scene, task))
    artifact["task"] = task
    assert measure_task(artifact, {"require_review": False})["status"] == "pass"
    for id in ("cube", "blue"):
        assert any(frame["objects"][id]["attached"] for frame in artifact["frames"])
        assert not artifact["frames"][-1]["objects"][id]["attached"]
    assert all(sum(obj["attached"] for obj in frame["objects"].values()) <= 1 for frame in artifact["frames"])
    artifact["frames"][-1]["objects"]["blue"]["position"] = [-.3, .2, .026]
    failed = measure_task(artifact, {"require_review": False})
    assert failed["status"] == "fail"
    assert next(outcome for outcome in failed["outcomes"] if outcome["id"] == "blue_inside_region")["status"] == "fail"


def test_human_task_versions_preserve_old_recordings_and_change_future_checks():
    with TestClient(app) as client:
        original = client.post("/api/scenes/template?task_id=lift-and-hold").json()
        assert original["verification"]["status"] == "pass"
        old_task = copy.deepcopy(original["task"])
        body = {key: old_task[key] for key in ["name", "description", "requirements", "operations"]}
        body["parent_id"] = old_task["id"]
        body["requirements"][0].update(min_height=.35, definition="Hold the block center above 0.35 m at episode end.")
        revised = client.post("/api/tasks", json=body)
        assert revised.status_code == 201, revised.text
        task = revised.json()
        assert task["version"] == 2 and task["id"] != old_task["id"]
        assert client.get(f"/api/scenes/{original['id']}").json() == original
        assert client.get("/api/tasks/lift-and-hold").json()["requirements"][0]["min_height"] == .12
        future = client.post(f"/api/scenes/template?task_id={task['id']}").json()
        assert future["verification"]["status"] == "fail"
        assert future["task"]["requirements"][0]["min_height"] == .35
        exported = client.get(f"/api/scenes/{original['id']}/export").json()
        assert exported["task"] == original["task"]
        demo = client.post(f"/api/scenes/{original['id']}/record", json={"name": "Hold example"}).json()
        assert demo["task_id"] == "lift-and-hold"
        assert demo["task"] == original["task"]


def test_task_bound_to_scene_cannot_omit_human_named_object():
    with pytest.raises(ValueError, match="blue"):
        compile_program(SceneProgram(name="Missing blue", description="Only the primary object."), get_task("two-box-sort"))
    program = SceneProgram(name="Two objects", description="Both required objects.", scene=Scene.model_validate(get_task("two-box-sort")["suggested_scene"]))
    _, artifact, verification = compile_program(program, get_task("two-box-sort"))
    assert artifact["task"]["id"] == "two-box-sort"
    assert verification["status"] == "pass"


def test_missing_state_is_unknown_and_optional_failure_does_not_block():
    task = get_task("two-box-sort")
    scene = Scene.model_validate(task["suggested_scene"])
    artifact = record(scene, task_waypoints(scene, task))
    artifact["task"] = bind_task(scene, task)
    for r in artifact["task"]["requirements"]:
        if r["object_id"] == "blue":
            r["required"] = False
    for frame in artifact["frames"]:
        del frame["objects"]["blue"]
    result = measure_task(artifact, {"require_review": False})
    assert result["status"] == "pass"
    assert result["needs_review"]
    assert all(o["status"] == "unknown" for o in result["outcomes"] if o["id"].startswith("blue_"))


def test_obj_import_compiles_embedded_geometry_and_survives_export():
    with TestClient(app) as client:
        response = client.post("/api/assets", json={"name": "Machine housing.obj", "obj": TETRA})
        assert response.status_code == 201, response.text
        asset = response.json()
        assert asset["vertex_count"] == 4 and "vertices" not in asset
        assert client.get("/api/assets").json() == [asset]
        parent = client.post("/api/scenes/template").json()
        program = copy.deepcopy(parent["spec"])
        program["scene"]["props"] = [{"name": "housing", "shape": "mesh", "asset_id": asset["id"], "position": [-.43, .2, .1], "size": [.07, .06, .1], "collidable": False}]
        response = client.post(f"/api/scenes/{parent['id']}/preview", json=program)
        assert response.status_code == 201, response.text
        scene = response.json()
        assert scene["asset_ids"] == [asset["id"]]
        assert scene["assets"][0]["id"] == asset["id"]
        artifact = client.get(f"/api/artifacts/{scene['artifact_id']}").json()
        geom = next(g for g in artifact["frames"][0]["geoms"] if g["name"] == "prop_housing")
        assert geom["type"] == "mesh"
        assert len(geom["mesh_vertices"]) == 4
        assert len(geom["mesh_faces"]) == 4
        world = np.asarray(geom["mesh_vertices"]) @ np.asarray(geom["matrix"]).reshape(3, 3).T + geom["position"]
        assert world.min(axis=0) == pytest.approx([-.5, .14, 0], abs=1e-5)
        assert world.max(axis=0) == pytest.approx([-.36, .26, .2], abs=1e-5)
        assert 'file=' not in artifact["xml"] and '<mesh name="asset_housing" vertex=' in artifact["xml"]
        exported = client.get(f"/api/scenes/{scene['id']}/export").json()
        assert exported["xml"] == artifact["xml"]


@pytest.mark.parametrize("obj", [
    TETRA + "mtllib ../../private.mtl\n",
    TETRA + "include /etc/passwd\n",
    TETRA.replace("v 1 0 0", "v NaN 0 0"),
    TETRA.replace("f 1 3 2", "f 1 99 2"),
    "v 0 0 0\n" * 401,
    TETRA.replace("v 0 0 1", "v 1 1 0"),
])
def test_obj_rejects_external_references_unbounded_or_invalid_geometry(obj):
    with pytest.raises((ValueError, OverflowError)):
        parse_obj(obj)
    with TestClient(app) as client:
        response = client.post("/api/assets", json={"name": "Bad.obj", "obj": obj})
        assert response.status_code == 422
        assert "private.mtl" not in response.text
        assert client.get("/api/assets").json() == []


def test_draft_snapshots_measurements_assets_and_task_without_a_model_call():
    with TestClient(app) as client:
        asset = client.post("/api/assets", json={"name": "Machine.obj", "obj": TETRA}).json()
        draft = client.post("/api/scenes", json={"name": "Measured scene", "prompt": "Recreate the station.", "task_id": "two-box-sort", "measurements": [{"label": "Conveyor width", "value": 40, "unit": "cm"}], "asset_ids": [asset["id"]]})
        assert draft.status_code == 201, draft.text
        saved = draft.json()
        assert saved["task"]["id"] == "two-box-sort"
        assert saved["measurements"][0]["value"] == 40
        assert saved["assets"][0]["id"] == asset["id"]
        assert "vertices" not in saved["assets"][0]
        assert saved["status"] == "draft"


def test_movable_objects_and_mesh_references_are_bounded():
    with pytest.raises(ValidationError):
        SceneObject(id="cube", position=[0, 0, .03])
    with pytest.raises(ValidationError):
        Scene(objects=[{"id": "blue", "position": [0, 0, .03]}] * 2)
    with pytest.raises(ValidationError):
        SceneProp(name="mesh", shape="mesh", asset_id="../../secret", position=[0, 0, .1], size=[.1]*3)
    with pytest.raises(ValidationError):
        SceneProp(name="mesh", shape="mesh", position=[0, 0, .1], size=[.1]*3)
    frame = Simulator(Scene()).snapshot()
    assert frame["euler_xyz"] == [0, 0, 0]
    assert frame["embodiment"] == "cartesian-gripper-v1"
