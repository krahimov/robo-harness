import math

import pytest
from pydantic import ValidationError

from studio.adapters.franka import available
from studio.adapters.simulation import Simulator, record
from studio.articulated import fold_towel_scene
from studio.authoring import SceneProgram, compile_program
from studio.schemas import Scene
from studio.skills import compile_skills
from studio.task_api import bind_task, get_task, measure_task

franka = pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py")
TOWEL = {"name": "towel", "position": [-.1, 0, .004], "count": [5, 4], "spacing": .02}


def test_cloth_schema_validation():
    with pytest.raises(ValidationError, match="pins a vertex index beyond its grid"):
        Scene(cloths=[{**TOWEL, "pinned": [20]}])
    with pytest.raises(ValidationError, match="Cloth names must be unique"):
        Scene(cloths=[TOWEL], objects=[{"id": "towel", "position": [-.2, .1, .026]}])
    with pytest.raises(ValidationError, match="at least one vertex that is not pinned"):
        Scene(cloths=[{**TOWEL, "count": [3, 3], "pinned": list(range(9))}])
    with pytest.raises(ValidationError, match="must sit over the table"):
        Scene(cloths=[{**TOWEL, "position": [.9, 0, .01]}])


def test_cloth_state_rests_on_the_table_and_reaches_the_browser():
    frame = Simulator(Scene(include_cube=False, include_tray=False, cloths=[TOWEL])).snapshot()
    cloth = frame["cloths"]["towel"]
    assert len(cloth["vertices"]) == 20 and cloth["count"] == [5, 4] and cloth["attached"] is False
    assert min(v[2] for v in cloth["vertices"]) > -.001  # Contact margin keeps vertices above the surface.
    mesh = next(g for g in frame["geoms"] if g["name"] == "cloth_towel")
    assert mesh["type"] == "mesh" and len(mesh["mesh_vertices"]) == 20 and len(mesh["mesh_faces"]) == 4 * 3 * 4
    assert "cloths" not in Simulator(Scene()).snapshot()


def test_idealized_pinch_holds_the_nearest_unpinned_vertex_until_release():
    # The preset's 2 cm board puts the corner within reach of the gantry's lowest point.
    board = fold_towel_scene()
    scene = board.model_copy(update={"embodiment": "cartesian-gripper-v1", "cloths": [board.cloths[0].model_copy(update={"pinned": [80]})]})
    sim = Simulator(scene)
    corner = sim.snapshot()["cloths"]["towel"]["vertices"][0]
    for _ in range(30):
        sim.step({"x": corner[0], "y": corner[1], "z": .025, "closed": False})
    sim.step({"x": corner[0], "y": corner[1], "z": .025, "closed": True})
    assert sim.snapshot()["cloths"]["towel"]["pinched_vertex"] == 0
    for _ in range(20):
        frame = sim.step({"x": corner[0], "y": corner[1], "z": .12, "closed": True})
    assert frame["cloths"]["towel"]["vertices"][0][2] > .08 and frame["cloths"]["towel"]["attached"]
    frame = sim.step({"x": corner[0], "y": corner[1], "z": .12, "closed": False})
    assert frame["cloths"]["towel"]["attached"] is False and frame["cloths"]["towel"]["pinched_vertex"] is None


@pytest.mark.parametrize("embodiment", ["cartesian-gripper-v1", pytest.param("franka-panda-v1", marks=franka)])
def test_fold_towel_preset_passes(embodiment):
    task = get_task("fold-towel")
    spec, artifact, verification = compile_program(SceneProgram(name="Fold", description="Fold.", scene=task["suggested_scene"]), task, embodiment)
    assert verification["status"] == "pass", verification["outcomes"]
    assert max(f["cloths"]["towel"]["vertices"][0][2] for f in artifact["frames"]) > .08
    assert "idealized pinch" in artifact["frames"][-1]["cloth_grasp"]


def test_fold_is_not_passed_without_folding():
    scene = fold_towel_scene().model_copy(update={"embodiment": "cartesian-gripper-v1"})
    task = bind_task(scene, get_task("fold-towel"))
    idle = compile_skills(scene, [{"kind": "move_to", "target": {"cloth": "towel", "vertex": 0}, "offset": [0, 0, .1]}, {"kind": "grasp"}, {"kind": "release"}])
    result = measure_task({**record(scene, idle), "task": task}, {"require_review": False})
    assert {o["id"]: o["status"] for o in result["outcomes"]}["corner_folded"] == "fail"


def test_cloth_requirement_and_target_validation():
    scene = Scene(include_cube=False, include_tray=False, cloths=[TOWEL])
    task = {"name": "Fold", "version": 1, "schema_version": 2, "requirements": [{"id": "fold", "label": "Fold", "definition": "Fold.",
        "required": True, "check": "cloth_points_near", "object_id": "towel", "vertex_a": 0, "vertex_b": 40, "max_distance": .05}]}
    with pytest.raises(ValueError, match="needs cloth towel with vertices 0 and 40"):
        bind_task(scene, task)
    with pytest.raises(ValueError, match="unknown cloth vertex towel\\[40\\]"):
        compile_skills(scene, [{"kind": "move_to", "target": {"cloth": "towel", "vertex": 40}}, {"kind": "grasp"}])
    with pytest.raises(ValidationError, match="cloth target needs both cloth and vertex"):
        compile_skills(scene, [{"kind": "move_to", "target": {"cloth": "towel"}}, {"kind": "grasp"}])
