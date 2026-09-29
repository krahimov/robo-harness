import copy
import math

import mujoco
import numpy as np
import pytest

from studio.adapters.franka import rotation_matrix, available
from studio.adapters.simulation import geometry_extent, interpolate_orientation, xml
from studio.schemas import Scene, SceneObject
from studio.task_api import bind_task, get_task, measure_task
from studio.workflows import pen_pouch_scene, pen_pouch_waypoints


def test_orientation_does_not_spin_between_equivalent_euler_angles():
    middle = interpolate_orientation([-math.pi + 1e-5, 0, 0], [math.pi, 0, math.pi/2], .5)
    expected = rotation_matrix([math.pi, 0, math.pi/4])
    np.testing.assert_allclose(rotation_matrix(middle), expected, atol=1e-5)


def test_horizontal_cylinder_extent_matches_its_long_axis():
    model = mujoco.MjModel.from_xml_string(xml(pen_pouch_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(geometry_extent(model, data, "object_pen_geom"), [.008, .07, .008], atol=1e-8)
    # A 14 cm pen pointing across the 12 cm opening must not fit.
    assert geometry_extent(model, data, "object_pen_geom")[1] > .06


def test_pen_requires_native_rotation_and_valid_cylinder_geometry():
    with pytest.raises(ValueError, match="requires"):
        bind_task(pen_pouch_scene().model_copy(update={"embodiment": "cartesian-gripper-v1"}), get_task("pen-into-open-pouch"))
    with pytest.raises(ValueError, match="equal radii"):
        SceneObject(id="pen", shape="cylinder", position=[0, 0, .1], size=[.008, .02, .07])
    assert Scene().objects == []


def test_drop_into_pouch_does_not_count_as_held_insertion():
    scene = pen_pouch_scene()
    task = bind_task(scene, get_task("pen-into-open-pouch"))
    def frame(t, z, held, extent=(.07, .008, .008)):
        return {"t": t, "objects": {"pen": {"position": [.1, .1, z], "extent": list(extent), "attached": held, "speed": 0}}}
    frames = [frame(0, .2, True), frame(1, .021, False), frame(2, .021, False)]
    artifact = {"task": task, "frames": frames}
    verdict = measure_task(artifact, {"require_review": True})
    assert next(o for o in verdict["outcomes"] if o["id"] == "pen_inside")["status"] == "pass"
    assert next(o for o in verdict["outcomes"] if o["id"] == "pen_inserted")["status"] == "fail"
    inserted = copy.deepcopy(artifact)
    inserted["frames"].insert(1, frame(.5, .065, True))
    assert measure_task(inserted, {"require_review": True})["status"] == "pass"
    inserted["frames"][1]["objects"]["pen"]["extent"] = [.008, .07, .008]
    assert measure_task(inserted, {"require_review": True})["status"] == "fail"


@pytest.mark.skipif(not available(), reason="Menagerie assets not installed")
def test_native_pen_workflow_retains_contact_and_rests_above_floor():
    from studio.adapters.simulation import record
    scene = pen_pouch_scene()
    artifact = record(scene, pen_pouch_waypoints(scene))
    artifact["task"] = bind_task(scene, get_task("pen-into-open-pouch"))
    assert measure_task(artifact, {"require_review": True})["status"] == "pass"
    assert artifact["frames"][-1]["objects"]["pen"]["position"][2] > .02
