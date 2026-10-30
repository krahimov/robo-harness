import math

import numpy as np
import pytest

from studio.adapters.arms import allegro_available
from studio.authoring import SceneProgram, compile_program
from studio.robots import DEXTEROUS_HAND, robot
from studio.schemas import Scene
from studio.skills import compile_skills
from studio.task_api import get_task

pytestmark = pytest.mark.skipif(not allegro_available(), reason="Run integrations/icrt/prepare_arms.py to install the Allegro hand")
HAND = "ur5e-allegro-v1"


def test_hand_robot_is_registered_with_tool_points():
    spec = robot(HAND)
    assert DEXTEROUS_HAND in spec.capabilities and spec.available()
    assert spec.tool_offset("pinch") != (0, 0, 0) and robot("franka-panda-v1").tool_offset("pinch") == (0., 0., 0.)


@pytest.mark.parametrize("task_id", ["lift-and-hold", "block-to-tray"])
def test_power_grasp_lifts_and_places_the_block(task_id):
    task = get_task(task_id)
    spec, artifact, verification = compile_program(SceneProgram(name="Grasp", description="Grasp.", scene=Scene(embodiment=HAND)), task, HAND)
    assert verification["status"] == "pass", verification["outcomes"]
    assert any(frame["attached"] for frame in artifact["frames"])
    assert artifact["frames"][-1]["grasp_kind"].startswith("dexterous hand")


@pytest.mark.parametrize("task_id", ["close-box-flaps", "open-drawer"])
def test_articulated_presets_use_the_hand_push_point(task_id):
    task = get_task(task_id)
    spec, artifact, verification = compile_program(SceneProgram(name="Hand", description="Hand.", scene=task["suggested_scene"]), task, HAND)
    assert verification["status"] == "pass", verification["outcomes"]


def test_tool_points_shift_the_commanded_grasp_site():
    scene = Scene(embodiment=HAND, include_cube=False, include_tray=False)
    point = [-.1, 0, .2]
    grasp = compile_skills(scene, [{"kind": "move_to", "target": {"point": point}}, {"kind": "grasp"}])[0]
    pinch = compile_skills(scene, [{"kind": "move_to", "target": {"point": point}, "tool": "pinch"}, {"kind": "grasp"}])[0]
    # Top-down hand: the pinch point sits 3.3 cm beyond the grasp site along the downward-pointing site Z axis.
    assert [grasp[a] for a in "xyz"] == pytest.approx(point, abs=1e-6)
    assert pinch["z"] == pytest.approx(point[2] + .033, abs=1e-4)


def test_hand_cad_is_exported_by_link():
    import json, struct
    from studio.adapters.arms import UR5E_ALLEGRO
    from studio.robot_visuals import arm_visual_glb
    payload = arm_visual_glb(UR5E_ALLEGRO)
    document = json.loads(payload[20:20 + struct.unpack("<I", payload[12:16])[0]])
    bodies = {node["extras"]["body"] for node in document["nodes"]}
    assert {"ur5e_hand_palm", "ur5e_hand_th_tip", "ur5e_wrist_3_link"} <= bodies
