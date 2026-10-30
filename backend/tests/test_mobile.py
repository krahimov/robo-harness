import pytest

from studio.adapters.arms import TIDYBOT, TidyBotSimulator, tidybot_available
from studio.authoring import SceneProgram, compile_program
from studio.robots import MOBILE_BASE, robot
from studio.schemas import Scene
from studio.task_api import get_task

pytestmark = pytest.mark.skipif(not tidybot_available(), reason="Run integrations/icrt/prepare_arms.py to install TidyBot")
MOBILE = "tidybot-v1"


def test_mobile_robot_stands_on_a_floor_beside_the_table():
    assert MOBILE_BASE in robot(MOBILE).capabilities
    sim = TidyBotSimulator(Scene(embodiment=MOBILE))
    frame = sim.snapshot()
    names = {g["name"] for g in frame["geoms"]}
    assert {"floor", "table_leg_0"} <= names
    assert frame["ik_position_error"] < .002 and frame["ee"] == pytest.approx(list(TIDYBOT.start), abs=.003)


def test_whole_body_ik_drives_the_base_along_the_table():
    sim = TidyBotSimulator(Scene(embodiment=MOBILE))
    base = sim.model.joint("tidy_joint_y").id
    reached = []
    for x in [-.35, .3]:
        for _ in range(40):
            frame = sim.step({"x": x, "y": -.1, "z": .1, "closed": False})
        reached.append((frame["ee"], float(sim.data.qpos[sim.model.jnt_qposadr[base]])))
    (left, base_left), (right, base_right) = reached
    assert left == pytest.approx([-.35, -.1, .1], abs=.003) and right == pytest.approx([.3, -.1, .1], abs=.003)
    # The base is rotated 90°, so its Y slide points along world -X: reaching +X drives the slide negative.
    assert base_right - base_left < -.3


def test_base_stays_outside_the_table_footprint():
    sim = TidyBotSimulator(Scene(embodiment=MOBILE))
    for _ in range(40):
        frame = sim.step({"x": 0, "y": .3, "z": .1, "closed": False})  # Far edge: beyond reach.
    base = sim.model.body("tidy_base_link").id
    edge = max(sim.data.geom_xpos[i][1] + sim.model.geom_aabb[i][4] for i in range(sim.model.ngeom)
        if sim.model.geom_bodyid[i] == base and sim.model.geom_group[i] != 2)
    assert edge < -.42 and frame["ik_position_error"] > .05  # It reports the miss instead of driving under the table.


def test_mobile_pick_and_place_on_the_near_half():
    scene = Scene(embodiment=MOBILE, tray_x=.15, tray_y=-.2)
    spec, artifact, verification = compile_program(SceneProgram(name="Mobile", description="Pick and place.", scene=scene), get_task("block-to-tray"), MOBILE)
    assert verification["status"] == "pass", verification["outcomes"]
