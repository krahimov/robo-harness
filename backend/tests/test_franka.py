import math

import mujoco
import numpy as np
import pytest

from studio.adapters.franka import FrankaSimulator, available, euler_xyz, rotation_matrix
from studio.adapters.simulation import default_waypoints, record
from studio.judges import state_judge
from studio.schemas import Scene


@pytest.mark.parametrize("angles", [[.2, .3, .4], [3.14, .2, -.3], [-2.8, -.5, 1.2]])
def test_franka_rotation_matches_intrinsic_xyz_round_trip(angles):
    assert euler_xyz(rotation_matrix(angles)) == pytest.approx(angles)


@pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py to install the licensed assets")
def test_franka_actuates_full_pose_with_joint_limits_and_no_weld():
    sim = FrankaSimulator(Scene(embodiment="franka-panda-v1"))
    orientation = [3.14, .15, .2]
    for _ in range(30):
        frame = sim.step({"x": -.10, "y": -.03, "z": .28, "euler_xyz": orientation,
            "closed": False, "gripper": .25})
    assert frame["embodiment"] == "franka-panda-v1"
    assert frame["grasp_kind"] == "two-finger contact"
    assert frame["ee"] == pytest.approx([-.10, -.03, .28], abs=.003)
    assert rotation_matrix(frame["euler_xyz"]) == pytest.approx(rotation_matrix(orientation), abs=.01)
    assert all(sim.model.eq_type != mujoco.mjtEq.mjEQ_WELD)
    assert all(sim.arm_limits[:, 0] <= sim.data.qpos[sim.arm_qpos])
    assert all(sim.data.qpos[sim.arm_qpos] <= sim.arm_limits[:, 1])
    assert sim.model.camera("wrist").id >= 0 and sim.model.camera("overview").id >= 0
    assert all("mesh_vertices" not in geom for geom in frame["geoms"])


@pytest.mark.skipif(not available(), reason="Run integrations/icrt/prepare_franka.py to install the licensed assets")
def test_franka_demonstration_uses_contact_grasp_and_passes_placement():
    scene = Scene(embodiment="franka-panda-v1")
    artifact = record(scene, default_waypoints(scene))
    assert artifact["frequency"] == 15
    assert all(frame["embodiment"] == "franka-panda-v1" for frame in artifact["frames"])
    assert any(frame["attached"] and frame["cube"][2] > .1 for frame in artifact["frames"])
    assert state_judge(artifact, {"require_review": False})["status"] == "pass"


@pytest.mark.skipif(not available(), reason="Native assets unavailable")
def test_wrist_pinch_target_is_inside_icrt_center_crop_after_arm_motion():
    sim=FrankaSimulator(Scene(embodiment='franka-panda-v1'))
    camera=sim.model.camera('wrist').id
    for target in ([ -.22,-.08,.28],[-.1,.04,.14]):
        for _ in range(15):sim.step(dict(zip(('x','y','z'),target))|{'closed':False})
        # Independent camera projection: MuJoCo cameras look down local -Z.
        local=sim.data.cam_xmat[camera].reshape(3,3).T@(sim.data.site('ee').xpos-sim.data.cam_xpos[camera])
        depth=-local[2]
        crop_half_height=depth*np.tan(np.deg2rad(sim.model.cam_fovy[camera])/2)*224/248
        assert depth>0
        assert np.max(np.abs(local[:2]))<crop_half_height*.05
        # Being in the crop is insufficient if the camera looks through the hand.
        origin=sim.data.cam_xpos[camera]
        direction=sim.data.site('ee').xpos-origin
        distance=np.linalg.norm(direction)
        hit=np.array([-1],dtype=np.int32)
        first_hit=mujoco.mj_ray(sim.model,sim.data,origin,direction/distance,None,1,-1,hit)
        assert first_hit<0 or first_hit>=distance-.005
    # Former parallel mounting misses both axes after square cropping.
    old_delta=np.array([0.,0.,.1034])-np.array([.06,-.06,.035])
    old_crop_half=old_delta[2]*np.tan(np.deg2rad(65)/2)*224/248
    assert np.all(np.abs(old_delta[:2])>old_crop_half)
