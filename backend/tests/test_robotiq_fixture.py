import sys
from pathlib import Path
import numpy as np
import mujoco
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integrations/icrt'))
from robotiq_fixture import RobotiqFranka,pose,action,BASE,FLANGE_FROM_HAND
from robotiq_trial import teacher,scene,metrics
pytestmark=pytest.mark.skipif(not (ROOT/'.data/models/menagerie/robotiq_2f85/2f85.xml').exists(),reason='Optional Robotiq assets absent')

def test_robot_flange_conversion_matches_independent_mujoco_mount():
    sim=RobotiqFranka(scene(71));frame=sim.snapshot();p=pose(frame)
    np.testing.assert_allclose(p[:3],sim.data.body('hand').xpos-BASE,atol=2e-5)
    command,clipped=action(p,frame)
    np.testing.assert_allclose([command[k] for k in ['x','y','z']],frame['ee'],atol=1e-5)
    assert not clipped and frame['embodiment']=='franka-robotiq-2f85-research-v1'
    assert .07<frame['gripper_width_m']<.09 and frame['gripper']<.1
    assert all(sim.model.eq_type!=mujoco.mjtEq.mjEQ_WELD)

def test_robotiq_contact_teacher_passes_without_changing_task_thresholds():
    a=teacher(scene(71));m=metrics(a)
    assert m['verdict']['status']=='pass'
    assert m['grasp_frames']>20 and m['max_cube_z']>.15
    assert len(a['frames'])==len(a['actions'])+1
