"""Regression tests for temporal pose alignment; no checkpoint inference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from runtime import ICRTRuntime, current_pose_action


@pytest.mark.parametrize("rotation", [[0, 0, 0], [np.pi, 0, 0], [.3, .2, -.4]])
def test_rebase_preserves_local_translation_rotation_and_continuous_gripper(rotation):
    previous = np.array([.4, .1, .3, .1, -.2, .3, 0.])
    current = np.array([.5, -.1, .2, *rotation, 0.])
    delta_position = np.array([.01, -.02, .03])
    delta_rotation = Rotation.from_euler("XYZ", [.04, .03, -.02])
    old = Rotation.from_euler("XYZ", previous[3:6])
    new = Rotation.from_euler("XYZ", current[3:6])
    upstream = np.r_[previous[:3] + old.apply(delta_position), (old * delta_rotation).as_euler("XYZ"), .37]
    corrected = current_pose_action(upstream, previous, current)
    np.testing.assert_allclose(corrected[:3], current[:3] + new.apply(delta_position), atol=1e-10)
    np.testing.assert_allclose(Rotation.from_euler("XYZ", corrected[3:6]).as_matrix(),
        (new * delta_rotation).as_matrix(), atol=1e-10)
    assert corrected[6] == .37
    np.testing.assert_array_equal(current_pose_action(upstream, None, current), upstream)


def test_runtime_zero_delta_holds_current_pose_and_reset_clears_reference():
    source = Path(__file__).resolve().parents[2] / ".data/models/icrt-source"
    if not source.exists():
        pytest.skip("Run prepare.py to check the pinned upstream evaluator")
    sys.path.insert(0, str(source))
    from icrt.models.policy.icrt import ICRT
    from icrt.data.utils import euler_to_rot_6d, rot_6d_to_euler

    class ZeroDeltaPolicy(torch.nn.Module):
        reset = ICRT.reset
        get_action_eval = ICRT.get_action_eval

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.num_pred_steps = 16
            self.use_delta_action = True

        def forward_inference(self, observations, start_pos):
            # Replace only the network prediction. The real pinned evaluator
            # still maintains history and converts local deltas to poses.
            delta = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., .37])
            return delta[None, None].repeat(1, 16, 1), None

    policy = ZeroDeltaPolicy()
    runtime = ICRTRuntime.__new__(ICRTRuntime)
    runtime.model = policy
    runtime.reset([])

    def wrapper(side, wrist, proprio, **kwargs):
        value = proprio[0]
        rotation = euler_to_rot_6d(value[None, 3:6]).reshape(-1)
        observation = {"observation": torch.zeros(1, 1, 1),
            "proprio": torch.tensor(np.r_[value[:3], rotation, value[6]][None, None], dtype=torch.float32),
            "action": None}
        action = policy.get_action_eval(observation, **kwargs).numpy()
        return np.r_[action[:3], rot_6d_to_euler(action[3:9]).reshape(-1), action[9]]

    runtime.wrapper = wrapper
    for value in [[.4, 0, .3, 0, 0, 0, 0], [.5, .02, .35, .1, -.2, .3, 0], [.53, .01, .32, -.2, .1, .4, 0]]:
        action = runtime.act({"side": None, "wrist": None, "proprio": value})
        np.testing.assert_allclose(action[:3], value[:3], atol=1e-7)
        np.testing.assert_allclose(Rotation.from_euler("XYZ", action[3:6]).as_matrix(),
            Rotation.from_euler("XYZ", value[3:6]).as_matrix(), atol=1e-7)
        assert action[6] == pytest.approx(.37)
        # The evaluator still caches the unchanged zero local delta.
        np.testing.assert_allclose(policy.last_action[0, 0, :3].numpy(), 0, atol=1e-7)
    runtime.reset([])
    assert runtime.previous_proprio is None
    value = [.6, -.1, .2, .2, -.1, -.3, 0]
    action = runtime.act({"side": None, "wrist": None, "proprio": value})
    np.testing.assert_allclose(action[:3], value[:3], atol=1e-7)


def test_executed_command_replaces_raw_history_in_previous_pose_frame():
    source=Path(__file__).resolve().parents[2]/'.data/models/icrt-source'
    if not source.exists():pytest.skip('Pinned source unavailable')
    sys.path.insert(0,str(source))
    from icrt.data.utils import rot_6d_to_rot_mat
    runtime=ICRTRuntime.__new__(ICRTRuntime)
    runtime.device=torch.device('cpu');runtime.executed_feedback_steps=0
    runtime.model=type('Policy',(),{'use_delta_action':True,'last_action':torch.ones(1,1,11)})()
    previous=np.array([.4,.1,.3,np.pi,0,.3,0.])
    runtime.previous_proprio=previous.copy()
    executed=np.array([.42,.09,.28,np.pi,.1,.2,.65])
    runtime.set_executed_action(executed)
    cached=runtime.model.last_action[0,0].numpy()
    rotation=Rotation.from_euler('XYZ',previous[3:6])
    np.testing.assert_allclose(cached[:3],rotation.inv().apply(executed[:3]-previous[:3]),atol=1e-7)
    expected=(rotation.inv()*Rotation.from_euler('XYZ',executed[3:6])).as_matrix()
    np.testing.assert_allclose(rot_6d_to_rot_mat(cached[None,3:9])[0],expected,atol=1e-6)
    assert cached[9]==pytest.approx(.65) and cached[10]==0
    np.testing.assert_array_equal(runtime.previous_proprio,previous)
    runtime.previous_proprio=None
    with pytest.raises(ValueError,match='previous observation'):runtime.set_executed_action(executed)
