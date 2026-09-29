import numpy as np
import pytest

from reference_eval import metrics, validate_split


def test_reference_split_rejects_leakage_and_truncation():
    context = {"id": "a", "role": "context", "frames": 117}
    query = {"id": "b", "role": "query", "frames": 101}
    assert validate_split({"episodes": [context, query]}) == (context, [query])
    with pytest.raises(ValueError, match="distinct"):
        validate_split({"episodes": [context, {**query, "id": "a"}]})
    with pytest.raises(ValueError, match="exceed"):
        validate_split({"episodes": [context, {**query, "frames": 400}]})


def test_metrics_use_rotation_distance_and_balanced_gripper_classes():
    target = np.zeros((4, 7))
    predicted = target.copy()
    target[:, 3] = np.pi
    predicted[:, 3] = -np.pi
    target[0, 6] = 1
    predicted[:, 0] = .03
    result = metrics(predicted, target, np.zeros_like(target))
    assert result["translation_mean_l2_m"] == pytest.approx(.03)
    assert result["rotation_mean_geodesic_deg"] == pytest.approx(0, abs=1e-9)
    assert result["gripper_agreement"] == .75
    assert result["gripper_balanced_accuracy"] == .5


def test_recorded_action_uses_its_own_past_pose():
    import sys
    from pathlib import Path
    import torch
    from types import SimpleNamespace
    from reference_eval import recorded_previous_action
    source = Path(__file__).resolve().parents[2] / ".data/models/icrt-source"
    if not source.exists():
        pytest.skip("Pinned upstream source not installed")
    sys.path.insert(0, str(source))
    encoded = recorded_previous_action(
        SimpleNamespace(device=torch.device("cpu")),
        {"proprio": [.5, .1, .2, 0, 0, 0, 0]},
        [.52, .1, .2, 0, 0, 0, 1])
    np.testing.assert_allclose(encoded[0, 0, :3], [.02, 0, 0], atol=1e-6)
    assert encoded[0, 0, -2] == 1
    assert encoded[0, 0, -1] == 0
