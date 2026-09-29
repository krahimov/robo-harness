"""Pretrained ICRT inference, retaining the upstream sensorimotor interface."""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = "icrt-runtime-v3"


def current_pose_action(action, previous_proprio, current_proprio):
    """Re-anchor upstream's previous-pose absolute result to the current pose.

    The pinned evaluator predicts for the last observation but converts its
    local delta through the first of the two cached proprioception entries.
    Recover that unchanged local delta, then apply it to the current entry.
    Upstream's cached delta remains correct and no network weights change.
    """
    result = np.asarray(action, dtype=float).copy()
    if previous_proprio is None:
        return result
    previous = np.asarray(previous_proprio, dtype=float)
    current = np.asarray(current_proprio, dtype=float)
    old_rotation = Rotation.from_euler("XYZ", previous[3:6])
    new_rotation = Rotation.from_euler("XYZ", current[3:6])
    change_of_reference = new_rotation * old_rotation.inv()
    result[:3] = current[:3] + change_of_reference.apply(result[:3] - previous[:3])
    result[3:6] = (change_of_reference * Rotation.from_euler("XYZ", result[3:6])).as_euler("XYZ")
    return result


class ConfigLoader(yaml.SafeLoader):
    """Treat upstream dataclass tags as data, never instantiate YAML objects."""


def _configuration_mapping(loader, suffix, node):
    if not suffix.startswith("icrt.util.args."):
        raise ValueError("Unsupported checkpoint configuration tag")
    return loader.construct_mapping(node, deep=True)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _configuration_mapping)


class ICRTRuntime:
    frequency = 15
    max_context_frames = 256
    max_sequence_frames = 512

    def __init__(self, model_dir=None, source_dir=None, device="cpu"):
        self.folder = Path(model_dir or ROOT / ".data/models/icrt")
        source = Path(source_dir or ROOT / ".data/models/icrt-source")
        if device not in {"cpu", "cuda"}:
            raise ValueError("ICRT supports cpu or cuda; its complex rotary embeddings are not ported to MPS")
        if not (self.folder / "manifest.json").exists():
            raise ValueError("Run integrations/icrt/prepare.py before starting the worker")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.set_num_threads(min(8, torch.get_num_threads()))
        sys.path.insert(0, str(source))
        from icrt.util.args import ModelConfig, PolicyConfig, SharedConfig, VisionEncoderConfig
        from icrt.util.model_constructor import model_constructor
        from icrt.models.policy.icrt_wrapper import ICRTWrapper
        import timm
        config = yaml.load((self.folder / "run.yaml").read_text(), Loader=ConfigLoader)
        policy = config["model_cfg"]["policy_cfg"]
        policy.update(load_llama=False, scratch_llama_config=str(source / "config/model_config/custom_transformer.json"))
        vision = config["model_cfg"]["vision_encoder_cfg"]
        vision.update(vision_encoder=str(self.folder / "cross-mae-rtx-vitb.pth"), vision_nonpretrained=True)
        shared = SharedConfig(**config["shared_cfg"])
        # Construction does not fetch any further weights. The separately
        # released vision checkpoint supplies the frozen visual encoder.
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = model_constructor(ModelConfig(policy_cfg=PolicyConfig(**policy),
                vision_encoder_cfg=VisionEncoderConfig(**vision)), shared, train=False)
            checkpoint = torch.load(self.folder / "policy.pth", map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint)
            incompatible = self.model.load_state_dict(state, strict=False)
        missing_policy = [name for name in incompatible.missing_keys if not name.startswith("vision_encoder.")]
        if missing_policy or incompatible.unexpected_keys:
            raise ValueError("ICRT checkpoint does not match the pinned model architecture")
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        # Use the upstream wrapper's observation/action transforms without its
        # unsafe yaml.Loader or hardcoded constructor paths.
        self.wrapper = ICRTWrapper.__new__(ICRTWrapper)
        torch.nn.Module.__init__(self.wrapper)
        self.wrapper.model = self.model
        self.wrapper.device = self.device
        self.wrapper.args = type("RuntimeArgs", (), {"shared_cfg": shared})()
        data_cfg = timm.data.resolve_data_config(self.model.vision_encoder.model.pretrained_cfg)
        self.wrapper.preprocess_PIL = timm.data.create_transform(**data_cfg)
        self.wrapper.preprocess_tensor = self.wrapper.preprocess_PIL
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.context_frames = 0
        self.steps = 0
        self.reset([])

    def metadata(self):
        return {"model": "ICRT compact DROID pretrained", "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()), "fixed_weights": True,
            "gradient_updates": 0, "frequency": self.frequency, "context_frames": self.context_frames,
            "supports_executed_action_feedback": True, "executed_feedback_steps": self.executed_feedback_steps,
            "runtime_adapter_version": ADAPTER_VERSION, "action_reference": "current_observation",
            "upstream_pose_reference_correction": True,
            "source_commit": self.manifest["source_commit"],
            "policy_sha256": self.manifest["files"]["policy.pth"]["sha256"]}

    def _prepare(self, frames, actions=None):
        side = [frame["side"] for frame in frames]
        wrist = [frame["wrist"] for frame in frames]
        proprio = np.asarray([frame["proprio"] for frame in frames], dtype=np.float32)
        values = self.wrapper.prepare_observations(side, wrist, proprio,
            np.asarray(actions, dtype=np.float32) if actions is not None else None)
        return {key: tensor.to(self.device) if tensor is not None else None for key, tensor in values.items()}

    def reset(self, demonstrations):
        self.context_frames = sum(len(demo["frames"]) for demo in demonstrations)
        if self.context_frames > self.max_context_frames:
            raise ValueError("ICRT supports at most 256 demonstration frames in this worker")
        self.model.reset(action_exec_horizon=1)
        self.steps = 0
        self.previous_proprio = None
        self.executed_feedback_steps = 0
        # Chunk vision preprocessing to bound activation memory. This preserves
        # every context frame and its original order; no training occurs.
        with torch.inference_mode():
            for demo in demonstrations:
                if len(demo["frames"]) != len(demo["actions"]):
                    raise ValueError("Each demonstration frame needs its paired action")
                for start in range(0, len(demo["frames"]), 8):
                    self.model.prompt(self._prepare(demo["frames"][start:start+8], demo["actions"][start:start+8]))

    def set_executed_action(self, action):
        """Replace the cached prediction with the command actually sent to the robot."""
        if self.previous_proprio is None:
            raise ValueError("There is no previous observation for executed-action feedback")
        value = np.asarray(action, dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all() or not 0 <= value[6] <= 1:
            raise ValueError("Expected a finite executed pose and normalized gripper command")
        from icrt.data.utils import euler_to_rot_6d, convert_delta_action
        def encode(p):
            return np.r_[p[:3], euler_to_rot_6d(p[None, 3:6]).reshape(-1), p[6]]
        encoded = encode(value)
        if self.model.use_delta_action:
            encoded = convert_delta_action(encoded[None, None], encode(self.previous_proprio)[None, None]).reshape(-1)
        self.model.last_action = torch.tensor(np.r_[encoded, 0.], dtype=torch.float32, device=self.device)[None, None]
        self.executed_feedback_steps += 1

    def act(self, observation, previous_action=None):
        if self.context_frames + self.steps + 2 > self.max_sequence_frames:
            raise ValueError("ICRT context capacity reached; start a new session")
        if previous_action is not None:
            self.set_executed_action(previous_action)
        with torch.inference_mode():
            result = self.wrapper(observation["side"], observation["wrist"],
                np.asarray([observation["proprio"]], dtype=np.float32),
                use_temporal=False, teacher_forcing=False, binary_gripper=False)
        self.steps += 1
        if result.shape != (7,) or not np.isfinite(result).all():
            raise ValueError("ICRT returned an invalid action")
        if self.model.use_delta_action:
            result = current_pose_action(result, self.previous_proprio, observation["proprio"])
        self.previous_proprio = np.asarray(observation["proprio"], dtype=float).copy()
        return result.astype(float).tolist()
