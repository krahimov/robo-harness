"""ICRT-specific visual observations, physical prompts, and action transforms.

The compact pretrained policy predicts full DROID/Franka poses. Cartesian
projection is exposed as a research mode with explicit embodiment limitations;
it never silently advertises compatibility with the training robot.
"""
import json
import math
import os
import select
import subprocess
import sys
import xml.etree.ElementTree as ET

import httpx
import numpy as np
from studio.adapters.icrt_frames import tip_to_flange, flange_to_tip

FREQUENCY = 15
MAX_CONTEXT_FRAMES = 256
DEFAULT_ENDPOINT = "http://127.0.0.1:8790"
ADAPTER_VERSION = "icrt-studio-v4"
TIMESTAMP_TOLERANCE = 1e-5  # Snapshots serialize simulation time to five decimals.
LIMITATIONS = [
    "The released policy was trained on Franka/DROID observations; this simulated camera and scene distribution is uncalibrated.",
    "Cartesian research mode projects the 7D action to XYZ and gripper; predicted orientation is recorded but cannot be executed.",
    "Simulated grasping and camera geometry do not establish real-robot transfer or ICL task success.",
]


def endpoint():
    return os.environ.get("STUDIO_ICRT_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def headers():
    token = os.environ.get("STUDIO_ICRT_TOKEN", "")
    return {"Authorization": "Bearer " + token} if token else {}


def status():
    ready, details = False, None
    try:
        with httpx.Client(timeout=0.8) as client:
            response = client.get(endpoint() + "/health", headers=headers())
            data = response.json()
            ready = response.status_code == 200 and isinstance(data, dict) and data.get("ready") is True and data.get("protocol") == "robot-harness-icrt-v1"
            if ready:
                details = {key: data.get(key) for key in ["model", "device", "parameters", "fixed_weights", "source_commit", "policy_sha256", "runtime_adapter_version", "action_reference", "upstream_pose_reference_correction", "supports_executed_action_feedback"]}
    except (httpx.HTTPError, ValueError):
        pass
    return {"id": "icrt", "name": "ICRT · pretrained research actor", "type": "model", "ready": ready,
        "supports_zero_demo": True, "supports_language": False, "description": "Released fixed-weight ICRT policy with side/wrist camera context and no language conditioning. Franka runs execute pose and gripper; Cartesian runs project away rotation. Task performance and sim-to-real transfer are unvalidated.",
        "setup": "Start integrations/icrt/worker.py on port 8790 after preparing the checkpoints.", "details": details}


def pose(frame, *, base=(-0.65, 0, 0), projected=True):
    coordinates = np.asarray(frame["ee"], dtype=float) - np.asarray(base, dtype=float)
    rotation = [math.pi, 0, 0] if projected else frame["euler_xyz"]
    if not projected:
        coordinates, rotation = tip_to_flange(coordinates, rotation)
    gripper = float(frame["gripper"] if "gripper" in frame else frame["closed"])
    values = [*coordinates.tolist(), *rotation, gripper]
    if len(values) != 7 or not np.isfinite(values).all():
        raise ValueError("ICRT requires a finite 7D end-effector pose")
    return values


def demonstration_indices(artifact, budget):
    count = len(artifact["actions"])
    if count < 1 or len(artifact["frames"]) != count + 1:
        raise ValueError("ICRT demonstrations need aligned observation/action pairs")
    frequency = float(artifact["frequency"])
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Invalid demonstration frequency")
    indices = np.unique(np.minimum(count - 1, np.rint(np.arange(0, count / frequency, 1 / FREQUENCY) * frequency).astype(int)))
    if len(indices) > budget:
        indices = indices[np.linspace(0, len(indices) - 1, budget).round().astype(int)]
    return indices.tolist()


def validate_camera_source(source):
    try:
        cameras = {node.get("name") for node in ET.fromstring(source).iter("camera")}
    except (ET.ParseError, TypeError):
        raise ValueError("ICRT needs a valid MuJoCo scene with side and wrist cameras") from None
    if not {"overview", "wrist"}.issubset(cameras):
        raise ValueError("ICRT needs overview and wrist cameras. Re-record this demonstration with the current simulator")


class CameraClient:
    def __init__(self):
        self.process = subprocess.Popen([sys.executable, "-m", "studio.adapters.icrt_render"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1)

    def render(self, source, frames):
        validate_camera_source(source)
        results = []
        for start in range(0, len(frames), 16):
            minimal = [{key: frame[key] for key in ["qpos", "closed"]} for frame in frames[start:start+16]]
            try:
                self.process.stdin.write(json.dumps({"xml": source, "frames": minimal}) + "\n")
                self.process.stdin.flush()
                if not select.select([self.process.stdout], [], [], 45)[0]:
                    raise RuntimeError("ICRT camera rendering timed out")
                response = json.loads(self.process.stdout.readline())
            except (OSError, ValueError):
                raise ValueError("ICRT needs working side and wrist camera rendering") from None
            if "error" in response or len(response.get("frames", [])) != len(minimal):
                raise ValueError("ICRT needs working side and wrist camera rendering")
            results.extend(response["frames"])
        return results

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def context_budget(horizon_steps, demonstration_count):
    if horizon_steps is None:
        return MAX_CONTEXT_FRAMES
    if not 1 <= horizon_steps <= 510:
        raise ValueError("ICRT requires a horizon of 1 to 510 frames")
    budget=min(MAX_CONTEXT_FRAMES,512-horizon_steps-2)
    if demonstration_count and budget < 16*demonstration_count:
        raise ValueError("ICRT horizon leaves fewer than 16 context frames per demonstration; shorten the horizon")
    return budget


class ICRTModel:
    supports_zero_demo = True

    def __init__(self, demonstrations, observation, simulator, horizon_steps=None, **kwargs):
        self.simulator = simulator
        self.projected = observation.get("embodiment", "cartesian-gripper-v1") != "franka-panda-v1"
        self.base = observation.get("robot_base", [-0.65, 0, 0])
        self.camera = CameraClient()
        self.client = httpx.Client(base_url=endpoint(), headers=headers(), timeout=120)
        self.session_id = None
        self.next_prediction = 0.0
        self.last_action = {**dict(zip(["x", "y", "z"], observation["ee"])), "closed": bool(observation["closed"])}
        self.predictions = []
        self.metadata = {"adapter": "icrt", "projection": self.projected, "limitations": list(LIMITATIONS),
            "context_sampling": [], "clipped_predictions": 0, "fixed_weights": True, "gradient_updates": 0,
            "adapter_version": ADAPTER_VERSION, "timestamp_tolerance_seconds": TIMESTAMP_TOLERANCE,
            "supports_language": False, "instructions_ignored": True,
            "model_pose_frame": "panda_link8" if not self.projected else "projected_tcp",
            "simulator_pose_frame": "fingertip_site", "native_tip_offset_m": .1034,
            "wrist_camera_version": observation.get("wrist_camera_version", "legacy-unversioned"),
            "context_wrist_camera_versions": []}
        if not self.projected:
            self.metadata["limitations"] = [LIMITATIONS[0], LIMITATIONS[2]]
        try:
            if len(demonstrations) > 4:
                raise ValueError("ICRT supports at most four selected demonstrations")
            converted = []
            total_budget = context_budget(horizon_steps,len(demonstrations))
            self.metadata["context_frame_budget"] = total_budget
            self.metadata["reserved_query_frames"] = horizon_steps
            budget = total_budget // max(1, len(demonstrations))
            for demo in demonstrations:
                artifact = demo["artifact"]
                camera_version = artifact["frames"][0].get("wrist_camera_version", "legacy-unversioned")
                self.metadata["context_wrist_camera_versions"].append({"id": demo["id"], "version": camera_version})
                if camera_version != self.metadata["wrist_camera_version"]:
                    warning = "Context and rollout wrist-camera mounts differ. Re-record context with the current simulator for a matched-camera experiment."
                    if warning not in self.metadata["limitations"]:
                        self.metadata["limitations"].append(warning)
                native_demo = artifact["frames"][0].get("embodiment") == "franka-panda-v1"
                if native_demo == self.projected:
                    raise ValueError("ICRT demonstration and rollout embodiments must match")
                indices = demonstration_indices(artifact, budget)
                frames = [artifact["frames"][index] for index in indices]
                cameras = self.camera.render(artifact["xml"], frames)
                demo_base = frames[0].get("robot_base", self.base)
                observations = [{**image, "proprio": pose(frame, base=demo_base, projected=self.projected)} for frame, image in zip(frames, cameras)]
                actions = []
                for index in indices:
                    command = artifact["actions"][index]
                    action_frame = {"ee": [command[key] for key in ["x", "y", "z"]],
                        "euler_xyz": command.get("euler_xyz", [math.pi, 0, 0] if native_demo else artifact["frames"][index].get("euler_xyz", [0, 0, 0])),
                        "closed": command["closed"], "gripper": float(command.get("gripper", command["closed"]))}
                    actions.append(pose(action_frame, base=demo_base, projected=self.projected))
                converted.append({"id": demo["id"], "frames": observations, "actions": actions})
                self.metadata["context_sampling"].append({"id": demo["id"], "source_frequency": artifact["frequency"], "frame_indices": indices})
            response = self.client.post("/sessions", json={"protocol": "robot-harness-icrt-v1", "demonstrations": converted})
            self._check_response(response)
            payload = response.json()
            self.session_id = payload["session_id"]
            self.executed_feedback = payload.get("metadata", {}).get("supports_executed_action_feedback") is True
            self.metadata["action_history"] = "executed commands" if self.executed_feedback else "raw predictions; legacy worker"
            self.metadata["executed_feedback_steps"] = 0
            if not isinstance(self.session_id, str) or not self.session_id.isalnum() or len(self.session_id) > 128:
                raise ValueError("Invalid ICRT session response")
            self.metadata["model"] = {key: payload.get("metadata", {}).get(key) for key in
                ["model", "device", "parameters", "frequency", "context_frames", "source_commit", "policy_sha256",
                 "runtime_adapter_version", "action_reference", "upstream_pose_reference_correction"]}
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check_response(response):
        if response.status_code >= 400:
            messages = {401: "ICRT worker authentication failed", 409: "ICRT worker is busy with another session",
                422: "ICRT rejected the observation or reached its context capacity"}
            raise ValueError(messages.get(response.status_code, "ICRT worker request failed; check its local runtime"))

    def act(self, observation):
        if observation["t"] + TIMESTAMP_TOLERANCE < self.next_prediction:
            return dict(self.last_action)
        image = self.camera.render(self.simulator.source, [observation])[0]
        request = {"observation": {**image, "proprio": pose(observation, base=self.base, projected=self.projected)}}
        if self.executed_feedback and self.predictions:
            command = self.last_action
            previous = {"ee": [command[k] for k in ("x", "y", "z")],
                "euler_xyz": command.get("euler_xyz", [math.pi, 0, 0]),
                "closed": command["closed"], "gripper": float(command.get("gripper", command["closed"]))}
            request["previous_action"] = pose(previous, base=self.base, projected=self.projected)
        response = self.client.post(f"/sessions/{self.session_id}/act", json=request)
        self._check_response(response)
        self.metadata["executed_feedback_steps"] += int("previous_action" in request)
        raw = np.asarray(response.json().get("action"), dtype=float)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError("ICRT returned an invalid 7D action")
        xyz = raw[:3] + np.asarray(self.base)
        rotation = raw[3:6].tolist()
        if not self.projected:
            xyz, rotation = flange_to_tip(xyz, rotation)
        current = np.asarray(observation["ee"])
        # Record the intervention. Out-of-distribution predictions cannot command
        # arbitrarily large jumps even inside an offline simulation.
        bounded = np.clip(np.clip(xyz, [-.5, -.35, .025], [.5, .35, .45]), current - .04, current + .04)
        clipped = not np.allclose(xyz, bounded)
        self.metadata["clipped_predictions"] += int(clipped)
        action = {**dict(zip(["x", "y", "z"], bounded.tolist())), "closed": bool(raw[6] > .5)}
        if not self.projected:
            action["euler_xyz"] = rotation
            action["gripper"] = float(np.clip(raw[6], 0, 1))
        self.predictions.append({"t": observation["t"], "raw_action": raw.tolist(), "executed_action": action,
            "rotation_discarded": self.projected, "position_clipped": clipped})
        self.next_prediction += 1 / FREQUENCY
        self.last_action = action
        return dict(action)

    def close(self):
        try:
            if self.session_id:
                self.client.delete(f"/sessions/{self.session_id}")
        except httpx.HTTPError:
            pass
        finally:
            self.client.close()
            self.camera.close()
