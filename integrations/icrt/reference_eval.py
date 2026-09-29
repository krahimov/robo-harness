"""Offline reference episodes: action prediction, never robot task success."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from runtime import ICRTRuntime, ROOT


def validate_split(manifest, max_context=256, max_sequence=512):
    episodes = manifest["episodes"]
    ids = [episode["id"] for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Context and query episodes must be distinct")
    context = [e for e in episodes if e["role"] == "context"]
    queries = [e for e in episodes if e["role"] == "query"]
    if len(context) != 1 or not queries or len(context) + len(queries) != len(episodes):
        raise ValueError("Expected one context episode and separate queries")
    count = context[0]["frames"]
    if count > max_context or any(count + e["frames"] + 1 > max_sequence for e in queries):
        raise ValueError("Complete episodes exceed runtime capacity; do not truncate")
    return context[0], queries


def metrics(predicted, target, proprio):
    predicted, target, proprio = map(np.asarray, (predicted, target, proprio))
    if predicted.shape != target.shape or target.shape != proprio.shape:
        raise ValueError("Aligned prediction, target, and state arrays are required")
    if target.ndim != 2 or target.shape[1] != 7 or not all(np.isfinite(x).all() for x in (predicted, target, proprio)):
        raise ValueError("Expected finite seven-dimensional actions")
    distance = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1)
    rotation = (Rotation.from_euler("XYZ", predicted[:, 3:6]).inv()
                * Rotation.from_euler("XYZ", target[:, 3:6])).magnitude()
    labels, guesses = target[:, 6] > .5, predicted[:, 6] > .5
    recalls = [(guesses[labels == label] == label).mean() for label in (False, True) if np.any(labels == label)]
    moving = np.linalg.norm(target[:, :3] - proprio[:, :3], axis=1) > .01
    return {
        "frames": len(target), "translation_mean_l2_m": float(distance.mean()),
        "translation_p95_l2_m": float(np.percentile(distance, 95)),
        "rotation_mean_geodesic_deg": float(np.degrees(rotation).mean()),
        "gripper_mae": float(np.abs(predicted[:, 6] - target[:, 6]).mean()),
        "gripper_agreement": float((labels == guesses).mean()),
        "gripper_balanced_accuracy": float(np.mean(recalls)),
        "active_frames": int(moving.sum()),
        "active_translation_mean_l2_m": float(distance[moving].mean()) if moving.any() else None,
    }


def recorded_previous_action(runtime, previous_frame, previous_action):
    """Encode a known past action relative to its own observation, never the next."""
    from icrt.data.utils import convert_delta_action, euler_to_rot_6d
    def convert(value):
        value = np.asarray(value)
        return np.r_[value[:3], euler_to_rot_6d(value[None, 3:6]).reshape(-1), value[-1]]
    action = convert(previous_action)[None, None]
    proprio = convert(previous_frame["proprio"])[None, None]
    delta = convert_delta_action(action, proprio).reshape(-1)
    return torch.tensor(np.r_[delta, 0.], dtype=torch.float32, device=runtime.device)[None, None]


def load_episode(folder, item):
    path = folder / item["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Reference episode hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["indices"], np.arange(item["frames"])):
            raise ValueError("Full consecutive episodes required")
        frames = [{"side": Image.fromarray(s), "wrist": Image.fromarray(w), "proprio": p}
                  for s, w, p in zip(data["side"], data["wrist"], data["proprio"])]
        if len(frames) != item["frames"] or len(data["actions"]) != len(frames):
            raise ValueError("Misaligned or incomplete episode")
        return {"frames": frames, "actions": data["actions"].copy(), "proprio": data["proprio"].copy()}


def evaluate(folder, output):
    manifest = json.loads((folder / "manifest.json").read_text())
    context_item, query_items = validate_split(manifest)
    # These conditions are fixed before any inference results are seen.
    modes = ["production_own_history", "corrected_recorded_history", "upstream_notebook"]
    output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "icrt-offline-reference-v1", "dataset": manifest,
              "modes": modes, "contexts": [0, 1], "results": [],
              "limits": "Recorded query observations, separate episodes but unknown training overlap; no closed-loop execution or task-success score. Notebook mode retains upstream temporal/reference behavior and default 16-step execution horizon."}
    (output / "plan.json").write_text(json.dumps(report, indent=2))
    context = load_episode(folder, context_item)
    runtime = ICRTRuntime(device="cpu")
    report["model"] = runtime.metadata()
    for item in query_items:
        query = load_episode(folder, item)
        baseline = metrics(query["proprio"], query["actions"], query["proprio"])
        for mode in modes:
            for context_count in (0, 1):
                started = time.monotonic()
                runtime.reset([])
                if mode == "upstream_notebook":
                    # Match tools/inference.ipynb: default reset, whole prompt,
                    # previous recorded action, teacher_forcing=True, no ensemble.
                    runtime.wrapper.reset()
                    if context_count:
                        runtime.wrapper.prompt(
                            [f["side"] for f in context["frames"]],
                            [f["wrist"] for f in context["frames"]],
                            context["proprio"], context["actions"])
                else:
                    runtime.reset([context] if context_count else [])
                predictions = []
                for index, frame in enumerate(query["frames"]):
                    if mode == "upstream_notebook":
                        prediction = runtime.wrapper(
                            frame["side"], frame["wrist"], query["proprio"][index:index+1],
                            action=query["actions"][index-1:index] if index else None,
                            use_temporal=False, teacher_forcing=True)
                    else:
                        if mode == "corrected_recorded_history" and index:
                            runtime.model.last_action = recorded_previous_action(
                                runtime, query["frames"][index-1], query["actions"][index-1])
                        prediction = runtime.act(frame)
                    predictions.append(np.asarray(prediction).tolist())
                result = {"query": item["id"], "mode": mode, "context_episodes": context_count,
                          "context_frames": context_item["frames"] if context_count else 0,
                          "seconds": time.monotonic() - started,
                          "metrics": metrics(predictions, query["actions"], query["proprio"]),
                          "hold_pose_baseline": baseline}
                name = f"{item['id']}-{mode}-{context_count}"
                (output / (name + ".json")).write_text(json.dumps({**result, "predictions": predictions}, indent=2))
                report["results"].append(result)
                (output / "report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".data/validation/icrt-reference-1")
    parser.add_argument("--output", type=Path, default=ROOT / ".data/validation/icrt-reference-1/evaluation")
    args = parser.parse_args()
    evaluate(args.data, args.output)
