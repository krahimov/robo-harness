"""Re-record a teacher on current physics/cameras, then test the production adapter.

Old artifacts remain immutable. The teacher uses recorded commands; the query
uses only ICRT predictions. Task requirements come from the frozen experiment.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from studio import storage as db
from studio.adapters.icrt import ICRTModel
from studio.adapters.simulation import make_simulator
from studio.judges import state_judge
from studio.schemas import Scene


def run(out, experiment_id, demo_id, seed, steps):
    out.mkdir(parents=True, exist_ok=False)
    source = db.get("experiment", experiment_id)
    demo = db.get("demo", demo_id)
    old = db.read_artifact(demo["artifact_id"])
    plan = {"experiment_id": experiment_id, "source_demo_id": demo_id,
        "source_artifact_id": demo["artifact_id"], "query_seed": seed,
        "steps": steps, "actor": "ICRT production adapter; fixed weights",
        "teacher": "Old commands re-executed with current simulator and camera",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    teacher = make_simulator(Scene(**old["scene"]))
    teacher_frames = [teacher.snapshot()]
    for command in old["actions"]:
        teacher_frames.append(teacher.step(command))
    recorded = {"scene": teacher.scene.model_dump(), "task": source["task"],
        "xml": teacher.source, "frames": teacher_frames, "actions": old["actions"],
        "frequency": teacher.frequency, "provenance": plan}
    (out / "teacher.json").write_text(json.dumps(recorded))
    gate = state_judge(recorded, {"require_review": True})
    (out / "teacher-verdict.json").write_text(json.dumps(gate, indent=2))
    print("teacher", gate["status"], flush=True)
    if gate["status"] != "pass":
        raise ValueError("Re-recorded teacher failed; no model episode started")
    sim = make_simulator(Scene(**{**source["scene"], "seed": seed}))
    frames, actions, actor, error = [sim.snapshot()], [], None, None
    try:
        actor = ICRTModel([{"id": f"rerecorded-{demo_id}", "artifact": recorded}],
            frames[0], sim, horizon_steps=steps)
        for i in range(steps):
            command = actor.act(frames[-1])
            actions.append(command)
            frames.append(sim.step(command))
            if i % 100 == 0:
                print("step", i, flush=True)
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        if actor:
            actor.close()
        artifact = {"scene": sim.scene.model_dump(), "task": source["task"],
            "xml": sim.source, "frames": frames, "actions": actions,
            "frequency": sim.frequency, "model": actor.metadata if actor else {},
            "predictions": actor.predictions if actor else [], "plan": plan,
            "execution_error": error}
        (out / "rollout.json").write_text(json.dumps(artifact))
        target = "pen" if "pen" in frames[0].get("objects", {}) else "cube"
        positions = [f["objects"][target]["position"] if target == "pen" else f["cube"] for f in frames]
        report = {"verdict": state_judge(artifact, {"require_review": True}),
            "steps": len(actions), "execution_error": error,
            "grasp_frames": sum(f["objects"][target]["attached"] if target == "pen" else f["attached"] for f in frames),
            "closest_m": min(float(np.linalg.norm(np.asarray(f["ee"]) - p)) for f, p in zip(frames, positions)),
            "clipped_predictions": artifact["model"].get("clipped_predictions"),
            "executed_feedback_steps": artifact["model"].get("executed_feedback_steps")}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--demo", required=True)
    p.add_argument("--seed", type=int, default=62000)
    p.add_argument("--steps", type=int, default=415)
    a = p.parse_args()
    if not 1 <= a.steps <= 510:
        p.error("steps must fit the 512-frame worker sequence")
    run(a.output, a.experiment, a.demo, a.seed, a.steps)
