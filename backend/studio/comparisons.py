"""Matched-seed context experiments and stable, evidence-linked summaries."""
import math
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import Field, model_validator

from studio import storage as db
from studio.schemas import ExperimentCreate, Scene, StrictModel

router = APIRouter(prefix="/api/comparisons")


class Condition(StrictModel):
    name: str = Field(min_length=1, max_length=60)
    demo_ids: list[str] = Field(default_factory=list, max_length=8)


class ComparisonCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    conditions: list[Condition] = Field(min_length=2, max_length=5)
    judge_id: str
    adapter: Literal["replay", "http", "icrt"] = "replay"
    scene: Scene = Field(default_factory=lambda: Scene(seed=1000, variation=.04))
    episodes: int = Field(default=8, ge=1, le=24)
    horizon_steps: int = Field(default=180, ge=1, le=1200)
    instructions: str = Field(default="Place the target box inside its destination and release it.", max_length=4000)
    task_id: str | None = None

    @model_validator(mode="after")
    def distinct_conditions(self):
        names = [c.name.strip() for c in self.conditions]
        contexts = [tuple(c.demo_ids) for c in self.conditions]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Give every condition a distinct name")
        if len(set(contexts)) != len(contexts):
            raise ValueError("Use a different demonstration context for each condition")
        return self


def interval(passes, count):
    """Wilson interval for resolved pass/fail outcomes, not provider failures."""
    if not count:
        return None
    z = 1.959963984540054
    p = passes / count
    denom = 1 + z*z/count
    center = (p + z*z/(2*count)) / denom
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denom
    return [max(0, center-radius), min(1, center+radius)]


def comparison_detail(id):
    comparison = db.get("comparison", id)
    if not comparison["conditions"]:
        return {**comparison, "status": "interrupted", "error": "Comparison initialization was interrupted. Create a new comparison."}
    episodes = db.all_records("episode")
    evaluations = {e["id"]: e for e in db.all_records("evaluation")}
    rows = []
    states = []
    for condition in comparison["conditions"]:
        experiment = db.get("experiment", condition["experiment_id"])
        states.append(experiment["status"])
        attempts = []
        counts = {key: 0 for key in ("pass", "fail", "unknown", "error", "pending")}
        for episode in sorted((e for e in episodes if e["experiment_id"] == experiment["id"]), key=lambda e: e["index"]):
            # A later rejudge must not rewrite the comparison's original evidence.
            evaluation_id = episode.get("initial_evaluation_id") or episode.get("evaluation_id")
            evaluation = evaluations.get(evaluation_id, {})
            status = evaluation.get("status", "pending")
            status = status if status in counts else "unknown"
            counts[status] += 1
            attempts.append({"seed": episode["scene"]["seed"], "status": status,
                "episode_id": episode["id"], "evaluation_id": evaluation_id})
        resolved = counts["pass"] + counts["fail"]
        rows.append({**condition, "status": experiment["status"], "error": experiment.get("error"),
            "counts": counts, "recorded": len(attempts), "planned": comparison["episodes"],
            "resolved": resolved, "unresolved": len(attempts)-resolved,
            "success_rate": counts["pass"]/resolved if resolved else None,
            "confidence_interval": interval(counts["pass"], resolved), "attempts": attempts})
    base = {a["seed"]: a["status"] for a in rows[0]["attempts"]}
    for row in rows:
        paired = [(base[a["seed"]], a["status"]) for a in row["attempts"]
            if base.get(a["seed"]) in {"pass", "fail"} and a["status"] in {"pass", "fail"}]
        gained = sum(a == "fail" and b == "pass" for a, b in paired)
        lost = sum(a == "pass" and b == "fail" for a, b in paired)
        row["paired"] = {"baseline": rows[0]["name"], "count": len(paired), "gained": gained,
            "lost": lost, "delta": (gained-lost)/len(paired) if paired else None}
    status = "running" if any(s in {"queued", "running"} for s in states) else (
        "failed" if "failed" in states else "interrupted" if "interrupted" in states else
        "cancelled" if "cancelled" in states else "completed")
    return {**comparison, "status": status, "conditions": rows}


@router.get("")
def list_comparisons():
    return [comparison_detail(c["id"]) for c in db.all_records("comparison")]


@router.post("", status_code=202)
def create_comparison(spec: ComparisonCreate):
    from studio import main
    seeds = list(range(spec.scene.seed, spec.scene.seed + spec.episodes))
    contexts = {id: db.get("demo", id) for c in spec.conditions for id in c.demo_ids}
    if any(d["scene"]["seed"] in seeds for d in contexts.values()):
        raise HTTPException(422, "Evaluation seeds overlap a selected demonstration. Choose a new starting seed.")
    with main.queue_lock:
        if len(main.cancelled) + len(spec.conditions) > 8:
            raise HTTPException(429, "This comparison would exceed the eight queued experiment limit.")
        # Validate all conditions before saving or scheduling any of them.
        prepared = []
        for c in spec.conditions:
            payload = {"name": f"{spec.name} / {c.name}"[:100], "demo_ids": c.demo_ids,
                "judge_id": spec.judge_id, "adapter": spec.adapter, "scene": spec.scene,
                "episodes": spec.episodes, "instructions": spec.instructions, "horizon_steps":spec.horizon_steps}
            if "task_id" in ExperimentCreate.model_fields:
                payload["task_id"] = spec.task_id
            item = main.prepare_experiment(ExperimentCreate(**payload))
            item["horizon_steps"] = spec.horizon_steps
            prepared.append(item)
        comparison_id = uuid.uuid4().hex
        conditions = []
        experiments = []
        for c, item in zip(spec.conditions, prepared):
            experiment_id = uuid.uuid4().hex
            experiments.append({**item, "id": experiment_id, "comparison_id": comparison_id})
            conditions.append({**c.model_dump(), "experiment_id": experiment_id})
        comparison = {**spec.model_dump(), "id": comparison_id, "seeds": seeds, "conditions": conditions,
            "note": "Only context changes between conditions. Seeds differ from selected demonstration seeds; this does not establish independence from model pretraining."}
        persisted = db.save_many([("comparison", comparison), *[("experiment", e) for e in experiments]])
        try:
            for experiment in persisted[1:]:
                main.schedule_experiment(experiment)
        except RuntimeError:
            for experiment in persisted[1:]:
                event = main.cancelled.get(experiment["id"])
                if event:
                    event.set()
                db.update("experiment", experiment["id"], status="interrupted",
                    error="The runtime stopped while queuing this comparison. Create a new comparison.")
    return comparison_detail(comparison_id)


@router.get("/{id}")
def detail(id: str):
    return comparison_detail(id)


@router.post("/{id}/cancel")
def cancel_comparison(id: str):
    from studio import main
    comparison = db.get("comparison", id)
    for c in comparison["conditions"]:
        main.cancel_experiment(c["experiment_id"])
    return {"requested": True}


@router.get("/{id}/export")
def export_comparison(id: str):
    from studio import main
    summary = comparison_detail(id)
    return {"format": "robot-harness-comparison-v1", "comparison": summary,
        "experiments": [main.experiment_detail(c["experiment_id"]) for c in summary["conditions"]]}
