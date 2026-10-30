import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field
from studio import storage as db
from studio.schemas import Scene, DemoCreate, JudgeSpec, ExperimentCreate, Review, Promotion, TASK, StrictModel, Waypoint
from studio.adapters.simulation import Simulator, make_simulator, default_waypoints, task_waypoints, record
from studio.adapters.models import ReplayModel, HTTPModel, build_model, adapter_statuses
from studio.judges import DEFAULT_RUBRIC, evaluate, rendered_frames, judge_connection, judge_connections
from studio import authoring_api, comparisons, judge_lab_api, task_api, assets, blender_adapter, scene_packages, robot_visuals
from studio.schemas import task_for_scene
from studio import calibration_sets

logger = logging.getLogger("studio")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode")
cancelled: dict[str, threading.Event] = {}
queue_lock = threading.RLock()


def motion_profile(artifact):
    frames = artifact["frames"]
    distances = [math.dist(a["ee"], b["ee"]) for a, b in zip(frames, frames[1:])]
    peaks = [max(distances[i * len(distances)//35 : max(i * len(distances)//35 + 1, (i+1) * len(distances)//35)], default=0) for i in range(35)]
    maximum = max(peaks, default=0) or 1
    return [round(value/maximum, 3) for value in peaks]


def store_demo(name, scene, artifact, source, **extra):
    return db.save("demo", {"name": name.strip(), "version": 1, "scene": scene, "artifact_id": db.write_artifact(artifact),
        "duration": artifact["frames"][-1]["t"], "frames": len(artifact["frames"]), "source": source,
        "embodiment": artifact["frames"][0].get("embodiment", TASK["embodiment"]), "task": artifact.get("task"), "status": "candidate", "motion_profile": motion_profile(artifact), **extra})


def seed():
    if not db.all_records("judge"):
        db.save("judge", {**JudgeSpec(name="Task state checks", rubric=DEFAULT_RUBRIC).model_dump(), "version": 1})
    if not db.all_records("demo"):
        scene = Scene()
        store_demo("Block to tray / full demonstration", scene.model_dump(), record(scene, default_waypoints(scene)), "scripted")
    for experiment in db.all_records("experiment"):
        if experiment["status"] in ["queued", "running"]:
            db.update("experiment", experiment["id"], status="interrupted", error="Server restarted. Completed episode artifacts were preserved.")


@asynccontextmanager
async def lifespan(app):
    global executor
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode")
    seed()
    authoring_api.startup()
    yield
    authoring_api.shutdown()
    for event in list(cancelled.values()):
        event.set()
    executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Robot Harness Studio", version="0.1.0", lifespan=lifespan)
app.include_router(authoring_api.router)
app.include_router(comparisons.router)
app.include_router(judge_lab_api.router)
app.include_router(task_api.router)
app.include_router(assets.router)
app.include_router(blender_adapter.router)
app.include_router(scene_packages.router)
app.include_router(robot_visuals.router)
app.include_router(calibration_sets.router)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    allowed = {"http://127.0.0.1:5178", "http://localhost:5178", "http://127.0.0.1:8788", "http://localhost:8788"}
    # The private cloud studio uses an SSH-forwarded loopback port.
    if os.environ.get("STUDIO_UI_ORIGIN") == "http://127.0.0.1:8789":
        allowed.add("http://127.0.0.1:8789")
    if request.method not in ["GET", "HEAD", "OPTIONS"] and origin and origin not in allowed:
        return JSONResponse({"detail": "This local API does not accept writes from that origin."}, status_code=403)
    return await call_next(request)


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/api/bootstrap")
def bootstrap():
    demos = db.all_records("demo")
    for demo in demos:
        if "motion_profile" not in demo:
            demo["motion_profile"] = motion_profile(db.read_artifact(demo["artifact_id"]))
    return {"task": TASK, "demos": demos, "judges": db.all_records("judge"), "experiments": db.all_records("experiment"), "tasks": task_api.list_tasks(),
        "adapters": [
            *adapter_statuses(),
            {"id": "mujoco", "name": "MuJoCo", "type": "simulator", "ready": True, "description": "Cartesian gripper with position control, object contact physics, and an idealized proximity grasp."},
            {"id": "vlm", "name": "VLM judge", "type": "judge", "ready": any(c["configured"] for c in judge_connections()), "description": "Configurable judge using a saved Codex or Claude Code login, or a VLM endpoint, with state and camera evidence, human labels, and calibration."},
        ]}


@app.post("/api/scene/preview")
def preview(scene: Scene, task_id: str | None = None):
    task = task_api.resolve_task(scene, task_id)
    return {"frame": make_simulator(scene).snapshot(), "waypoints": task_waypoints(scene, task), "task": task}


@app.get("/api/artifacts/{id}")
def artifact(id: str):
    return db.read_artifact(id)


@app.post("/api/demos", status_code=201)
def create_demo(spec: DemoCreate):
    task = task_api.resolve_task(spec.scene, getattr(spec, "task_id", None))
    waypoints = [w.model_dump(exclude_none=True) for w in spec.waypoints] if spec.waypoints else task_waypoints(spec.scene, task, spec.clearance)
    artifact = record(spec.scene, waypoints)
    artifact["task"] = task
    return store_demo(spec.name, spec.scene.model_dump(), artifact, "keyframes" if spec.waypoints else "scripted", waypoints=waypoints, task_id=getattr(spec, "task_id", None))


class ImportDemo(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    scene: Scene
    actions: list[Waypoint] = Field(min_length=1, max_length=1200)
    task: dict | None = None


@app.post("/api/demos/import", status_code=201)
def import_demo(spec: ImportDemo):
    task = task_api.resolve_task(spec.scene)
    imported_task = None
    if spec.task is not None:
        if spec.task.get("schema_version") == 2:
            validated = task_api.TaskCreate(**{key: spec.task[key] for key in ["name", "description", "requirements", "operations"] if key in spec.task})
            imported_task = {**validated.model_dump(), "schema_version": 2, "version": 1,
                "owner": "human", "source": "imported", "embodiment": spec.scene.embodiment}
            task = task_api.bind_task(spec.scene, imported_task)
        elif spec.task.get("requirements") != task["requirements"]:
            raise ValueError("Unsupported task snapshot. Import a version 2 task or the original task requirements.")
    sim = make_simulator(spec.scene)
    frames = [sim.snapshot()]
    actions = []
    for action in spec.actions:
        item = action.model_dump(exclude={"seconds"}, exclude_none=True)
        actions.append(item)
        frames.append(sim.step(item))
    if imported_task:
        saved_task = db.save("task", imported_task)
        task = task_api.bind_task(spec.scene, saved_task)
    artifact = {"xml": sim.source, "scene": spec.scene.model_dump(), "frames": frames, "actions": actions, "frequency": sim.frequency, "task": task}
    return store_demo(spec.name, spec.scene.model_dump(), artifact, "imported", task_id=task.get("id") if imported_task else None)


@app.get("/api/demos/{id}/export")
def export_demo(id: str):
    demo = db.get("demo", id)
    # Serialize applied actuator limits, including early fixture recordings with
    # targets microscopically outside a bound that MuJoCo clamped internally.
    bounds = {"x": (-0.5, 0.5), "y": (-0.35, 0.35), "z": (0.025, 0.45)}
    actions = [{**action, **{axis: round(min(high, max(low, action[axis])), 7) for axis, (low, high) in bounds.items()}} for action in db.read_artifact(demo["artifact_id"])["actions"]]
    return {"name": demo["name"], "scene": demo["scene"], "actions": actions,
        "task": db.read_artifact(demo["artifact_id"]).get("task")}


@app.post("/api/judges", status_code=201)
def create_judge(spec: JudgeSpec):
    if spec.kind == "state" and "inspect_state" not in spec.tools:
        raise ValueError("State checks require the inspect_state tool")
    if spec.kind == "vlm" and not spec.model.strip():
        raise ValueError("A VLM judge needs a model name")
    return db.save("judge", {**spec.model_dump(), "version": 1})


@app.post("/api/judges/{id}/versions", status_code=201)
def version_judge(id: str, spec: JudgeSpec):
    parent = db.get("judge", id)
    if spec.kind == "state" and "inspect_state" not in spec.tools:
        raise ValueError("State checks require the inspect_state tool")
    if spec.kind == "vlm" and not spec.model.strip():
        raise ValueError("A VLM judge needs a model name")
    family = parent.get("root_id", parent["id"])
    versions = [j["version"] for j in db.all_records("judge") if j.get("root_id", j["id"]) == family]
    return db.save("judge", {**spec.model_dump(), "version": max(versions) + 1, "parent_id": id, "root_id": family})


def save_evaluation(episode_id, artifact, judge):
    try:
        result = evaluate(artifact, judge)
    except Exception as exc:
        logger.error("Evaluation failed (%s)", type(exc).__name__)
        # Preserve simulation evidence even when the configured judge fails.
        result = {"status": "error", "outcomes": [], "needs_review": True, "kind": judge["kind"], "summary": "Judge failed. Check its endpoint, tool access, and response format.", "error_type": type(exc).__name__}
    return db.save("evaluation", {"episode_id": episode_id, "judge": judge, **result})


def run_experiment(id, event):
    if db.get("experiment", id)["adapter"] == "astra":
        from studio import astra_runs
        try:
            astra_runs.run(id, event)
        finally:
            cancelled.pop(id, None)
        return
    experiment = db.get("experiment", id)
    db.update("experiment", id, status="running")
    demos = [{**d, "artifact": db.read_artifact(d["artifact_id"])} for d in experiment["context"]]
    try:
        for index in range(experiment["episodes"]):
            if event.is_set():
                break
            scene = Scene(**{**experiment["scene"], "seed": experiment["scene"]["seed"] + index})
            sim = make_simulator(scene)
            frames, actions = [sim.snapshot()], []
            model = build_model(experiment["adapter"], demonstrations=demos, observation=frames[0], task=experiment["task"], instructions=experiment["instructions"], simulator=sim, **({"horizon_steps": experiment.get("horizon_steps") or max((len(d["artifact"]["actions"]) for d in demos), default=180)} if experiment["adapter"] == "icrt" else {}))
            try:
                length = experiment.get("horizon_steps") or (len(model.actions) if isinstance(model, ReplayModel) else max((len(d["artifact"]["actions"]) for d in demos), default=180))
                for _ in range(min(length, 1200)):
                    if event.is_set():
                        break
                    action = model.act(frames[-1])
                    actions.append(action)
                    frames.append(sim.step(action))
            finally:
                model.close()
            if event.is_set():
                break
            artifact = {"scene": scene.model_dump(), "xml": sim.source, "frames": frames, "actions": actions, "frequency": sim.frequency, "task": experiment["task"], "model": getattr(model, "metadata", {"adapter": experiment["adapter"]}), "predictions": getattr(model, "predictions", [])}
            episode = db.save("episode", {"experiment_id": id, "index": index + 1, "scene": scene.model_dump(), "artifact_id": db.write_artifact(artifact), "duration": frames[-1]["t"], "selected_demo_id": model.selected["id"] if isinstance(model, ReplayModel) else None})
            evaluation = save_evaluation(episode["id"], artifact, experiment["judge"])
            db.update("episode", episode["id"], evaluation_id=evaluation["id"], initial_evaluation_id=evaluation["id"])
            db.update("experiment", id, completed=index + 1)
        db.update("experiment", id, status="cancelled" if event.is_set() else "completed", finished_at=db.now())
    except Exception as exc:
        logger.error("Experiment failed (%s)", type(exc).__name__)
        db.update("experiment", id, status="failed", error=f"{type(exc).__name__}: execution failed. Verify the selected model endpoint and adapter protocol.")
    finally:
        cancelled.pop(id, None)


def prepare_experiment(spec: ExperimentCreate):
    if spec.adapter == "icrt":
        connection = next(a for a in adapter_statuses() if a["id"] == "icrt")
        if not connection["ready"]:
            raise HTTPException(409, "Start the ICRT checkpoint worker before running inference. See the ICRT integration guide.")
    if spec.adapter == "replay" and not spec.demo_ids:
        raise HTTPException(422, "Trajectory replay requires a demonstration. Choose an inference model for the zero-demo condition.")
    if spec.adapter == "http" and not os.environ.get("STUDIO_ROBOT_ENDPOINT"):
        raise HTTPException(409, "The robotics model endpoint is not connected")
    if len(set(spec.demo_ids)) != len(spec.demo_ids):
        raise ValueError("Context contains duplicate demonstrations")
    judge = db.get("judge", spec.judge_id)
    if judge["kind"] == "vlm" and not judge_connection(judge)["configured"]:
        raise HTTPException(409, judge_connection(judge)["message"])
    context = [db.get("demo", id) for id in spec.demo_ids]
    if any(d["embodiment"] != spec.scene.embodiment for d in context):
        raise ValueError("Demonstration embodiment does not match the scene")
    if spec.adapter == "icrt":
        if len(context) > 4:
            raise ValueError("ICRT supports up to four demonstrations per context")
        from studio.adapters.icrt import context_budget
        horizon = spec.horizon_steps or max((len(db.read_artifact(d['artifact_id'])['actions']) for d in context),default=180)
        context_budget(horizon,len(context))
        for demo in context:
            artifact = db.read_artifact(demo["artifact_id"])
            if 'name="wrist"' not in artifact["xml"]:
                raise ValueError("This demonstration predates wrist-camera recording. Record a new demonstration for ICRT.")
    prepared = {**spec.model_dump(), "status": "queued", "completed": 0, "task": task_api.resolve_task(spec.scene, getattr(spec, "task_id", None)), "judge": judge, "context": context}
    if spec.adapter == "astra":
        from studio.astra_runs import prepare
        return prepare(spec, prepared)
    return prepared


def enqueue_experiment(prepared):
    experiment = db.save("experiment", prepared)
    schedule_experiment(experiment)
    return experiment


def schedule_experiment(experiment):
    event = cancelled[experiment["id"]] = threading.Event()
    executor.submit(run_experiment, experiment["id"], event)


@app.post("/api/experiments", status_code=202)
def create_experiment(spec: ExperimentCreate):
    with queue_lock:
        if len(cancelled) >= 8:
            raise HTTPException(429, "Eight experiments are already queued. Wait for a run to finish.")
        return enqueue_experiment(prepare_experiment(spec))


@app.get("/api/experiments/{id}")
def experiment_detail(id: str):
    experiment = db.get("experiment", id)
    evaluations = db.all_records("evaluation")
    reviews = db.all_records("review")
    episodes = sorted([e for e in db.all_records("episode") if e["experiment_id"] == id], key=lambda e: e["index"])
    for episode in episodes:
        episode["evaluations"] = [v for v in evaluations if v["episode_id"] == episode["id"]]
        episode["reviews"] = [r for r in reviews if r["episode_id"] == episode["id"]]
    return {**experiment, "results": episodes}


@app.post("/api/experiments/{id}/cancel")
def cancel_experiment(id: str):
    experiment = db.get("experiment", id)
    if id in cancelled:
        cancelled[id].set()
    return {"requested": experiment["status"] in ["queued", "running"]}


@app.post("/api/episodes/{id}/reviews", status_code=201)
def review_episode(id: str, spec: Review):
    episode = db.get("episode", id)
    return db.save("review", {"episode_id": id, "evaluation_id": episode.get("evaluation_id"), **spec.model_dump()})


@app.post("/api/episodes/{id}/promote", status_code=201)
def promote_episode(id: str, spec: Promotion):
    episode = db.get("episode", id)
    reviews = [r for r in db.all_records("review") if r["episode_id"] == id]
    if not reviews or reviews[0]["decision"] != "accept":
        raise HTTPException(409, "Accept the episode in human review before saving it as a demonstration")
    return store_demo(spec.name, episode["scene"], db.read_artifact(episode["artifact_id"]), "rollout", source_episode_id=id)


class Rejudge(StrictModel):
    judge_id: str


@app.post("/api/episodes/{id}/evaluate", status_code=201)
def rejudge_episode(id: str, spec: Rejudge):
    episode = db.get("episode", id)
    judge = db.get("judge", spec.judge_id)
    if judge["kind"] == "vlm" and not judge_connection(judge)["configured"]:
        raise HTTPException(409, judge_connection(judge)["message"])
    evaluation = save_evaluation(id, db.read_artifact(episode["artifact_id"]), judge)
    db.update("episode", id, evaluation_id=evaluation["id"])
    return evaluation


@app.get("/api/episodes/{id}/camera")
def camera_frame(id: str, frame: int = 0):
    episode = db.get("episode", id)
    artifact = db.read_artifact(episode["artifact_id"])
    if not 0 <= frame < len(artifact["frames"]):
        raise HTTPException(422, "Frame index is outside the recording")
    try:
        return rendered_frames(artifact, [frame])[0]
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
