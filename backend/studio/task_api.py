"""Human-owned, immutable task versions and numerical checks for recorded objects."""
from copy import deepcopy
from typing import Literal

from fastapi import APIRouter
from pydantic import Field, model_validator

from studio import storage as db
from studio.schemas import StrictModel, Scene, task_for_scene

router = APIRouter(prefix="/api")


class TaskRequirement(StrictModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    label: str = Field(min_length=1, max_length=150)
    definition: str = Field(min_length=1, max_length=1600)
    required: bool = True
    check: Literal["inside_region", "inside_while_held", "released", "settled", "no_drops", "lifted", "lifted_once"]
    object_id: str = Field(default="cube", min_length=1, max_length=40, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    region: Literal["tray"] | None = None
    bounds: list[float] | None = Field(default=None, min_length=6, max_length=6)
    min_height: float = Field(default=.12, ge=-.1, le=.5)
    max_speed: float = Field(default=.03, gt=0, le=.5)

    @model_validator(mode="after")
    def valid_region(self):
        if self.check in {"inside_region", "inside_while_held"} and self.region is None and self.bounds is None:
            raise ValueError("An inside_region requirement needs tray or explicit XYZ bounds")
        if self.bounds and (not all(-1 <= v <= 1 for v in self.bounds) or any(self.bounds[i] >= self.bounds[i+1] for i in (0, 2, 4))):
            raise ValueError("Region bounds must have increasing min/max pairs within ±1 m")
        return self


class TaskOperation(StrictModel):
    object_id: str = Field(default="cube", min_length=1, max_length=40, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    destination: Literal["tray", "region", "hold"] = "tray"
    target: list[float] | None = Field(default=None, min_length=3, max_length=3)

    @model_validator(mode="after")
    def valid_target(self):
        if self.destination != "tray" and self.target is None:
            raise ValueError("Region and hold operations require an XYZ target")
        if self.target and not all(low <= v <= high for v, (low, high) in zip(self.target, [(-.5, .5), (-.35, .35), (.025, .4)])):
            raise ValueError("Operation target exceeds the Cartesian workspace")
        return self


class TaskCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=3000)
    requirements: list[TaskRequirement] = Field(min_length=1, max_length=24)
    operations: list[TaskOperation] = Field(default_factory=list, max_length=2)
    parent_id: str | None = None

    @model_validator(mode="after")
    def meaningful(self):
        if not self.name.strip() or not self.description.strip():
            raise ValueError("A task needs a name and description")
        if len({r.id for r in self.requirements}) != len(self.requirements):
            raise ValueError("Requirement IDs must be unique")
        if not any(r.required for r in self.requirements):
            raise ValueError("A task needs at least one required outcome")
        return self


def _requirements(object_id="cube", prefix="", destination="tray", bounds=None):
    return [
        {"id": prefix + "inside_region", "label": f"{object_id} inside destination", "definition": "The complete object bounding box lies inside the declared destination at episode end.", "required": True, "check": "inside_region", "object_id": object_id, "region": destination, "bounds": bounds},
        {"id": prefix + "released", "label": f"{object_id} released", "definition": "The object's grasp attachment is inactive at episode end.", "required": True, "check": "released", "object_id": object_id},
        {"id": prefix + "settled", "label": f"{object_id} settled", "definition": "Object speed remains below 0.03 m/s for the final half-second.", "required": True, "check": "settled", "object_id": object_id},
        {"id": prefix + "no_drops", "label": f"{object_id} stays above the floor", "definition": "Object center never falls below z=-0.02 m.", "required": True, "check": "no_drops", "object_id": object_id},
    ]


def _preset(id, name, description, requirements, operations, scene=None):
    spec = TaskCreate(name=name, description=description, requirements=requirements, operations=operations)
    return {**spec.model_dump(), "id": id, "root_id": id, "version": 1, "schema_version": 2,
            "owner": "human", "source": "built-in", "embodiment": Scene.model_validate(scene or {}).embodiment,
            "success": "All required outcomes pass. Unknown required outcomes need review.",
            "suggested_scene": Scene.model_validate(scene or {}).model_dump()}


PRESETS = [
    _preset("block-to-tray", "Block to tray", "Place the orange block inside the tray and release it.", _requirements(), [{"object_id": "cube", "destination": "tray"}]),
    _preset("lift-and-hold", "Lift and hold", "Lift the orange block above 12 cm and hold it at the end.", [
        {"id": "lifted", "label": "Block held above 12 cm", "definition": "At episode end, the block center is at least 0.12 m above the table and the grasp is attached.", "check": "lifted", "min_height": .12},
        {"id": "no_drops", "label": "No off-table drops", "definition": "Block center never falls below z=-0.02 m.", "check": "no_drops"}],
        [{"object_id": "cube", "destination": "hold", "target": [-.22, -.08, .2]}]),
    _preset("two-box-sort", "Two-box sorting", "Place the orange block in the tray and the blue block in the rear table region.",
        _requirements() + _requirements("blue", "blue_", None, [-.01, .17, .14, .32, -.005, .09]),
        [{"object_id": "cube", "destination": "tray"}, {"object_id": "blue", "destination": "region", "target": [.08, .23, .045]}],
        {"objects": [{"id": "blue", "position": [-.25, .19, .026]}]}),
]

from studio.workflows import pen_pouch_scene, pen_pouch_requirements

PRESETS.append({**_preset("pen-into-open-pouch", "Pen into an open pouch",
    "Grasp the pen from its raised rests, lift it, rotate it to fit the pouch opening, insert it while held, then release and withdraw. The pouch is held open and modeled with rigid walls; cloth and zipper manipulation are not simulated.",
    pen_pouch_requirements(), [], pen_pouch_scene().model_dump()),
    "required_embodiment": "franka-panda-v1", "suggested_builder": "mujoco"})


def get_task(id):
    preset = next((task for task in PRESETS if task["id"] == id), None)
    return deepcopy(preset) if preset else db.get("task", id)


@router.get("/tasks")
def list_tasks():
    return [*db.all_records("task"), *deepcopy(PRESETS)]


@router.get("/tasks/{id}")
def task_detail(id: str):
    return get_task(id)


@router.post("/tasks", status_code=201)
def create_task(spec: TaskCreate):
    parent = get_task(spec.parent_id) if spec.parent_id else None
    return db.save("task", {**spec.model_dump(), "name": spec.name.strip(), "description": spec.description.strip(),
        "version": parent["version"] + 1 if parent else 1, "root_id": parent.get("root_id") or parent["id"] if parent else None,
        "schema_version": 2, "owner": "human", "source": "user", "embodiment": "cartesian-gripper-v1",
        "success": "All required outcomes pass. Unknown required outcomes need review.",
        "suggested_scene": deepcopy(parent.get("suggested_scene", {})) if parent else {}})


def bind_task(scene: Scene | dict, task: dict):
    scene = Scene.model_validate(scene)
    task = deepcopy(task)
    if task.get("required_embodiment") and task["required_embodiment"] != scene.embodiment:
        raise ValueError("This task requires the articulated Franka arm to rotate and contact-grasp the pen")
    if task.get("schema_version") != 2:
        return task_for_scene(scene)
    known = {"cube", *(obj.id for obj in scene.objects)}
    used = {r["object_id"] for r in task["requirements"]} | {op["object_id"] for op in task.get("operations", [])}
    if not used <= known:
        raise ValueError("Scene is missing task objects: " + ", ".join(sorted(used - known)))
    for requirement in task["requirements"]:
        if requirement.get("region") == "tray":
            requirement["bounds"] = [scene.tray_x - .115, scene.tray_x + .115, scene.tray_y - .085, scene.tray_y + .085, .015, .085]
    task["parameters"] = {"tray_x": scene.tray_x, "tray_y": scene.tray_y}
    task["embodiment"] = scene.embodiment
    return task


def resolve_task(scene: Scene | dict, task_id: str | None = None):
    return bind_task(scene, get_task(task_id)) if task_id else task_for_scene(scene)


def _object(frame, id):
    if "objects" in frame and id in frame["objects"]:
        return frame["objects"][id]
    if id == "cube":
        return {"position": frame["cube"], "extent": frame["cube_extent"], "speed": frame["cube_speed"], "attached": frame["attached"]}
    return None


def measure_task(artifact, judge):
    frames, task = artifact["frames"], artifact["task"]
    outcomes = []
    for r in task["requirements"]:
        observations = [_object(f, r["object_id"]) for f in frames]
        result, reason, timestamps = "unknown", "The recording has no complete state evidence for this object.", []
        if observations and all(obj is not None for obj in observations):
            final = observations[-1]
            timestamps = [frames[-1]["t"]]
            if r["check"] in {"inside_region", "inside_while_held"}:
                bounds = r.get("bounds")
                def inside(obj):
                    return bool(bounds) and all(bounds[2*i] <= obj["position"][i] - obj["extent"][i] and obj["position"][i] + obj["extent"][i] <= bounds[2*i+1] for i in range(3))
                passed = inside(final)
                reason = f"Final {r['object_id']} center {final['position']} m and half-extents {final['extent']} m; region {bounds}."
                if r["check"] == "inside_while_held":
                    matched = [f["t"] for f, obj in zip(frames, observations) if obj["attached"] and inside(obj)]
                    passed = bool(matched)
                    timestamps = matched[:1]
                    reason = f"Full object inside region while held in {len(matched)} recorded frames."
            elif r["check"] == "released":
                passed = not final["attached"]
                reason = f"Grasp attachment is {'active' if final['attached'] else 'inactive'} at episode end."
            elif r["check"] == "settled":
                recent = [obj["speed"] for f, obj in zip(frames, observations) if f["t"] >= frames[-1]["t"] - .5]
                passed = max(recent) < r.get("max_speed", .03)
                reason = f"Maximum speed in final 0.5 s: {max(recent):.4f} m/s."
            elif r["check"] == "no_drops":
                index = min(range(len(frames)), key=lambda i: observations[i]["position"][2])
                low = observations[index]["position"][2]
                passed = low >= -.02
                timestamps = [frames[index]["t"]]
                reason = f"Minimum object center height: {low:.4f} m."
            elif r["check"] == "lifted_once":
                matched = [f["t"] for f, obj in zip(frames, observations) if obj["attached"] and obj["position"][2] >= r.get("min_height", .12)]
                passed = bool(matched)
                timestamps = matched[:1]
                reason = f"Held above {r.get('min_height', .12):.3f} m in {len(matched)} recorded frames."
            else:
                passed = final["position"][2] >= r.get("min_height", .12) and final["attached"]
                reason = f"Final center height {final['position'][2]:.4f} m; grasp {'attached' if final['attached'] else 'released'}."
            result = "pass" if passed else "fail"
        outcomes.append({"id": r["id"], "status": result, "explanation": reason, "timestamps": timestamps})
    required = [o["status"] for o, r in zip(outcomes, task["requirements"]) if r["required"]]
    status = "fail" if "fail" in required else "unknown" if "unknown" in required else "pass"
    return {"status": status, "outcomes": outcomes, "needs_review": judge["require_review"] or any(o["status"] == "unknown" for o in outcomes),
        "kind": "state", "summary": f"Measured against human task {task['name']} v{task['version']}. Rubric prose does not change its thresholds.",
        "tool_trace": [{"tool": "inspect_state", "frames_inspected": len(frames), "ok": True}]}
