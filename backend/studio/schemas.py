from copy import deepcopy
import math
from typing import Annotated, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Color = Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=4, max_length=4)]


class SceneProp(StrictModel):
    """Static geometry. Box size is XYZ half-extents; cylinder size is radius, unused, half-height."""
    name: str = Field(min_length=1, max_length=48, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    shape: Literal["box", "cylinder", "mesh"] = "box"
    asset_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    position: list[float] = Field(min_length=3, max_length=3)
    size: list[Annotated[float, Field(ge=0.005, le=0.4)]] = Field(min_length=3, max_length=3)
    color: Color = Field(default_factory=lambda: [0.45, 0.5, 0.48, 1])
    collidable: bool = True

    @model_validator(mode="after")
    def bounded_position(self):
        if (self.shape == "mesh") != bool(self.asset_id):
            raise ValueError("Mesh props require an imported asset_id; primitive props cannot use one")
        if not all(low <= value <= high for value, (low, high) in zip(self.position, [(-0.9, 0.9), (-0.7, 0.7), (0, 0.8)])):
            raise ValueError("Prop positions must fit x ±0.9 m, y ±0.7 m, and z 0–0.8 m")
        return self


class SceneObject(StrictModel):
    """Movable boxes or cylinders. Cylinder size is radius, radius, half-length."""
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    shape: Literal["box", "cylinder"] = "box"
    position: list[float] = Field(min_length=3, max_length=3)
    size: list[Annotated[float, Field(ge=0.004, le=0.15)]] = Field(default_factory=lambda: [.025, .025, .025], min_length=3, max_length=3)
    euler_xyz: list[Annotated[float, Field(ge=-2*math.pi, le=2*math.pi)]] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    variation: float = Field(default=0, ge=0, le=.02)
    mass: float = Field(default=.06, ge=.01, le=.3)
    color: Color = Field(default_factory=lambda: [.25, .55, .85, 1])

    @model_validator(mode="after")
    def fits_workspace(self):
        if self.id == "cube":
            raise ValueError("The primary box already uses object ID cube")
        if self.shape == "box" and not all(.015 <= value <= .05 for value in self.size):
            raise ValueError("Movable box half-extents must be between .015 and .05 m")
        if self.shape == "cylinder" and (not math.isclose(self.size[0], self.size[1]) or self.size[0] > .025 or self.size[2] < self.size[0]):
            raise ValueError("Cylinder needs equal radii up to .025 m and a half-length at least its radius")
        if not all(low <= value <= high for value, (low, high) in zip(self.position, [(-.48, .48), (-.32, .32), (.015, .4)])):
            raise ValueError("Movable object positions must fit the tabletop workspace")
        return self


class Conveyor(StrictModel):
    """One-box belt approximation: full length/width, top elevation, velocity in metres/second."""
    x: float = Field(default=-0.22, ge=-0.5, le=0.5)
    y: float = Field(default=-0.08, ge=-0.3, le=0.3)
    length: float = Field(default=0.38, ge=0.12, le=0.8)
    width: float = Field(default=0.14, ge=0.08, le=0.4)
    height: float = Field(default=0.04, ge=0.015, le=0.12)
    speed: float = Field(default=0.025, ge=-0.08, le=0.08)
    direction: Literal["x", "y"] = "x"

    @model_validator(mode="after")
    def fits_table(self):
        half_x, half_y = (self.length / 2, self.width / 2) if self.direction == "x" else (self.width / 2, self.length / 2)
        if abs(self.x) + half_x > 0.6 or abs(self.y) + half_y > 0.4:
            raise ValueError("Conveyor must fit on the table, within x ±0.6 m and y ±0.4 m")
        return self


class Scene(StrictModel):
    embodiment: Literal["cartesian-gripper-v1", "franka-panda-v1"] = "cartesian-gripper-v1"
    seed: int = Field(default=42, ge=0, le=2**31-100)
    variation: float = Field(default=0, ge=0, le=0.15)
    friction: float = Field(default=0.8, ge=0.1, le=2)
    finger_contact_dim: Literal[3, 4, 6] = 3
    contact_solver: Literal["default", "precision"] = "default"
    cube_x: float = Field(default=-0.22, ge=-0.4, le=0.0)
    cube_y: float = Field(default=-0.08, ge=-0.25, le=0.25)
    tray_x: float = Field(default=0.28, ge=-0.35, le=0.4)
    tray_y: float = Field(default=0.05, ge=-0.25, le=0.25)
    cube_color: Color = Field(default_factory=lambda: [0.91, 0.41, 0.16, 1])
    table_color: Color = Field(default_factory=lambda: [0.23, 0.29, 0.26, 1])
    props: list[SceneProp] = Field(default_factory=list, max_length=24)
    conveyor: Conveyor | None = None
    objects: list[SceneObject] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def unique_props(self):
        names = [prop.name for prop in self.props]
        if len(names) != len(set(names)):
            raise ValueError("Prop names must be unique")
        if len({obj.id for obj in self.objects}) != len(self.objects):
            raise ValueError("Movable object IDs must be unique")
        if sum(prop.shape == "mesh" for prop in self.props) > 4:
            raise ValueError("A scene supports at most four imported mesh props")
        return self


class Waypoint(StrictModel):
    x: float = Field(ge=-0.5, le=0.5)
    y: float = Field(ge=-0.35, le=0.35)
    z: float = Field(ge=0.025, le=0.45)
    closed: bool = False
    seconds: float = Field(default=1, ge=0.2, le=3)
    euler_xyz: list[Annotated[float, Field(ge=-2*math.pi, le=2*math.pi)]] | None = Field(default=None, min_length=3, max_length=3)
    gripper: float | None = Field(default=None, ge=0, le=1)


class DemoCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    scene: Scene = Field(default_factory=Scene)
    clearance: float = Field(default=0.2, ge=0.04, le=0.35)
    task_id: str | None = Field(default=None, max_length=100)
    waypoints: list[Waypoint] | None = Field(default=None, min_length=2, max_length=30)


class JudgeSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["state", "vlm"] = "state"
    rubric: str = Field(min_length=10, max_length=16000)
    tools: list[Literal["inspect_state", "replay_frames"]] = Field(default_factory=lambda: ["inspect_state", "replay_frames"])
    require_review: bool = True
    model: str = Field(default="", max_length=150)
    provider: Literal["codex", "openai"] = "openai"

    @model_validator(mode="after")
    def valid_judge_connection(self):
        if self.kind == "vlm" and not self.tools:
            raise ValueError("A VLM judge requires at least one evidence tool")
        if self.kind == "vlm" and self.provider == "codex" and self.model != "gpt-6-astra":
            raise ValueError("Codex judges require model gpt-6-astra")
        return self


class ExperimentCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    demo_ids: list[str] = Field(min_length=0, max_length=8)
    judge_id: str
    scene: Scene = Field(default_factory=Scene)
    episodes: int = Field(default=4, ge=1, le=24)
    adapter: Literal["replay", "http", "icrt", "astra"] = "replay"
    max_attempts: int = Field(default=3, ge=1, le=6)
    feedback_mode: Literal["none", "checkpoints"] = "none"
    max_checkpoints: int = Field(default=2, ge=1, le=3)
    stress_suite: Literal["none", "pen-v1"] = "none"
    horizon_steps: int | None = Field(default=None, ge=1, le=1200)
    task_id: str | None = Field(default=None, max_length=100)
    instructions: str = Field(default="Place the orange block inside the tray and release it.", max_length=4000)


class Review(StrictModel):
    decision: Literal["accept", "reject", "revise_context"]
    note: str = Field(default="", max_length=4000)


class Promotion(StrictModel):
    name: str = Field(min_length=1, max_length=100)


TASK = {
    "id": "block-to-tray", "version": 1, "name": "Block to tray",
    "description": "Place the orange block inside the tray and release it.",
    "embodiment": "cartesian-gripper-v1",
    "requirements": [
        {"id": "inside_tray", "label": "Block fully inside tray", "required": True,
         "definition": "At episode end, block bounding box fits within the tray interior: x 0.165–0.395 m, y -0.035–0.135 m; center z 0.03–0.07 m."},
        {"id": "released", "label": "Gripper released", "required": True,
         "definition": "At episode end the grasp constraint is inactive."},
        {"id": "settled", "label": "Block at rest", "required": True,
         "definition": "Block speed stays below 0.03 m/s for the final 0.5 seconds."},
        {"id": "no_drops", "label": "No off-table drops", "required": True,
         "definition": "Block center never goes below z = -0.02 m during the episode."},
    ],
    "success": "All required outcomes pass. Unknown outcomes require review.",
}


def task_for_scene(scene: Scene | dict):
    """Bind stable task requirements to the authored tray, independently of the rubric."""
    scene = Scene.model_validate(scene)
    task = deepcopy(TASK)
    x_min, x_max = scene.tray_x - 0.115, scene.tray_x + 0.115
    y_min, y_max = scene.tray_y - 0.085, scene.tray_y + 0.085
    task["description"] = "Place the block inside the tray and release it."
    task["parameters"] = {"tray_x": scene.tray_x, "tray_y": scene.tray_y}
    task["embodiment"] = scene.embodiment
    if scene.embodiment == "franka-panda-v1":
        task["requirements"][1]["definition"] = "At episode end, the object is no longer held between the gripper fingers."
    task["requirements"][0]["definition"] = (
        "At episode end, block bounding box fits within the tray interior: "
        f"x {x_min:.3f}–{x_max:.3f} m, y {y_min:.3f}–{y_max:.3f} m; center z 0.03–0.07 m."
    )
    return task
