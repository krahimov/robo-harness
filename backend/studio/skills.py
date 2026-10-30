"""Teacher skills compiled into timed end-effector waypoints.

Skills name scene targets (sites, bodies, joints) instead of raw coordinates.
The compiler executes each skill in a private simulator before resolving the
next one, so later targets reflect the state earlier motion produced. The
result is an ordinary scripted demonstration, not a learned policy.
"""
import math
from typing import Annotated, Literal, Union

import mujoco
import numpy as np
from pydantic import Field, model_validator

from studio.robots import CONTACT_FINGERS, WRIST_ROTATION, require, robot
from studio.scene_graph import StrictModel

Offset = Annotated[list[Annotated[float, Field(ge=-.3, le=.3)]], Field(min_length=3, max_length=3)]
Yaw = Annotated[float, Field(ge=-2*math.pi, le=2*math.pi)]
MAX_SKILLS = 24
MAX_SECONDS = 60
WORKSPACE = [(-.5, .5), (-.35, .35), (.025, .45)]


class Target(StrictModel):
    """Exactly one of a scene-graph site, a body/object center, a world point in metres, or a cloth vertex."""
    site: str | None = Field(default=None, max_length=40)
    body: str | None = Field(default=None, max_length=40)
    point: list[float] | None = Field(default=None, min_length=3, max_length=3)
    cloth: str | None = Field(default=None, max_length=40)
    vertex: int | None = Field(default=None, ge=0, le=224)

    @model_validator(mode="after")
    def exactly_one(self):
        if sum(value is not None for value in [self.site, self.body, self.point, self.cloth]) != 1:
            raise ValueError("A skill target names exactly one site, body, point, or cloth vertex")
        if (self.cloth is None) != (self.vertex is None):
            raise ValueError("A cloth target needs both cloth and vertex")
        return self


class MoveTo(StrictModel):
    """Move the fingertip center to target + offset. `via_height` first travels above the target at that height."""
    kind: Literal["move_to"]
    target: Target
    offset: Offset = Field(default_factory=lambda: [0, 0, 0])
    yaw: Yaw = 0
    via_height: float | None = Field(default=None, ge=.05, le=.45)
    seconds: float = Field(default=1.5, ge=.3, le=6)
    # Which part of the hand goes to the target: the grasp point, the pinch point, or the pushing surface.
    tool: Literal["grasp", "pinch", "push"] = "grasp"


class Gripper(StrictModel):
    """Close (grasp) or open (release) the fingers in place."""
    kind: Literal["grasp", "release"]
    seconds: float = Field(default=.8, ge=.2, le=3)


class FollowJoint(StrictModel):
    """Drive a joint to `to` by tracking a site on the moving part along the joint's own arc or slide.

    Place the site where the fingertips should touch, e.g. just outside a flap's
    face to push it, or on a handle after grasping. The offset is in world metres.
    """
    kind: Literal["follow_joint"]
    joint: str = Field(max_length=40)
    site: str = Field(max_length=40)
    to: float = Field(ge=-2*math.pi, le=2*math.pi)
    offset: Offset = Field(default_factory=lambda: [0, 0, 0])
    yaw: Yaw = 0
    steps: int = Field(default=8, ge=2, le=16)
    seconds: float = Field(default=3, ge=.5, le=10)
    tool: Literal["grasp", "pinch", "push"] = "push"


Skill = Annotated[Union[MoveTo, Gripper, FollowJoint], Field(discriminator="kind")]


def validate_skills(skills):
    if len(skills) > MAX_SKILLS:
        raise ValueError(f"A skill program can contain at most {MAX_SKILLS} skills")
    if sum(skill.seconds for skill in skills) > MAX_SECONDS:
        raise ValueError(f"A skill program can contain at most {MAX_SECONDS} seconds of motion")


def _bounded(position, label):
    for value, (low, high), axis in zip(position, WORKSPACE, "xyz"):
        if not low - 1e-9 <= value <= high + 1e-9:
            raise ValueError(f"{label} needs {axis}={value:.3f} m, outside the robot workspace {low}..{high} m")
    return [round(float(value), 5) for value in position]


class SkillCompiler:
    def __init__(self, scene):
        from studio.adapters.simulation import make_simulator, interpolate_orientation
        self.scene = scene
        self.spec = robot(scene.embodiment)
        self.sim = make_simulator(scene)
        self.interpolate = interpolate_orientation
        state = self.sim.snapshot()
        self.pose = {"x": state["ee"][0], "y": state["ee"][1], "z": state["ee"][2],
            "closed": False, "euler_xyz": state.get("euler_xyz", [0, 0, 0])}
        self.yaw = 0.
        self.points = []

    def orientation(self, yaw):
        if WRIST_ROTATION not in self.spec.capabilities:
            if abs(yaw) > 1e-9:
                raise ValueError(f"{self.spec.name} cannot rotate its wrist; remove the skill yaw")
            return None
        return [math.pi, 0., float(yaw)]

    def emit(self, position, seconds, yaw=None, closed=None, label="Skill", tool="grasp"):
        yaw = self.yaw if yaw is None else yaw
        closed = self.pose["closed"] if closed is None else closed
        orientation = self.orientation(yaw)
        offset = np.asarray(self.spec.tool_offset(tool), dtype=float)
        if offset.any():
            # Command the grasp site so the named tool point lands on the target.
            from studio.adapters.franka import rotation_matrix
            position = np.asarray(position, dtype=float) - rotation_matrix(orientation) @ offset
        point = dict(zip("xyz", _bounded(position, label)), closed=closed, seconds=round(float(seconds), 4))
        if orientation is not None:
            point["euler_xyz"] = orientation
        self.execute(point)
        self.points.append(point)
        self.yaw = yaw

    def execute(self, waypoint):
        """Step the private simulator exactly as record() will replay the waypoint."""
        count = round(waypoint["seconds"] * self.sim.frequency)
        previous = self.pose
        for i in range(count):
            fraction = (i + 1) / count
            action = {axis: previous[axis] + (waypoint[axis] - previous[axis]) * fraction for axis in "xyz"}
            action["closed"] = waypoint["closed"]
            if waypoint.get("euler_xyz") is not None:
                action["euler_xyz"] = self.interpolate(previous.get("euler_xyz") or [0, 0, 0], waypoint["euler_xyz"], fraction)
            self.sim.step(action)
        self.pose = waypoint

    def state(self):
        return self.sim.snapshot()

    def resolve(self, target, label):
        state = self.state()
        if target.point is not None:
            return np.asarray(target.point, dtype=float)
        if target.cloth is not None:
            cloth = state.get("cloths", {}).get(target.cloth)
            if cloth is None or target.vertex >= len(cloth["vertices"]):
                raise ValueError(f"{label} references unknown cloth vertex {target.cloth}[{target.vertex}]")
            return np.asarray(cloth["vertices"][target.vertex], dtype=float)
        if target.site is not None:
            sites = state.get("sites", {})
            if target.site not in sites:
                raise ValueError(f"{label} references unknown site {target.site}")
            return np.asarray(sites[target.site], dtype=float)
        bodies = {**state.get("objects", {}), **state.get("bodies", {})}
        if target.body not in bodies:
            raise ValueError(f"{label} references unknown body or object {target.body}")
        return np.asarray(bodies[target.body]["position"], dtype=float)

    def move_to(self, skill, label):
        goal = self.resolve(skill.target, label) + skill.offset
        if skill.via_height is not None:
            here = [self.pose["x"], self.pose["y"], self.pose["z"]]
            lift = max(skill.via_height, here[2])
            self.emit([here[0], here[1], lift], skill.seconds * .25, skill.yaw, label=label)
            self.emit([goal[0], goal[1], lift], skill.seconds * .4, skill.yaw, label=label, tool=skill.tool)
            self.emit(goal, skill.seconds * .35, skill.yaw, label=label, tool=skill.tool)
        else:
            self.emit(goal, skill.seconds, skill.yaw, label=label, tool=skill.tool)

    def follow_joint(self, skill, label):
        require(self.scene.embodiment, CONTACT_FINGERS, "push or swing articulated parts")
        model, data = self.sim.model, self.sim.data
        try:
            joint = model.joint("joint_" + skill.joint)
            site = model.site("site_" + skill.site).id
        except KeyError:
            raise ValueError(f"{label} references an unknown joint or site") from None
        if int(model.jnt_type[joint.id]) not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            raise ValueError(f"{label} can only follow hinge or slide joints")
        address = model.jnt_qposadr[joint.id]
        if model.jnt_limited[joint.id]:
            low, high = model.jnt_range[joint.id]
            if not low <= skill.to <= high:
                raise ValueError(f"{label} target {skill.to} lies outside joint {skill.joint} range {low:.3f}..{high:.3f}")
        # Kinematic preview on a scratch copy: the site's path if only this joint moved.
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        start = float(data.qpos[address])
        for step in range(1, skill.steps + 1):
            scratch.qpos[address] = start + (skill.to - start) * step / skill.steps
            mujoco.mj_kinematics(model, scratch)
            self.emit(scratch.site_xpos[site] + skill.offset, skill.seconds / skill.steps, skill.yaw, label=label, tool=skill.tool)

    def compile(self, skills):
        validate_skills(skills)
        for index, skill in enumerate(skills, 1):
            label = f"Skill {index} ({skill.kind})"
            if skill.kind == "move_to":
                self.move_to(skill, label)
            elif skill.kind in {"grasp", "release"}:
                here = [self.pose["x"], self.pose["y"], self.pose["z"]]
                self.emit(here, skill.seconds, closed=skill.kind == "grasp", label=label)
            else:
                self.follow_joint(skill, label)
        if len(self.points) < 2:
            raise ValueError("A skill program must produce at least two motion waypoints")
        return self.points


def compile_skills(scene, skills):
    from studio.schemas import Scene
    from pydantic import TypeAdapter
    skills = TypeAdapter(list[Skill]).validate_python([s if isinstance(s, dict) else s.model_dump() for s in skills])
    return SkillCompiler(Scene.model_validate(scene)).compile(skills)
