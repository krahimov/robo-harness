"""Validated articulated scene graph, compiled to MJCF by the studio.

Authors describe nested bodies, joints, geoms, and sites. The studio writes the
MuJoCo XML itself; model-written XML, meshes with file references, plugins, and
actuators are never accepted here. Joint dynamics are simulation assumptions,
not measured material properties.
"""
import math
from typing import Annotated, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    # Same policy as studio.schemas.StrictModel; kept local so schemas can import this module.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Color = Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=4, max_length=4)]
Name = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
Vector = Annotated[list[Annotated[float, Field(ge=-1, le=1)]], Field(min_length=3, max_length=3)]
Angles = Annotated[list[Annotated[float, Field(ge=-2*math.pi, le=2*math.pi)]], Field(min_length=3, max_length=3)]

# Plain ints: model.geom_type values do not hash like mjtGeom enum members.
SPHERE, CYLINDER, CAPSULE, MESH = (int(t) for t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_MESH))
MAX_BODIES = 24
MAX_GEOMS = 120
MAX_DEPTH = 5
MAX_MESHES = 16


def _numbers(values):
    return " ".join(str(float(value)) for value in values)


class GraphGeom(StrictModel):
    """Box and mesh size is XYZ half-extents. Cylinder/capsule: [radius, radius, half-length]. Sphere: [radius, radius, radius].

    A mesh references an imported mesh asset; its geometry is scaled to fit the half-extents.
    """
    name: Name | None = None
    shape: Literal["box", "cylinder", "sphere", "capsule", "mesh"] = "box"
    asset_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    size: list[Annotated[float, Field(ge=.002, le=.4)]] = Field(min_length=3, max_length=3)
    position: Vector = Field(default_factory=lambda: [0, 0, 0])
    euler_xyz: Angles = Field(default_factory=lambda: [0, 0, 0])
    color: Color = Field(default_factory=lambda: [.72, .58, .40, 1])
    mass: float | None = Field(default=None, ge=.001, le=5)
    friction: float | None = Field(default=None, ge=.05, le=2)
    collidable: bool = True

    @model_validator(mode="after")
    def mesh_asset(self):
        if (self.shape == "mesh") != bool(self.asset_id):
            raise ValueError("Mesh geoms require an imported asset_id; primitive geoms cannot use one")
        return self


class GraphJoint(StrictModel):
    """Hinge range is radians; slide range is metres. `initial` is the starting joint value."""
    name: Name
    type: Literal["hinge", "slide", "ball", "free"] = "hinge"
    axis: Vector = Field(default_factory=lambda: [0, 0, 1])
    position: Vector = Field(default_factory=lambda: [0, 0, 0])
    range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    initial: float | None = None
    damping: float = Field(default=.05, ge=0, le=50)
    stiffness: float = Field(default=0, ge=0, le=100)
    spring_ref: float = Field(default=0, ge=-2*math.pi, le=2*math.pi)
    friction_loss: float = Field(default=0, ge=0, le=10)
    armature: float = Field(default=.001, ge=0, le=1)

    @model_validator(mode="after")
    def valid_joint(self):
        if np.linalg.norm(self.axis) < 1e-6:
            raise ValueError(f"Joint {self.name} needs a nonzero axis")
        if self.type in {"ball", "free"} and (self.range or self.initial is not None or self.stiffness):
            raise ValueError(f"{self.type} joint {self.name} cannot declare a range, initial value, or stiffness")
        if self.range:
            low, high = self.range
            limit = 2 * math.pi if self.type == "hinge" else .6
            if not -limit <= low < high <= limit:
                raise ValueError(f"Joint {self.name} range must increase within ±{limit:.2f}")
            if self.initial is not None and not low <= self.initial <= high:
                raise ValueError(f"Joint {self.name} initial value must lie inside its range")
        return self


class GraphSite(StrictModel):
    """A named point on a body, such as a handle or a push point, used by skills and checks."""
    name: Name
    position: Vector = Field(default_factory=lambda: [0, 0, 0])
    size: float = Field(default=.006, ge=.002, le=.03)


class SceneBody(StrictModel):
    """A rigid body in its parent's frame. A root body without joints is fixed to the world."""
    name: Name
    position: Vector = Field(default_factory=lambda: [0, 0, 0])
    euler_xyz: Angles = Field(default_factory=lambda: [0, 0, 0])
    joints: list[GraphJoint] = Field(default_factory=list, max_length=3)
    geoms: list[GraphGeom] = Field(min_length=1, max_length=24)
    sites: list[GraphSite] = Field(default_factory=list, max_length=8)
    children: list["SceneBody"] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def valid_body(self):
        free = [joint for joint in self.joints if joint.type == "free"]
        if free and len(self.joints) > 1:
            raise ValueError(f"Body {self.name} cannot combine a free joint with other joints")
        if any(child.joints and child.joints[0].type == "free" for child in self.children):
            raise ValueError("Only root bodies can use a free joint")
        return self

    def walk(self, depth=1):
        yield self, depth
        for child in self.children:
            yield from child.walk(depth + 1)


def validate_graph(bodies: list[SceneBody], reserved=()):
    """Scene-level limits: unique names, depth, counts, and root placement."""
    walked = [item for body in bodies for item in body.walk()]
    if len(walked) > MAX_BODIES:
        raise ValueError(f"A scene supports at most {MAX_BODIES} articulated bodies")
    if max((depth for _, depth in walked), default=0) > MAX_DEPTH:
        raise ValueError(f"Articulated bodies can nest at most {MAX_DEPTH} levels")
    if sum(len(body.geoms) for body, _ in walked) > MAX_GEOMS:
        raise ValueError(f"Articulated bodies support at most {MAX_GEOMS} geoms")
    if sum(geom.shape == "mesh" for body, _ in walked for geom in body.geoms) > MAX_MESHES:
        raise ValueError(f"Articulated bodies support at most {MAX_MESHES} mesh geoms")
    names = [body.name for body, _ in walked]
    joints = [joint.name for body, _ in walked for joint in body.joints]
    sites = [site.name for body, _ in walked for site in body.sites]
    for label, values in [("Body", names), ("Joint", joints), ("Site", sites)]:
        if len(values) != len(set(values)):
            raise ValueError(f"{label} names must be unique across the scene graph")
    for body, _ in walked:
        geoms = [geom.name for geom in body.geoms if geom.name]
        if len(geoms) != len(set(geoms)):
            raise ValueError(f"Geom names must be unique within body {body.name}")
    clashes = set(names) & set(reserved)
    if clashes:
        raise ValueError("Body names clash with existing objects: " + ", ".join(sorted(clashes)))
    for body in bodies:
        x, y, z = body.position
        if not (abs(x) <= .6 and abs(y) <= .4 and 0 <= z <= .8):
            raise ValueError(f"Root body {body.name} must sit on the table within x ±0.6 m, y ±0.4 m, z 0–0.8 m")


def _quat(euler):
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=float), "xyz")
    return quat


def _mesh_name(body, index):
    return f"graph_{body.name}_{index}"


def _geom_xml(body, index, geom):
    name = f"geom_{body.name}_{geom.name or index}"
    if geom.shape == "mesh":
        shape = f'type="mesh" mesh="{_mesh_name(body, index)}"'
    else:
        size = geom.size if geom.shape == "box" else [geom.size[0]] if geom.shape == "sphere" else [geom.size[0], geom.size[2]]
        shape = f'type="{geom.shape}" size="{_numbers(size)}"'
    extra = ""
    if geom.mass is not None:
        extra += f' mass="{geom.mass}"'
    if geom.friction is not None:
        extra += f' friction="{geom.friction} .01 .001"'
    contact = "" if geom.collidable else ' contype="0" conaffinity="0"'
    return (f'<geom name="{name}" {shape} pos="{_numbers(geom.position)}" '
        f'quat="{_numbers(_quat(geom.euler_xyz))}" rgba="{_numbers(geom.color)}"{extra}{contact}/>')


def _joint_xml(joint):
    if joint.type == "free":
        return f'<freejoint name="joint_{joint.name}"/>'
    attributes = [f'name="joint_{joint.name}"', f'type="{joint.type}"', f'pos="{_numbers(joint.position)}"',
        f'damping="{joint.damping}"', f'armature="{joint.armature}"', f'frictionloss="{joint.friction_loss}"']
    if joint.type != "ball":
        attributes += [f'axis="{_numbers(joint.axis)}"', f'stiffness="{joint.stiffness}"', f'springref="{joint.spring_ref}"']
    if joint.range:
        attributes += ['limited="true"', f'range="{_numbers(joint.range)}"']
    return "<joint " + " ".join(attributes) + "/>"


def _body_xml(body):
    parts = [_joint_xml(joint) for joint in body.joints]
    parts += [_geom_xml(body, index, geom) for index, geom in enumerate(body.geoms)]
    parts += [f'<site name="site_{site.name}" pos="{_numbers(site.position)}" size="{site.size}" rgba="1 .8 .1 .6" group="4"/>' for site in body.sites]
    parts += [_body_xml(child) for child in body.children]
    return f'<body name="body_{body.name}" pos="{_numbers(body.position)}" quat="{_numbers(_quat(body.euler_xyz))}">{"".join(parts)}</body>'


def graph_assets(bodies: list[SceneBody]):
    """Mesh asset elements for mesh geoms; vertices come from the bounded studio mesh store."""
    from studio.assets import mesh_data
    elements = []
    for root in bodies:
        for body, _ in root.walk():
            for index, geom in enumerate(body.geoms):
                if geom.shape == "mesh":
                    asset = mesh_data(geom.asset_id)
                    elements.append(f'<mesh name="{_mesh_name(body, index)}" vertex="{_numbers(v for point in asset["vertices"] for v in point)}" '
                        f'face="{" ".join(str(int(v)) for face in asset["faces"] for v in face)}" scale="{_numbers(geom.size)}"/>')
    return "".join(elements)


def graph_xml(bodies: list[SceneBody]):
    return "".join(_body_xml(body) for body in bodies)


def apply_initial(model, data, bodies: list[SceneBody]):
    """Set declared starting joint values before the simulator settles."""
    for body in bodies:
        for item, _ in body.walk():
            for joint in item.joints:
                if joint.initial is not None:
                    data.qpos[model.jnt_qposadr[model.joint("joint_" + joint.name).id]] = joint.initial


class GraphState:
    """Resolve graph names to MuJoCo ids once, then read per-frame state."""

    def __init__(self, model, bodies: list[SceneBody]):
        self.model = model
        walked = [item for body in bodies for item, _ in body.walk()]
        self.bodies = {body.name: model.body("body_" + body.name).id for body in walked}
        self.joints = {joint.name: (model.joint("joint_" + joint.name).id, joint.type) for body in walked for joint in body.joints}
        self.sites = {site.name: model.site("site_" + site.name).id for body in walked for site in body.sites}
        self.body_geoms = {name: [i for i in range(model.ngeom) if model.geom_bodyid[i] == id] for name, id in self.bodies.items()}

    def aabb(self, data, name):
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for i in self.body_geoms[name]:
            rotation = data.geom_xmat[i].reshape(3, 3)
            size = self.model.geom_size[i]
            kind = int(self.model.geom_type[i])
            if kind == MESH:
                center = data.geom_xpos[i] + rotation @ self.model.geom_aabb[i][:3]
                half = np.abs(rotation) @ self.model.geom_aabb[i][3:]
                low, high = np.minimum(low, center - half), np.maximum(high, center + half)
                continue
            if kind == SPHERE:
                half = np.full(3, size[0])
            elif kind in {CYLINDER, CAPSULE}:
                axis = rotation[:, 2]
                length = size[1] + (size[0] if kind == CAPSULE else 0)
                half = np.abs(axis) * length + size[0] * np.sqrt(np.maximum(0, 1 - axis**2))
            else:
                half = np.abs(rotation) @ size[:3]
            center = data.geom_xpos[i]
            low, high = np.minimum(low, center - half), np.maximum(high, center + half)
        return (low + high) / 2, (high - low) / 2

    def joint_value(self, data, name):
        id, kind = self.joints[name]
        start = self.model.jnt_qposadr[id]
        width = {"free": 7, "ball": 4}.get(kind, 1)
        values = data.qpos[start:start + width]
        return round(float(values[0]), 6) if width == 1 else values.round(6).tolist()

    def snapshot(self, data, grasps=None):
        velocity = np.empty(6)
        bodies = {}
        for name, id in self.bodies.items():
            center, half = self.aabb(data, name)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, id, velocity, 0)
            bodies[name] = {"position": center.round(5).tolist(), "extent": half.round(5).tolist(),
                "origin": data.xpos[id].round(5).tolist(), "quat": data.xquat[id].round(6).tolist(),
                "speed": round(float(np.linalg.norm(velocity[3:])), 5), "attached": bool((grasps or {}).get(name, False))}
        joints = {}
        for name, (id, kind) in self.joints.items():
            dof = self.model.jnt_dofadr[id]
            joints[name] = {"type": kind, "value": self.joint_value(data, name),
                "velocity": round(float(np.linalg.norm(data.qvel[dof:dof + {"free": 6, "ball": 3}.get(kind, 1)])), 5)}
        sites = {name: data.site_xpos[id].round(5).tolist() for name, id in self.sites.items()}
        return {"bodies": bodies, "joints": joints, "sites": sites}


def mesh_geoms(scene: dict):
    """Every scene-graph mesh geom dict in a dumped scene, for asset collection and remapping."""
    def walk(body):
        yield from (geom for geom in body.get("geoms", []) if geom.get("asset_id"))
        for child in body.get("children", []):
            yield from walk(child)
    return [geom for body in scene.get("bodies", []) for geom in walk(body)]


def scene_asset_ids(scene: dict):
    """Mesh asset IDs used by props and scene-graph geoms, in first-use order."""
    ids = [prop["asset_id"] for prop in scene.get("props", []) if prop.get("asset_id")] + [geom["asset_id"] for geom in mesh_geoms(scene)]
    return list(dict.fromkeys(ids))


class GraphEquality(StrictModel):
    """Constraint between scene-graph parts. connect pins an anchor (body1 frame) to body2 or the world;
    weld fixes a relative pose; joint couples joint2 = polycoef(joint1)."""
    type: Literal["connect", "weld", "joint"]
    body1: Name | None = None
    body2: Name | None = None
    joint1: Name | None = None
    joint2: Name | None = None
    anchor: Vector = Field(default_factory=lambda: [0, 0, 0])
    relpose: list[float] | None = Field(default=None, min_length=7, max_length=7)
    polycoef: list[Annotated[float, Field(ge=-100, le=100)]] = Field(default_factory=lambda: [0, 1, 0, 0, 0], min_length=5, max_length=5)
    solref: list[Annotated[float, Field(ge=-1000, le=1000)]] | None = Field(default=None, min_length=2, max_length=2)
    solimp: list[Annotated[float, Field(ge=0, le=1)]] | None = Field(default=None, min_length=5, max_length=5)

    @model_validator(mode="after")
    def references(self):
        if self.type == "joint" and (self.joint1 is None or self.body1 or self.body2):
            raise ValueError("A joint equality names joint1 (and optionally joint2), not bodies")
        if self.type != "joint" and (self.body1 is None or self.joint1 or self.joint2):
            raise ValueError("Connect and weld equalities name body1 (and optionally body2), not joints")
        return self


def validate_constraints(bodies, equalities, excludes):
    walked = [item for body in bodies for item, _ in body.walk()]
    names = {body.name for body in walked}
    joints = {joint.name for body in walked for joint in body.joints if joint.type in {"hinge", "slide"}}
    for equality in equalities:
        missing = [n for n in [equality.body1, equality.body2] if n and n not in names] + \
            [n for n in [equality.joint1, equality.joint2] if n and n not in joints]
        if missing:
            raise ValueError("Equality references unknown bodies or hinge/slide joints: " + ", ".join(missing))
    for pair in excludes:
        if not set(pair) <= names:
            raise ValueError("Contact exclusion references unknown bodies: " + ", ".join(sorted(set(pair) - names)))


def constraint_xml(equalities, excludes):
    """Equality elements (named scene_eq_*) and contact exclusions for the scene graph."""
    elements = []
    for index, equality in enumerate(equalities):
        attributes = [f'name="scene_eq_{index}"']
        if equality.type == "joint":
            attributes += [f'joint1="joint_{equality.joint1}"', f'polycoef="{_numbers(equality.polycoef)}"']
            if equality.joint2:
                attributes.append(f'joint2="joint_{equality.joint2}"')
        else:
            attributes.append(f'body1="body_{equality.body1}"')
            if equality.body2:
                attributes.append(f'body2="body_{equality.body2}"')
            attributes.append(f'anchor="{_numbers(equality.anchor)}"')
            if equality.type == "weld" and equality.relpose:
                attributes.append(f'relpose="{_numbers(equality.relpose)}"')
        if equality.solref:
            attributes.append(f'solref="{_numbers(equality.solref)}"')
        if equality.solimp:
            attributes.append(f'solimp="{_numbers(equality.solimp)}"')
        elements.append(f"<{equality.type} " + " ".join(attributes) + "/>")
    exclusions = "".join(f'<exclude body1="body_{a}" body2="body_{b}"/>' for a, b in excludes)
    return "".join(elements), exclusions


MAX_CLOTH_SIDE = 15


class Cloth(StrictModel):
    """Rectangular grid cloth simulated as a MuJoCo flex. Vertex i sits at row i // ny, column i % ny.

    `position` is the cloth center; `count` is vertices along X and Y before `yaw`. Edge lengths are
    constrained; bending, stretch, and material stiffness are not calibrated.
    """
    name: Name
    position: Vector
    yaw: float = Field(default=0, ge=-2 * math.pi, le=2 * math.pi)
    count: list[Annotated[int, Field(ge=3, le=MAX_CLOTH_SIDE)]] = Field(default_factory=lambda: [9, 9], min_length=2, max_length=2)
    spacing: float = Field(default=.02, ge=.008, le=.04)
    mass: float = Field(default=.03, ge=.005, le=.5)
    thickness: float = Field(default=.002, ge=.001, le=.006)
    damping: float = Field(default=.002, ge=0, le=.05)
    friction: float = Field(default=1, ge=.1, le=2)
    self_collide: bool = False
    pinned: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list, max_length=16)
    color: Color = Field(default_factory=lambda: [.22, .48, .78, 1])

    @model_validator(mode="after")
    def valid_cloth(self):
        x, y, z = self.position
        if not (abs(x) <= .5 and abs(y) <= .35 and 0 <= z <= .5):
            raise ValueError(f"Cloth {self.name} must sit over the table within x ±0.5 m, y ±0.35 m, z 0–0.5 m")
        if any(i >= self.count[0] * self.count[1] for i in self.pinned):
            raise ValueError(f"Cloth {self.name} pins a vertex index beyond its grid")
        return self

    @property
    def vertices(self):
        return self.count[0] * self.count[1]


def cloth_body(cloth_name, index):
    return f"cloth_{cloth_name}_{index}"


def cloth_faces(nx, ny):
    """Grid triangles in vertex-index order, both windings so the browser shows either side."""
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, i * ny + j + 1, (i + 1) * ny + j, (i + 1) * ny + j + 1
            faces += [[a, c, b], [b, c, d], [a, b, c], [b, d, c]]
    return faces


def pinch_target(cloths: list[Cloth]):
    """A movable vertex body to initialize the inactive pinch constraint with."""
    cloth = cloths[0]
    return cloth_body(cloth.name, next(i for i in range(cloth.vertices) if i not in cloth.pinned))


def cloth_xml(cloths: list[Cloth]):
    elements = []
    for cloth in cloths:
        pins = f'<pin id="{" ".join(str(i) for i in cloth.pinned)}"/>' if cloth.pinned else ""
        quat = _quat([0, 0, cloth.yaw])
        elements.append(f'<flexcomp name="cloth_{cloth.name}" type="grid" dim="2" count="{cloth.count[0]} {cloth.count[1]} 1" '
            f'spacing="{cloth.spacing} {cloth.spacing} {cloth.spacing}" pos="{_numbers(cloth.position)}" quat="{_numbers(quat)}" '
            f'radius="{cloth.thickness}" mass="{cloth.mass}" rgba="{_numbers(cloth.color)}">'
            # Stiff contact plus a margin equal to the thickness keeps the light vertices from sinking into surfaces.
            f'<contact condim="3" selfcollide="{"auto" if cloth.self_collide else "none"}" friction="{cloth.friction} .01 .001" '
            f'solref=".003 1" margin="{cloth.thickness}"/>'
            f'<edge equality="true" damping="{cloth.damping}"/>{pins}</flexcomp>')
    return "".join(elements)


class ClothState:
    """Cloth vertex positions and an idealized pinch: one cloth vertex pinned to the gripper by a point constraint."""
    PINCH_RADIUS = .02

    def __init__(self, model, cloths: list[Cloth], pinch_body: str):
        self.model = model
        self.cloths = cloths
        self.flex = {cloth.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cloth_" + cloth.name) for cloth in cloths}
        self.faces = {cloth.name: cloth_faces(*cloth.count) for cloth in cloths}
        self.pinch_eq = model.eq("cloth_pinch").id
        self.pinch_body = model.body(pinch_body).id
        self.pinched = None

    def vertices(self, data, name):
        flex = self.flex[name]
        start, count = self.model.flex_vertadr[flex], self.model.flex_vertnum[flex]
        return data.flexvert_xpos[start:start + count]

    def update(self, data, closed, point):
        """Release when the gripper opens; when it closes within reach of a vertex, pin the nearest one."""
        if not closed:
            data.eq_active[self.pinch_eq] = 0
            self.pinched = None
            return
        if self.pinched is not None:
            return
        best = None
        for cloth in self.cloths:
            vertices = self.vertices(data, cloth.name)
            distances = np.linalg.norm(vertices - point, axis=1)
            distances[list(cloth.pinned)] = np.inf  # Pinned vertices belong to the world.
            index = int(np.argmin(distances))
            distance = float(np.linalg.norm(vertices[index] - point))
            if distance <= self.PINCH_RADIUS and (best is None or distance < best[0]):
                best = (distance, cloth.name, index, vertices[index].copy())
        if best is None:
            return
        _, name, index, position = best
        flex = self.flex[name]
        self.model.eq_obj2id[self.pinch_eq] = self.model.flex_vertbodyid[self.model.flex_vertadr[flex] + index]
        rotation = data.xmat[self.pinch_body].reshape(3, 3)
        self.model.eq_data[self.pinch_eq, 0:3] = rotation.T @ (position - data.xpos[self.pinch_body])
        self.model.eq_data[self.pinch_eq, 3:6] = 0
        data.eq_active[self.pinch_eq] = 1
        self.pinched = (name, index)

    def snapshot(self, data):
        cloths, geoms = {}, []
        for cloth in self.cloths:
            vertices = self.vertices(data, cloth.name)
            flex = self.flex[cloth.name]
            start = self.model.flex_vertadr[flex]
            bodies = self.model.flex_vertbodyid[start:start + len(vertices)]
            speed = max((float(np.linalg.norm(data.cvel[b][3:])) for b in bodies), default=0.)
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            pinched = self.pinched[1] if self.pinched and self.pinched[0] == cloth.name else None
            cloths[cloth.name] = {"vertices": vertices.round(5).tolist(), "count": cloth.count, "position": ((low + high) / 2).round(5).tolist(),
                "extent": ((high - low) / 2).round(5).tolist(), "speed": round(speed, 5), "attached": pinched is not None, "pinched_vertex": pinched}
            geoms.append({"name": "cloth_" + cloth.name, "type": "mesh", "size": [1, 1, 1], "color": list(cloth.color),
                "position": [0, 0, 0], "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "mesh_vertices": vertices.round(5).tolist(), "mesh_faces": self.faces[cloth.name]})
        return {"cloths": cloths, "cloth_grasp": "idealized pinch: a point constraint holds one vertex while the gripper is closed"}, geoms


class TendonJoint(StrictModel):
    joint: Name
    coef: float = Field(default=1, ge=-10, le=10)


class GraphTendon(StrictModel):
    """Fixed tendons couple joint values; spatial tendons run straight between sites (cables, springs)."""
    name: Name
    type: Literal["fixed", "spatial"] = "spatial"
    joints: list[TendonJoint] = Field(default_factory=list, max_length=8)
    sites: list[Name] = Field(default_factory=list, max_length=8)
    stiffness: float = Field(default=0, ge=0, le=1000)
    damping: float = Field(default=0, ge=0, le=100)
    rest_length: float | None = Field(default=None, ge=0, le=2)
    range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    width: float = Field(default=.002, ge=.0005, le=.02)
    color: Color = Field(default_factory=lambda: [.85, .75, .3, 1])

    @model_validator(mode="after")
    def path(self):
        if self.type == "fixed" and (not self.joints or self.sites):
            raise ValueError(f"Fixed tendon {self.name} lists joints, not sites")
        if self.type == "spatial" and (len(self.sites) < 2 or self.joints):
            raise ValueError(f"Spatial tendon {self.name} runs through at least two sites")
        if self.range and not self.range[0] < self.range[1]:
            raise ValueError(f"Tendon {self.name} range must increase")
        return self


class GraphActuator(StrictModel):
    """Powered scene part. The control follows `schedule`, piecewise constant: [[seconds, value], ...].

    position: ctrl is a target (rad or m) tracked with kp/kv; velocity: ctrl is a target speed with kv;
    motor: ctrl times gear is a force or torque. Robot controllers never drive these.
    """
    name: Name
    type: Literal["position", "velocity", "motor"] = "position"
    joint: Name | None = None
    tendon: Name | None = None
    kp: float = Field(default=10, ge=0, le=10000)
    kv: float = Field(default=1, ge=0, le=1000)
    gear: float = Field(default=1, ge=-100, le=100)
    ctrl_range: list[float] = Field(default_factory=lambda: [-1, 1], min_length=2, max_length=2)
    force_range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    schedule: list[Annotated[list[float], Field(min_length=2, max_length=2)]] = Field(default_factory=lambda: [[0, 0]], min_length=1, max_length=32)

    @model_validator(mode="after")
    def valid_actuator(self):
        if (self.joint is None) == (self.tendon is None):
            raise ValueError(f"Actuator {self.name} drives exactly one joint or tendon")
        low, high = self.ctrl_range
        if not low < high:
            raise ValueError(f"Actuator {self.name} control range must increase")
        times = [t for t, _ in self.schedule]
        if times[0] != 0 or times != sorted(times) or len(set(times)) != len(times):
            raise ValueError(f"Actuator {self.name} schedule starts at 0 s and increases in time")
        if any(not low <= value <= high for _, value in self.schedule):
            raise ValueError(f"Actuator {self.name} schedule values must stay inside its control range")
        return self

    def control(self, time):
        value = self.schedule[0][1]
        for start, setting in self.schedule:
            if start <= time + 1e-9:
                value = setting
        return value


def validate_drives(bodies, tendons, actuators):
    walked = [item for body in bodies for item, _ in body.walk()]
    joints = {joint.name for body in walked for joint in body.joints if joint.type in {"hinge", "slide"}}
    sites = {site.name for body in walked for site in body.sites}
    names = [t.name for t in tendons] + [a.name for a in actuators]
    if len(names) != len(set(names)):
        raise ValueError("Tendon and actuator names must be unique")
    for tendon in tendons:
        missing = [j.joint for j in tendon.joints if j.joint not in joints] + [s for s in tendon.sites if s not in sites]
        if missing:
            raise ValueError(f"Tendon {tendon.name} references unknown hinge/slide joints or sites: " + ", ".join(missing))
    known = {t.name for t in tendons}
    for actuator in actuators:
        if actuator.joint and actuator.joint not in joints or actuator.tendon and actuator.tendon not in known:
            raise ValueError(f"Actuator {actuator.name} references an unknown joint or tendon")


def drive_xml(tendons, actuators):
    """Tendon and actuator elements; scene names are prefixed so robot controllers never confuse them."""
    tendon_elements = []
    for tendon in tendons:
        attributes = f'name="tendon_{tendon.name}" stiffness="{tendon.stiffness}" damping="{tendon.damping}"'
        if tendon.rest_length is not None:
            attributes += f' springlength="{tendon.rest_length}"'
        if tendon.range:
            attributes += f' limited="true" range="{_numbers(tendon.range)}"'
        if tendon.type == "fixed":
            body = "".join(f'<joint joint="joint_{j.joint}" coef="{j.coef}"/>' for j in tendon.joints)
            tendon_elements.append(f"<fixed {attributes}>{body}</fixed>")
        else:
            body = "".join(f'<site site="site_{s}"/>' for s in tendon.sites)
            tendon_elements.append(f'<spatial {attributes} width="{tendon.width}" rgba="{_numbers(tendon.color)}">{body}</spatial>')
    actuator_elements = []
    for actuator in actuators:
        target = f'joint="joint_{actuator.joint}"' if actuator.joint else f'tendon="tendon_{actuator.tendon}"'
        attributes = f'name="scene_act_{actuator.name}" {target} ctrlrange="{_numbers(actuator.ctrl_range)}" ctrllimited="true"'
        if actuator.force_range:
            attributes += f' forcerange="{_numbers(actuator.force_range)}" forcelimited="true"'
        if actuator.type == "position":
            actuator_elements.append(f'<position {attributes} kp="{actuator.kp}" kv="{actuator.kv}"/>')
        elif actuator.type == "velocity":
            actuator_elements.append(f'<velocity {attributes} kv="{actuator.kv}"/>')
        else:
            actuator_elements.append(f'<motor {attributes} gear="{actuator.gear}"/>')
    return "".join(tendon_elements), "".join(actuator_elements)


class Drives:
    """Applies scene actuator schedules each control step and records their state."""

    def __init__(self, model, actuators, tendons):
        self.items = [(actuator, model.actuator("scene_act_" + actuator.name).id) for actuator in actuators]
        self.tendons = {tendon.name: model.tendon("tendon_" + tendon.name).id for tendon in tendons}

    def apply(self, data):
        for actuator, id in self.items:
            data.ctrl[id] = actuator.control(data.time)

    def snapshot(self, data):
        return {"actuators": {actuator.name: {"ctrl": round(float(data.ctrl[id]), 6), "force": round(float(data.actuator_force[id]), 5)}
                for actuator, id in self.items},
            "tendons": {name: {"length": round(float(data.ten_length[id]), 6)} for name, id in self.tendons.items()}}
