"""MuJoCo cartesian manipulation fixture with an explicitly idealized grasp.

The robot uses three position-controlled slide joints. Object gravity, contact,
friction, and release are simulated. Grasping uses a proximity-gated weld rather
than finger contact mechanics. This is an integration fixture, not a real arm.
The optional belt applies a bounded velocity-drive force during box contact.
It has no rollers, deformable belt, or continuous box supply.
"""
import base64
import io
import numpy as np
import mujoco
from PIL import Image
from studio.schemas import Scene, task_for_scene
from studio.scene_graph import (ClothState, Drives, GraphState, apply_initial, cloth_xml, constraint_xml, drive_xml, graph_assets,
    graph_xml, pinch_target)


def numbers(values):
    return " ".join(str(value) for value in values)


def geometry_extent(model, data, name):
    """World AABB half-extents, including the circular cross-section of a cylinder."""
    geom = data.geom(name)
    spec = model.geom(name)
    rotation = geom.xmat.reshape(3, 3)
    if spec.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = rotation[:, 2]
        return np.abs(axis) * spec.size[1] + spec.size[0] * np.sqrt(np.maximum(0, 1-axis**2))
    return np.abs(rotation) @ spec.size


def with_cloth(frame, cloth, data, drives=None):
    """Add cloth vertex state, browser cloth meshes, and scene actuator/tendon state to a recorded frame."""
    if drives is not None:
        frame.update(drives.snapshot(data))
    if cloth is None:
        return frame
    state, geoms = cloth.snapshot(data)
    frame.update(state)
    frame["geoms"] = frame["geoms"] + geoms
    return frame


def geom_kind(model, index, meshes=()):
    """Browser shape name for a recorded geom; unknown primitives draw as boxes."""
    if index in meshes:
        return "mesh"
    return {int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule"}.get(int(model.geom_type[index]), "box")


def interpolate_orientation(start, end, fraction):
    """Shortest-path quaternion interpolation, avoiding Euler wraparound turns."""
    from studio.adapters.franka import euler_xyz
    first, last = np.empty(4), np.empty(4)
    mujoco.mju_euler2Quat(first, np.asarray(start, dtype=float), "xyz")
    mujoco.mju_euler2Quat(last, np.asarray(end, dtype=float), "xyz")
    dot = float(first @ last)
    if dot < 0:
        last, dot = -last, -dot
    angle = np.arccos(np.clip(dot, -1, 1))
    if angle < 1e-5:
        result = (1-fraction)*first + fraction*last
    else:
        result = (np.sin((1-fraction)*angle)*first + np.sin(fraction*angle)*last)/np.sin(angle)
    result /= np.linalg.norm(result)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, result)
    return euler_xyz(matrix.reshape(3, 3))


def conveyor_half_size(conveyor):
    return (conveyor.length / 2, conveyor.width / 2) if conveyor.direction == "x" else (conveyor.width / 2, conveyor.length / 2)


def on_conveyor(conveyor, x, y):
    half_x, half_y = conveyor_half_size(conveyor)
    return abs(x - conveyor.x) <= half_x and abs(y - conveyor.y) <= half_y


def xml(scene: Scene):
    rng = np.random.default_rng(scene.seed)
    x = scene.cube_x + rng.uniform(-scene.variation, scene.variation)
    y = scene.cube_y + rng.uniform(-scene.variation, scene.variation)
    cube_z = 0.026
    props, mesh_assets = [], []
    for prop in scene.props:
        if prop.shape == "mesh":
            from studio.assets import mesh_data
            asset = mesh_data(prop.asset_id)
            mesh_assets.append(f'<mesh name="asset_{prop.name}" vertex="{numbers(v for point in asset["vertices"] for v in point)}" face="{numbers(v for face in asset["faces"] for v in face)}" scale="{numbers(prop.size)}"/>')
            geometry = f'type="mesh" mesh="asset_{prop.name}"'
        else:
            size = prop.size if prop.shape == "box" else [prop.size[0], prop.size[2]]
            geometry = f'type="{prop.shape}" size="{numbers(size)}"'
        props.append(f'<geom name="prop_{prop.name}" {geometry} pos="{numbers(prop.position)}" rgba="{numbers(prop.color)}" contype="{int(prop.collidable)}" conaffinity="{int(prop.collidable)}"/>')
    object_bodies, object_grasps = [], []
    for obj in scene.objects:
        position = np.asarray(obj.position).copy()
        if obj.variation:
            position[:2] += rng.uniform(-obj.variation, obj.variation, 2)
        quaternion = np.empty(4)
        mujoco.mju_euler2Quat(quaternion, np.asarray(obj.euler_xyz, dtype=float), "xyz")
        size = obj.size if obj.shape == "box" else [obj.size[0], obj.size[2]]
        object_bodies.append(f'<body name="object_{obj.id}" pos="{numbers(position)}" quat="{numbers(quaternion)}"><freejoint name="object_{obj.id}_free"/><geom name="object_{obj.id}_geom" type="{obj.shape}" size="{numbers(size)}" mass="{obj.mass}" rgba="{numbers(obj.color)}"/></body>')
        object_grasps.append(f'<weld name="grasp_{obj.id}" body1="gripper" body2="object_{obj.id}" active="false" relpose="0 0 0 1 0 0 0" solref="0.005 1"/>')
    conveyor_xml = ""
    if scene.conveyor:
        belt = scene.conveyor
        half_x, half_y = conveyor_half_size(belt)
        # A low tangential contact friction avoids static locking against the
        # explicit surface-velocity drive. Normal support is still MuJoCo contact.
        conveyor_xml = f'<geom name="conveyor_belt" type="box" size="{half_x} {half_y} {belt.height / 2}" pos="{belt.x} {belt.y} {belt.height / 2}" rgba="0.14 0.18 0.19 1" friction="0.02 0.001 0.0001" priority="1"/>'
        if on_conveyor(belt, x, y):
            cube_z += belt.height
    tray_xml = f'''<geom name="tray_base" type="box" size="0.125 0.095 0.008" pos="{scene.tray_x} {scene.tray_y} 0.008" rgba="0.57 0.7 0.56 1"/>
<geom name="tray_north" type="box" size="0.125 0.005 0.025" pos="{scene.tray_x} {scene.tray_y + 0.09} 0.025" rgba="0.62 0.75 0.6 1"/>
<geom name="tray_south" type="box" size="0.125 0.005 0.025" pos="{scene.tray_x} {scene.tray_y - 0.09} 0.025" rgba="0.62 0.75 0.6 1"/>
<geom name="tray_east" type="box" size="0.005 0.085 0.025" pos="{scene.tray_x + 0.12} {scene.tray_y} 0.025" rgba="0.62 0.75 0.6 1"/>
<geom name="tray_west" type="box" size="0.005 0.085 0.025" pos="{scene.tray_x - 0.12} {scene.tray_y} 0.025" rgba="0.62 0.75 0.6 1"/>''' if scene.include_tray else ""
    cube_xml = f'''<body name="cube" pos="{x} {y} {cube_z}"><freejoint name="cube_free"/><geom name="block" type="box" size="0.025 0.025 0.025" mass="0.06" rgba="{numbers(scene.cube_color)}"/></body>''' if scene.include_cube else ""
    if scene.include_cube:
        object_grasps.insert(0, '<weld name="grasp" body1="gripper" body2="cube" active="false" relpose="0 0 0 1 0 0 0" solref="0.005 1"/>')
    scene_equalities, scene_excludes = constraint_xml(scene.equalities, scene.contact_excludes)
    scene_tendons, scene_actuators = drive_xml(scene.tendons, scene.actuators)
    if scene.cloths:
        # Idealized cloth pinch for the gantry; articulated robots replace body1 with their own hand.
        scene_equalities = f'<connect name="cloth_pinch" body1="gripper" body2="{pinch_target(scene.cloths)}" anchor="0 0 0" active="false"/>' + scene_equalities
    solver = 'cone="elliptic" impratio="100" solver="Newton" tolerance="1e-10" noslip_iterations="5"' if scene.contact_solver == "precision" else ''
    contact = 'solref=".004 1" solimp=".95 .99 .001"' if scene.contact_solver == "precision" else ''
    return f'''<mujoco model="Cartesian manipulation fixture">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" {solver}/>
      <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
      <default><geom friction="{scene.friction} 0.01 0.001" condim="3" {contact}/>
      <position kp="1800" kv="95"/></default>
      <asset>{''.join(mesh_assets)}{graph_assets(scene.bodies)}</asset>
      <worldbody>
        <light pos="0 -0.3 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <camera name="overview" pos="1.05 -1.2 1.05" xyaxes="0.75 0.66 0 -0.4 0.46 0.79"/>
        <camera name="overhead" pos="0 0 1.8" xyaxes="1 0 0 0 1 0"/>
        <geom name="table" type="box" size="0.63 0.42 0.035" pos="0 0 -0.035" rgba="{numbers(scene.table_color)}"/>
        {''.join(props)}
        {conveyor_xml}
        <geom name="rail_left" type="box" size="0.02 0.4 0.02" pos="-0.56 0 0.48" rgba="0.44 0.51 0.46 1" contype="0" conaffinity="0"/>
        <geom name="rail_right" type="box" size="0.02 0.4 0.02" pos="0.56 0 0.48" rgba="0.44 0.51 0.46 1" contype="0" conaffinity="0"/>
        <geom name="post_left" type="box" size="0.022 0.022 0.24" pos="-0.56 0.36 0.24" rgba="0.36 0.43 0.39 1" contype="0" conaffinity="0"/>
        <geom name="post_right" type="box" size="0.022 0.022 0.24" pos="0.56 0.36 0.24" rgba="0.36 0.43 0.39 1" contype="0" conaffinity="0"/>
        {tray_xml}
        <body name="gantry_y" pos="0 0 0"><joint name="y" type="slide" axis="0 1 0" range="-0.35 0.35" damping="15"/>
          <geom name="crossbeam" type="box" size="0.55 0.026 0.022" pos="0 0 0.48" mass="0.1" rgba="0.55 0.63 0.57 1" contype="0" conaffinity="0"/>
          <body name="carriage"><joint name="x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="15"/>
            <geom name="motor" type="box" size="0.055 0.047 0.042" pos="0 0 0.49" mass="0.1" rgba="0.82 0.85 0.78 1" contype="0" conaffinity="0"/>
            <body name="gripper"><joint name="z" type="slide" axis="0 0 1" range="0.025 0.45" damping="15"/>
              <camera name="wrist" pos="0.07 -0.10 0.12" xyaxes="1 0 0 0 .707 .707"/>
              <geom name="shaft" type="cylinder" size="0.012 0.10" pos="0 0 0.14" mass="0.05" rgba="0.62 0.67 0.64 1" contype="0" conaffinity="0"/>
              <geom name="palm" type="box" size="0.045 0.035 0.025" pos="0 0 0.06" mass="0.1" rgba="0.85 0.86 0.79 1" contype="0" conaffinity="0"/>
              <geom name="finger_left" type="box" size="0.007 0.023 0.035" pos="-0.035 0 0.01" mass="0.01" rgba="0.75 0.79 0.72 1" contype="0" conaffinity="0"/>
              <geom name="finger_right" type="box" size="0.007 0.023 0.035" pos="0.035 0 0.01" mass="0.01" rgba="0.75 0.79 0.72 1" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
        {cube_xml}
        {''.join(object_bodies)}
        {graph_xml(scene.bodies)}
        {cloth_xml(scene.cloths)}
      </worldbody>
      <equality>{''.join(object_grasps)}{scene_equalities}</equality>
      <contact>{scene_excludes}</contact>
      <tendon>{scene_tendons}</tendon>
      <actuator><position joint="x" ctrlrange="-0.5 0.5"/><position joint="y" ctrlrange="-0.35 0.35"/><position joint="z" ctrlrange="0.025 0.45"/>{scene_actuators}</actuator>
    </mujoco>'''


class Simulator:
    id = "mujoco-cartesian-v1"
    frequency = 20

    def __init__(self, scene: Scene):
        self.scene = scene
        self.source = xml(scene)
        self.model = mujoco.MjModel.from_xml_string(self.source)
        self.data = mujoco.MjData(self.model)
        self.belt_geom_id = self.model.geom("conveyor_belt").id if scene.conveyor else None
        # Grasp welds come first in the equality list, in object order; scene constraints follow.
        self.objects = {"cube": {"body": "cube", "geom": "block", "joint": "cube_free", "eq": 0}} if scene.include_cube else {}
        for index, obj in enumerate(scene.objects, len(self.objects)):
            self.objects[obj.id] = {"body": f"object_{obj.id}", "geom": f"object_{obj.id}_geom", "joint": f"object_{obj.id}_free", "eq": index}
        self.grasp_eqs = [obj["eq"] for obj in self.objects.values()]
        self.mesh_geometries = {}
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.model.geom_dataid[i]
                start, count = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
                face_start, face_count = self.model.mesh_faceadr[mesh_id], self.model.mesh_facenum[mesh_id]
                self.mesh_geometries[i] = {"mesh_vertices": self.model.mesh_vert[start:start+count].round(6).tolist(), "mesh_faces": self.model.mesh_face[face_start:face_start+face_count].tolist()}
        self.closed = False
        self.data.joint("x").qpos[0] = -0.22
        self.data.joint("y").qpos[0] = -0.08
        self.data.joint("z").qpos[0] = 0.28
        # The gantry owns the first three controls; scene actuators follow.
        self.data.ctrl[:3] = [-0.22, -0.08, 0.28]
        self.drives = Drives(self.model, scene.actuators, scene.tendons) if scene.actuators or scene.tendons else None
        apply_initial(self.model, self.data, scene.bodies)
        self.graph = GraphState(self.model, scene.bodies) if scene.bodies else None
        self.cloth = ClothState(self.model, scene.cloths, "gripper") if scene.cloths else None
        mujoco.mj_forward(self.model, self.data)
        if self.drives:
            self.drives.apply(self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0

    def drive_conveyor(self):
        """Drive unattached boxes touching the belt, without teleporting them."""
        for obj in self.objects.values():
            self.drive_object_conveyor(obj)

    def drive_object_conveyor(self, obj):
        body_id = self.model.body(obj["body"]).id
        geom_id = self.model.geom(obj["geom"]).id
        self.data.xfrc_applied[body_id, :3] = 0
        belt = self.scene.conveyor
        if not belt or self.data.eq_active[obj["eq"]]:
            return
        touching = any(
            {contact.geom1, contact.geom2} == {geom_id, self.belt_geom_id}
            for contact in self.data.contact[:self.data.ncon]
        )
        if not touching:
            return
        target = np.zeros(2)
        target[0 if belt.direction == "x" else 1] = belt.speed
        velocity = self.data.joint(obj["joint"]).qvel[:2]
        mass = self.model.body_mass[body_id]
        force = mass * (target - velocity) / 0.025
        maximum = mass * 9.81 * self.scene.friction
        norm = np.linalg.norm(force)
        if norm > maximum:
            force *= maximum / norm
        self.data.xfrc_applied[body_id, :2] = force

    def holding(self):
        return bool(self.grasp_eqs) and bool(self.data.eq_active[self.grasp_eqs].any())

    def snapshot(self):
        frame = {
            "t": round(self.data.time, 4),
            "ee": self.data.body("gripper").xpos.round(5).tolist(),
            "euler_xyz": [0., 0., 0.], "gripper": float(self.closed), "embodiment": "cartesian-gripper-v1",
            **({"cube": self.data.body("cube").xpos.round(5).tolist(),
                "cube_speed": round(float(np.linalg.norm(self.data.joint("cube_free").qvel[:3])), 5),
                "cube_extent": (np.abs(self.data.geom("block").xmat.reshape(3, 3)) @ np.array([0.025]*3)).round(5).tolist()}
                if self.scene.include_cube else {}),
            "attached": bool(self.data.eq_active[0]) if self.scene.include_cube else self.holding(), "closed": self.closed,
            "qpos": self.data.qpos.round(7).tolist(),
            "objects": {id: {"position": self.data.body(obj["body"]).xpos.round(5).tolist(), "speed": round(float(np.linalg.norm(self.data.joint(obj["joint"]).qvel[:3])), 5), "extent": geometry_extent(self.model, self.data, obj["geom"]).round(5).tolist(), "attached": bool(self.data.eq_active[obj["eq"]])} for id, obj in self.objects.items()},
            **(self.graph.snapshot(self.data) if self.graph else {}),
            "geoms": [{"name": self.model.geom(i).name, "type": geom_kind(self.model, i, self.mesh_geometries), "size": self.model.geom_size[i].round(5).tolist(), "color": self.model.geom_rgba[i].round(3).tolist(), "position": self.data.geom_xpos[i].round(5).tolist(), "matrix": self.data.geom_xmat[i].round(6).tolist(), **self.mesh_geometries.get(i, {})} for i in range(self.model.ngeom)],
        }
        return with_cloth(frame, self.cloth, self.data, self.drives)

    def step(self, action):
        if action.get("euler_xyz") is not None and any(abs(value) > 1e-7 for value in action["euler_xyz"]):
            raise ValueError("The Cartesian fixture cannot execute end-effector rotation. Select the Franka embodiment.")
        self.data.ctrl[:3] = [action["x"], action["y"], action["z"]]
        if self.drives:
            self.drives.apply(self.data)
        self.closed = action["closed"] if action.get("gripper") is None else action["gripper"] > .5
        gap = 0.028 if self.closed else 0.042
        self.model.geom("finger_left").pos[0] = -gap
        self.model.geom("finger_right").pos[0] = gap
        if not self.closed:
            self.data.eq_active[self.grasp_eqs] = 0
        if self.cloth:
            self.cloth.update(self.data, self.closed, self.data.body("gripper").xpos)
        for _ in range(25):
            if self.closed and self.objects and not self.holding():
                candidates = sorted(self.objects.values(), key=lambda obj: np.linalg.norm(self.data.body(obj["body"]).xpos - self.data.body("gripper").xpos))
                obj = candidates[0]
                delta = self.data.body(obj["body"]).xpos - self.data.body("gripper").xpos
                if np.linalg.norm(delta[:2]) < 0.025 and abs(delta[2]) < 0.018:
                    self.model.eq_data[obj["eq"], 3:6] = delta
                    self.model.eq_data[obj["eq"], 6:10] = self.data.joint(obj["joint"]).qpos[3:7]
                    self.data.eq_active[obj["eq"]] = 1
            self.drive_conveyor()
            mujoco.mj_step(self.model, self.data)
        return self.snapshot()

    def image(self, camera="overview"):
        with mujoco.Renderer(self.model, height=360, width=480) as renderer:
            renderer.update_scene(self.data, camera=camera)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode()


def make_simulator(scene: Scene):
    from studio.robots import robot
    return robot(scene.embodiment).factory()(scene)


def default_waypoints(scene: Scene, clearance=0.2):
    if not (scene.include_cube and scene.include_tray):
        raise ValueError("Scenes without the primary block and tray need a human task teacher, skills, or explicit waypoints")
    sim = make_simulator(scene)
    x, y, z = sim.snapshot()["cube"]
    z = max(0.025, z)
    approach = [
        {"x": x, "y": y, "z": clearance, "closed": False, "seconds": 1},
        {"x": x, "y": y, "z": z, "closed": False, "seconds": 1},
        {"x": x, "y": y, "z": z, "closed": True, "seconds": 0.5},
    ]
    if scene.conveyor and on_conveyor(scene.conveyor, x, y):
        # Predict the intercept with the same physical fixture. These are
        # teacher-authored waypoints, not observations supplied to an actor.
        clearance = max(clearance, z + 0.08)
        samples = {}
        for i in range(50):
            frame = sim.step({"x": -0.22, "y": -0.08, "z": 0.28, "closed": False})
            if i + 1 in [20, 40, 50]:
                samples[i + 1] = frame["cube"]
        approach = []
        for index, closed, seconds in [(20, False, 1), (40, False, 1), (50, True, 0.5)]:
            px, py, pz = samples[index]
            approach.append({"x": round(min(0.5, max(-0.5, px)), 5), "y": round(min(0.35, max(-0.35, py)), 5),
                "z": clearance if index == 20 else max(0.025, pz), "closed": closed, "seconds": seconds})
        x, y = approach[-1]["x"], approach[-1]["y"]
    return [
        *approach,
        {"x": x, "y": y, "z": clearance, "closed": True, "seconds": 1},
        {"x": scene.tray_x, "y": scene.tray_y, "z": clearance, "closed": True, "seconds": 2},
        {"x": scene.tray_x, "y": scene.tray_y, "z": 0.07, "closed": True, "seconds": 1},
        {"x": scene.tray_x, "y": scene.tray_y, "z": 0.07, "closed": False, "seconds": 0.5},
        {"x": scene.tray_x, "y": scene.tray_y, "z": 0.28, "closed": False, "seconds": 1},
    ]


def task_waypoints(scene: Scene, task: dict, clearance=.2):
    """Teacher motion for declared operations; never supplied to an actor as ground truth."""
    if task.get("id") == "pen-into-open-pouch" or task.get("root_id") == "pen-into-open-pouch":
        from studio.workflows import pen_pouch_waypoints
        return pen_pouch_waypoints(scene)
    if task.get("teacher_skills"):
        from studio.skills import compile_skills
        return compile_skills(scene, task["teacher_skills"])
    if not task.get("operations"):
        return default_waypoints(scene, clearance)
    sim, points = make_simulator(scene), []
    for operation in task["operations"]:
        state = sim.snapshot()
        obj = state["objects"].get(operation["object_id"])
        if obj is None:
            raise ValueError("Task operation references an object missing from the scene")
        x, y, z = obj["position"]
        z = max(.025, z)
        height = min(.4, max(clearance, z + .08))
        if not points and operation["object_id"] == "cube" and scene.conveyor:
            approach = default_waypoints(scene, height)[:4]
            x, y = approach[-1]["x"], approach[-1]["y"]
        else:
            approach = [
                {"x": x, "y": y, "z": height, "closed": False, "seconds": 1},
                {"x": x, "y": y, "z": z, "closed": False, "seconds": 1},
                {"x": x, "y": y, "z": z, "closed": True, "seconds": .5},
                {"x": x, "y": y, "z": height, "closed": True, "seconds": 1}]
        tx, ty, tz = [scene.tray_x, scene.tray_y, .07] if operation["destination"] == "tray" else operation["target"]
        motion = [*approach, {"x": tx, "y": ty, "z": max(height, tz), "closed": True, "seconds": 1.5},
            {"x": tx, "y": ty, "z": tz, "closed": True, "seconds": 1}]
        if operation["destination"] != "hold":
            motion += [{"x": tx, "y": ty, "z": tz, "closed": False, "seconds": .5},
                {"x": tx, "y": ty, "z": .28, "closed": False, "seconds": 1}]
        previous = dict(zip(["x", "y", "z"], state["ee"]))
        for waypoint in motion:
            count = round(waypoint["seconds"] * sim.frequency)
            for i in range(count):
                action = {axis: previous[axis] + (waypoint[axis] - previous[axis]) * ((i + 1) / count) for axis in ["x", "y", "z"]}
                sim.step({**action, "closed": waypoint["closed"]})
            previous = waypoint
        points.extend(motion)
    return points


def record(scene: Scene, waypoints):
    sim = make_simulator(scene)
    frames = [sim.snapshot()]
    previous = dict(zip(["x", "y", "z"], frames[0]["ee"]))
    previous["closed"] = False
    previous["euler_xyz"] = frames[0].get("euler_xyz", [0, 0, 0])
    previous["gripper"] = frames[0].get("gripper", 0)
    actions = []
    for waypoint in waypoints:
        count = round(waypoint["seconds"] * sim.frequency)
        for i in range(count):
            fraction = (i+1) / count
            action = {axis: round(previous[axis] + (waypoint[axis] - previous[axis]) * fraction, 7) for axis in ["x", "y", "z"]}
            action["closed"] = waypoint["closed"]
            if waypoint.get("euler_xyz") is not None:
                start = previous.get("euler_xyz") or frames[-1].get("euler_xyz", [0, 0, 0])
                action["euler_xyz"] = interpolate_orientation(start, waypoint["euler_xyz"], fraction)
            if waypoint.get("gripper") is not None:
                start_grip = previous.get("gripper")
                if start_grip is None:
                    start_grip = float(previous.get("closed", False))
                action["gripper"] = round(start_grip + (waypoint["gripper"] - start_grip) * fraction, 7)
            actions.append(action)
            frames.append(sim.step(action))
        previous = waypoint
    for _ in range(20):
        actions.append(actions[-1])
        frames.append(sim.step(actions[-1]))
    return {"scene": scene.model_dump(), "task": task_for_scene(scene), "xml": sim.source, "frames": frames, "actions": actions, "frequency": sim.frequency, "simulator_version": mujoco.__version__}
