"""Generic articulated-arm fixture assembled from MuJoCo Menagerie models.

An `ArmConfig` names the arm and gripper models, how they attach, the joints the
controller drives, the fingertip site, and which bodies count as left and right
fingers for contact grasp detection. MuJoCo's model editor (MjSpec) attaches the
parts with prefixed names; the studio's scene is compiled into the same model.
Controller gains and camera placement are research assumptions, not calibration.
"""
from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from studio.adapters.franka import euler_xyz, legacy_cube, rotation_matrix
from studio.adapters.simulation import geom_kind, xml as cartesian_xml
from studio.scene_graph import ClothState, Drives, GraphState, apply_initial

MENAGERIE_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
GANTRY = {"gantry_y", "rail_left", "rail_right", "post_left", "post_right"}


def menagerie_folder():
    from studio.adapters.franka import asset_folder
    return asset_folder().parent


@dataclass(frozen=True)
class Part:
    folder: str        # Menagerie folder name.
    file: str          # Model XML inside that folder.
    root: str          # Body to attach.
    prefix: str        # Name prefix applied by MjSpec.
    site: str | None = None  # Site on the previous part to attach to; None attaches to the robot base frame.


@dataclass(frozen=True)
class ArmConfig:
    id: str
    name: str
    parts: tuple[Part, ...]
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    home: tuple[float, ...]
    ee_site: str
    # The single gripper command (0 open .. 1 closed) interpolates every hand actuator between these poses:
    # one actuator for a parallel gripper, a whole-hand grasp synergy for a dexterous hand.
    hand_actuators: tuple[str, ...]
    hand_open: tuple[float, ...]
    hand_closed: tuple[float, ...]
    gripper_joint: str
    gripper_joint_closed: float
    left_bodies: tuple[str, ...]   # One side of an opposing grasp (left finger, or the thumb).
    right_bodies: tuple[str, ...]  # The other side (right finger, or the other fingers).
    wrist_body: str
    # Optional grasp-point site added in the frame of a site on the arm, for hands without one.
    ee_from: tuple[str, tuple[float, float, float]] | None = None
    grasp_kind: str = "two-finger contact"
    # Optional position-servo stiffness for the hand actuators (fixture assumption, not calibration).
    hand_kp: float | None = None
    # Mobile manipulators: floor height below the tabletop (adds a floor and table legs) and IK travel limits
    # for joints the model leaves unlimited, e.g. the base slides, so the base stays outside the table footprint.
    floor_z: float | None = None
    ik_limits: tuple = ()
    base_yaw: float = 0.  # Heading of the robot's mounting frame, radians about world Z.
    start: tuple[float, float, float] = (-.22, -.08, .28)  # Initial fingertip position; robots with short reach start lower.
    base: tuple[float, float, float] = (-.45, -.15, 0.)
    frequency: int = 15
    substeps: int = 40
    timestep: float = 1 / 600
    option: dict = field(default_factory=dict)

    def available(self):
        return all((menagerie_folder() / part.folder / part.file).is_file() for part in self.parts)

    def asset_folders(self):
        return [(menagerie_folder() / part.folder).resolve() for part in self.parts]


UR5E_2F85 = ArmConfig(
    id="ur5e-2f85-v1", name="UR5e + Robotiq 2F-85",
    parts=(Part("universal_robots_ur5e", "ur5e.xml", "base", "ur5e_"),
        Part("robotiq_2f85", "2f85.xml", "base_mount", "gripper_", site="attachment_site")),
    arm_joints=tuple("ur5e_" + name for name in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]),
    arm_actuators=tuple("ur5e_" + name for name in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]),
    home=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.),
    ee_site="ur5e_gripper_pinch", hand_actuators=("ur5e_gripper_fingers_actuator",), hand_open=(0,), hand_closed=(255,),
    gripper_joint="ur5e_gripper_right_driver_joint", gripper_joint_closed=.8,
    left_bodies=("ur5e_gripper_left_pad", "ur5e_gripper_left_follower"),
    right_bodies=("ur5e_gripper_right_pad", "ur5e_gripper_right_follower"),
    wrist_body="ur5e_gripper_base", option={"cone": mujoco.mjtCone.mjCONE_ELLIPTIC, "impratio": 10})

FINGERS = ("ff", "mf", "rf")
# Allegro power-grasp synergy: finger flexion joints close together while the thumb swings into opposition.
ALLEGRO_OPEN = tuple([0, 0, 0, 0] * 3 + [.3, 0, 0, 0])
ALLEGRO_CLOSED = tuple([0, 1.0, 1.0, 1.0] * 3 + [1.3, .6, .9, .9])
UR5E_ALLEGRO = ArmConfig(
    id="ur5e-allegro-v1", name="UR5e + Allegro hand",
    parts=(Part("universal_robots_ur5e", "ur5e.xml", "base", "ur5e_"),
        Part("wonik_allegro", "right_hand.xml", "palm", "hand_", site="attachment_site")),
    arm_joints=UR5E_2F85.arm_joints, arm_actuators=UR5E_2F85.arm_actuators, home=UR5E_2F85.home,
    # Grasp point where a power-grasped object sits: between palm, thumb, and curled fingers.
    ee_site="ur5e_grasp", ee_from=("ur5e_attachment_site", (-.005, -.02, .045)),
    hand_actuators=tuple(f"ur5e_hand_{finger}a{k}" for finger in (*FINGERS, "th") for k in range(4)),
    hand_open=ALLEGRO_OPEN, hand_closed=ALLEGRO_CLOSED,
    gripper_joint="ur5e_hand_mfj1", gripper_joint_closed=1.0,
    left_bodies=tuple(f"ur5e_hand_th_{part}" for part in ("medial", "distal", "tip")),
    # Power grasps press the object into the palm and proximal phalanges, not only the fingertips.
    right_bodies=("ur5e_hand_palm", *(f"ur5e_hand_{finger}_{part}" for finger in FINGERS for part in ("proximal", "medial", "distal", "tip"))),
    wrist_body="ur5e_hand_palm", grasp_kind="dexterous hand contact (thumb against fingers)", hand_kp=6,
    option={"cone": mujoco.mjtCone.mjCONE_ELLIPTIC, "impratio": 10})

TIDYBOT = ArmConfig(
    id="tidybot-v1", name="Stanford TidyBot (mobile base + Kinova Gen3 + 2F-85)",
    parts=(Part("stanford_tidybot", "tidybot.xml", "base_link", "tidy_"),),
    arm_joints=tuple("tidy_" + name for name in ["joint_x", "joint_y", "joint_th", *(f"joint_{i}" for i in range(1, 8))]),
    arm_actuators=tuple("tidy_" + name for name in ["joint_x", "joint_y", "joint_th", *(f"joint_{i}" for i in range(1, 8))]),
    home=(0, 0, 0, 0, .26179939, 3.14159265, -2.26892803, 0, .95993109, 1.57079633),
    ee_site="tidy_pinch_site", hand_actuators=("tidy_fingers_actuator",), hand_open=(0,), hand_closed=(255,),
    gripper_joint="tidy_right_driver_joint", gripper_joint_closed=.8,
    left_bodies=("tidy_left_pad", "tidy_left_follower"), right_bodies=("tidy_right_pad", "tidy_right_follower"),
    wrist_body="tidy_base", grasp_kind="two-finger contact (mobile base, whole-body IK)",
    # The base stands on a floor 0.5 m below the tabletop (a low table), beside the long -Y edge and facing it.
    # Rotated 90°, its X slide points at the table (it may back away but not drive under) and its Y slide runs along it.
    base=(-.1, -.75, -.5), base_yaw=math.pi / 2, floor_z=-.5, start=(-.22, -.2, .2),
    ik_limits=(("tidy_joint_x", -.15, .04), ("tidy_joint_y", -.45, .45), ("tidy_joint_th", -1.2, 1.2)),
    option={"cone": mujoco.mjtCone.mjCONE_ELLIPTIC, "impratio": 10})

CONFIGS = {config.id: config for config in [UR5E_2F85, UR5E_ALLEGRO, TIDYBOT]}


def _load_part(part):
    folder = menagerie_folder() / part.folder
    if not (folder / part.file).is_file():
        raise ValueError(f"Robot assets for {part.folder} are missing. Run integrations/icrt/prepare_arms.py first")
    spec = mujoco.MjSpec.from_file(str(folder / part.file))
    meshdir = folder / (spec.meshdir or "")
    for mesh in spec.meshes:
        mesh.file = str((meshdir / mesh.file).resolve())
    for texture in spec.textures:
        if texture.file:
            texture.file = str((meshdir / texture.file).resolve())
    spec.meshdir = ""
    spec.texturedir = ""
    for key in list(spec.keys):
        spec.delete(key)
    for light in list(spec.lights):
        spec.delete(light)
    # Name every geom so recorded frames and the browser can track it.
    for index, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"geom_{index}"
    return spec


def arm_spec(scene, config: ArmConfig):
    root = ET.fromstring(cartesian_xml(scene))
    root.set("model", config.name + " manipulation research fixture")
    world = root.find("worldbody")
    for item in list(world):
        if item.get("name") in GANTRY:
            world.remove(item)
    actuators = root.find("actuator")
    for motor in [item for item in actuators if item.get("joint") in {"x", "y", "z"}]:
        actuators.remove(motor)  # Gantry motors; scene actuators stay.
    equality = root.find("equality")
    for weld in [item for item in equality if item.get("name", "").startswith("grasp") or item.get("name") == "cloth_pinch"]:
        equality.remove(weld)
    root.find("option").set("timestep", str(config.timestep))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    parts = [_load_part(part) for part in config.parts]
    # Matching solver options on every spec avoids attach conflicts; the scene keeps the robot's choice.
    for item in [spec, *parts]:
        for key, value in config.option.items():
            setattr(item.option, key, value)
    if config.hand_kp:
        for actuator in parts[-1].actuators:
            actuator.gainprm[0] = config.hand_kp
            actuator.biasprm[1] = -config.hand_kp
    for previous, (part, child) in zip(parts, list(zip(config.parts, parts))[1:]):
        previous.site(part.site).attach_body(child.body(part.root), part.prefix, "")
    if config.floor_z is not None:
        # Floor and table legs for robots that stand beside the table rather than on it.
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.6, 1.2, .01],
            pos=[0, 0, config.floor_z - .01], rgba=[.36, .37, .36, 1], contype=0, conaffinity=0)
        height = (-.07 - config.floor_z) / 2
        for index, (x, y) in enumerate([(-.58, -.37), (-.58, .37), (.58, -.37), (.58, .37)]):
            spec.worldbody.add_geom(name=f"table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.03, .03, height],
                pos=[x, y, config.floor_z + height], rgba=[.3, .32, .3, 1])
    frame = spec.worldbody.add_frame(pos=list(config.base), quat=[math.cos(config.base_yaw / 2), 0, 0, math.sin(config.base_yaw / 2)])
    frame.attach_body(parts[0].body(config.parts[0].root), config.parts[0].prefix, "")
    if scene.cloths:
        from studio.scene_graph import pinch_target
        spec.add_equality(name="cloth_pinch", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=config.wrist_body, name2=pinch_target(scene.cloths), active=False)
    if config.ee_from:
        # Grasp point for hands without one: offset in the frame of a site on the arm.
        anchor, offset = config.ee_from
        site = spec.site(anchor)
        quat = np.asarray(site.quat, dtype=float)
        quat /= np.linalg.norm(quat)  # Menagerie writes some quaternions unnormalized, e.g. "-1 1 0 0".
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        position = np.asarray(site.pos) + rotation.reshape(3, 3) @ np.asarray(offset)
        site.parent.add_site(name=config.ee_site, pos=position.tolist(), quat=quat.tolist(), size=[.004, .004, .004], rgba=[0, 0, 0, 0])
    # Synthetic wrist camera beside the gripper, looking along the fingers.
    wrist = spec.body(config.wrist_body)
    wrist.add_camera(name="wrist", pos=[.1, 0, .03], xyaxes=[0, -1, 0, -.3, 0, 1], fovy=65)
    return spec


def robot_visual_bodies(model, config):
    """Robot links that carry visual meshes; their poses are recorded for browser CAD display."""
    prefix = config.parts[0].prefix
    return sorted({int(model.geom_bodyid[i]) for i in range(model.ngeom)
        if model.geom_group[i] == 2 and model.body(int(model.geom_bodyid[i])).name.startswith(prefix)})


class ArmSimulator:
    config: ArmConfig = UR5E_2F85

    @property
    def id(self):
        return "mujoco-" + self.config.id

    @property
    def frequency(self):
        return self.config.frequency

    def __init__(self, scene):
        config = self.config
        self.scene = scene
        spec = arm_spec(scene, config)
        self.model = spec.compile()
        self.source = spec.to_xml()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in config.arm_joints]
        self.arm_qpos = [int(self.model.jnt_qposadr[j]) for j in joints]
        self.arm_dofs = [int(self.model.jnt_dofadr[j]) for j in joints]
        # Unlimited joints report a [0, 0] range; give them generous bounds unless the config caps their travel.
        limited = self.model.jnt_limited[joints].astype(bool)
        self.arm_limits = np.where(limited[:, None], self.model.jnt_range[joints], [[-1e3, 1e3]])
        for name, low, high in config.ik_limits:
            self.arm_limits[list(config.arm_joints).index(name)] = [low, high]
        self.arm_ctrl = [self.model.actuator(name).id for name in config.arm_actuators]
        self.hand_ctrl = [self.model.actuator(name).id for name in config.hand_actuators]
        self.hand_open, self.hand_closed = np.asarray(config.hand_open, dtype=float), np.asarray(config.hand_closed, dtype=float)
        self.ee_id = self.model.site(config.ee_site).id
        self.left = {self.model.body(name).id for name in config.left_bodies}
        self.right = {self.model.body(name).id for name in config.right_bodies}
        self.closed = False
        self.ik_position_error = self.ik_rotation_error = 0.
        self.objects = {"cube": {"body": "cube", "geom": "block", "joint": "cube_free"}} if scene.include_cube else {}
        for item in scene.objects:
            self.objects[item.id] = {"body": f"object_{item.id}", "geom": f"object_{item.id}_geom", "joint": f"object_{item.id}_free"}
        self.data.qpos[self.arm_qpos] = config.home
        apply_initial(self.model, self.data, scene.bodies)
        self.graph = GraphState(self.model, scene.bodies) if scene.bodies else None
        self.cloth = ClothState(self.model, scene.cloths, config.wrist_body) if scene.cloths else None
        self.drives = Drives(self.model, scene.actuators, scene.tendons) if scene.actuators or scene.tendons else None
        if self.drives:
            self.drives.apply(self.data)
        mujoco.mj_forward(self.model, self.data)
        # Large first moves (e.g. a mobile base leaving its home pose) need several solves to converge.
        for _ in range(8):
            initial = self.inverse_kinematics(np.array(config.start), [math.pi, 0, 0])
            self.data.qpos[self.arm_qpos] = initial
            mujoco.mj_forward(self.model, self.data)
            if self.ik_position_error < .001:
                break
        self.data.ctrl[self.arm_ctrl] = initial
        self.data.ctrl[self.hand_ctrl] = self.hand_open
        # Start the hand in its open pose rather than at joint zero.
        for value, actuator in zip(self.hand_open, self.hand_ctrl):
            if self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT:
                self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[actuator][0]]] = value
        mujoco.mj_forward(self.model, self.data)
        for _ in range(120):
            self._compensate()
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        self.recorded = [i for i in range(self.model.ngeom) if self.model.geom_group[i] != 2]
        self.visual_bodies = robot_visual_bodies(self.model, config)

    def _compensate(self):
        self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]

    def inverse_kinematics(self, target, orientation):
        """Damped least squares on the fingertip site, seeded from the current pose."""
        self.ik_data.qpos[:] = self.data.qpos
        target_rotation = rotation_matrix(orientation)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(80):
            mujoco.mj_forward(self.model, self.ik_data)
            current = self.ik_data.site_xmat[self.ee_id].reshape(3, 3)
            position_error = target - self.ik_data.site_xpos[self.ee_id]
            rotation_error = .5 * sum(np.cross(current[:, i], target_rotation[:, i]) for i in range(3))
            if np.linalg.norm(position_error) < .0005 and np.linalg.norm(rotation_error) < .003:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.ee_id)
            jacobian = np.vstack([jp[:, self.arm_dofs], .4 * jr[:, self.arm_dofs]])
            error = np.concatenate([position_error, .4 * rotation_error])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * .001, error)
            delta *= min(1, .12 / max(np.max(np.abs(delta)), 1e-12))
            current_q = self.ik_data.qpos[self.arm_qpos] + delta
            self.ik_data.qpos[self.arm_qpos] = np.clip(current_q, self.arm_limits[:, 0] + .005, self.arm_limits[:, 1] - .005)
        self.ik_position_error = float(np.linalg.norm(position_error))
        self.ik_rotation_error = float(np.linalg.norm(rotation_error))
        return self.ik_data.qpos[self.arm_qpos].copy()

    def _contact_grasps(self, bodies=None):
        """Held means the gripper is closed and left and right finger bodies both touch the object."""
        bodies = bodies or {name: self.model.body(item["body"]).id for name, item in self.objects.items()}
        touches = {name: [False, False] for name in bodies}
        for contact in self.data.contact[:self.data.ncon]:
            pair = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            for name, id in bodies.items():
                if id in pair:
                    touches[name][0] |= bool(pair & self.left)
                    touches[name][1] |= bool(pair & self.right)
        return {name: self.closed and all(sides) for name, sides in touches.items()}

    def step(self, action):
        target = np.asarray([action[key] for key in "xyz"], dtype=float)
        orientation = np.asarray(action.get("euler_xyz", [math.pi, 0, 0]), dtype=float)
        if orientation.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(orientation).all():
            raise ValueError(f"{self.config.name} requires finite XYZ and Euler XYZ targets")
        self.data.ctrl[self.arm_ctrl] = self.inverse_kinematics(target, orientation)
        grip = float(action.get("gripper", float(action["closed"])))
        if not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError(f"{self.config.name} gripper target must be between zero and one")
        self.closed = grip > .5
        config = self.config
        self.data.ctrl[self.hand_ctrl] = self.hand_open + (self.hand_closed - self.hand_open) * grip
        if self.cloth:
            # The robot's pinch point (fingertips for a hand) decides which cloth vertex it can hold.
            from studio.robots import robot
            offset = np.asarray(robot(config.id).tool_offset("pinch"))
            point = self.data.site_xpos[self.ee_id] + self.data.site_xmat[self.ee_id].reshape(3, 3) @ offset
            self.cloth.update(self.data, self.closed, point)
        if self.drives:
            self.drives.apply(self.data)
        for _ in range(config.substeps):
            self._compensate()
            mujoco.mj_step(self.model, self.data)
        return self.snapshot()

    def _geom(self, i):
        """Browser geometry. Robot collision meshes are sent as their local bounding boxes."""
        model, data = self.model, self.data
        position, size, kind = data.geom_xpos[i], model.geom_size[i], geom_kind(model, i)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and (model.geom_bodyid[i] == 0 or model.geom(i).name.startswith("geom_")):
            mid = model.geom_dataid[i]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fstart, fcount = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            return {"name": model.geom(i).name, "type": "mesh", "size": np.round(size, 5).tolist(),
                "color": model.geom_rgba[i].round(3).tolist(), "position": np.round(position, 5).tolist(),
                "matrix": data.geom_xmat[i].round(6).tolist(), "mesh_vertices": model.mesh_vert[start:start + count].round(6).tolist(),
                "mesh_faces": model.mesh_face[fstart:fstart + fcount].tolist()}
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            center, half = model.geom_aabb[i][:3], model.geom_aabb[i][3:]
            position, size, kind = position + data.geom_xmat[i].reshape(3, 3) @ center, half, "box"
        return {"name": model.geom(i).name, "type": kind, "size": np.round(size, 5).tolist(),
            "color": model.geom_rgba[i].round(3).tolist(), "position": np.round(position, 5).tolist(),
            "matrix": data.geom_xmat[i].round(6).tolist()}

    def snapshot(self):
        from studio.adapters.simulation import with_cloth
        return with_cloth(self._snapshot(), self.cloth, self.data, self.drives)

    def _snapshot(self):
        grasps = self._contact_grasps()
        from studio.adapters.simulation import geometry_extent
        objects = {name: {"position": self.data.body(item["body"]).xpos.round(5).tolist(),
            "speed": round(float(np.linalg.norm(self.data.joint(item["joint"]).qvel[:3])), 5),
            "extent": geometry_extent(self.model, self.data, item["geom"]).round(5).tolist(),
            "attached": grasps[name]} for name, item in self.objects.items()}
        config = self.config
        grip = self.data.joint(config.gripper_joint).qpos[0] / config.gripper_joint_closed
        return {"t": round(self.data.time, 5), "ee": self.data.site_xpos[self.ee_id].round(5).tolist(),
            "euler_xyz": np.round(euler_xyz(self.data.site_xmat[self.ee_id].reshape(3, 3)), 6).tolist(),
            "gripper": float(np.clip(grip, 0, 1)), "closed": self.closed, "embodiment": config.id,
            "robot_base": list(config.base), **legacy_cube(objects, grasps), "objects": objects,
            "qpos": self.data.qpos.round(7).tolist(), "geoms": [self._geom(i) for i in self.recorded],
            **(self.graph.snapshot(self.data, self._contact_grasps(self.graph.bodies)) if self.graph else {}),
            "ik_position_error": round(self.ik_position_error, 6), "ik_rotation_error": round(self.ik_rotation_error, 6),
            "grasp_kind": config.grasp_kind, "robot_asset_commit": MENAGERIE_COMMIT,
            "robot_bodies": {self.model.body(b).name: [*self.data.xpos[b].round(5).tolist(), *self.data.xquat[b].round(6).tolist()]
                for b in self.visual_bodies}}

    def image(self, camera="overview"):
        with mujoco.Renderer(self.model, height=360, width=480) as renderer:
            renderer.update_scene(self.data, camera=camera)
            import base64, io
            from PIL import Image
            buffer = io.BytesIO()
            Image.fromarray(renderer.render()).save(buffer, format="JPEG", quality=80)
            return base64.b64encode(buffer.getvalue()).decode()


class UR5eSimulator(ArmSimulator):
    config = UR5E_2F85


class UR5eAllegroSimulator(ArmSimulator):
    config = UR5E_ALLEGRO


def ur5e_available():
    return UR5E_2F85.available()


def allegro_available():
    return UR5E_ALLEGRO.available()


class TidyBotSimulator(ArmSimulator):
    config = TIDYBOT


def tidybot_available():
    return TIDYBOT.available()
