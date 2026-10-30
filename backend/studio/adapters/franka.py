"""Articulated Panda fixture using the official MuJoCo Menagerie model.

Joint dynamics, contact fingers, and full end-effector rotation are simulated.
Camera placement and controller calibration are research approximations, not a
validated DROID digital twin. No proximity weld is used in this embodiment.
"""
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from studio.adapters.simulation import numbers, geometry_extent, geom_kind, xml as cartesian_xml
from studio.scene_graph import ClothState, Drives, GraphState, apply_initial

BASE = np.array([-.45, -.15, 0.])
ASSET_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"


def asset_folder():
    configured = os.environ.get("STUDIO_FRANKA_ASSETS")
    return Path(configured) if configured else Path(__file__).resolve().parents[3] / ".data/models/menagerie/franka_emika_panda"


def available():
    return (asset_folder() / "panda.xml").is_file()


def rotation_matrix(euler):
    quat = np.empty(4)
    matrix = np.empty(9)
    # MuJoCo lowercase denotes rotating/local axes; SciPy's (and ICRT's)
    # uppercase XYZ denotes the same intrinsic sequence.
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=float), "xyz")
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def euler_xyz(matrix):
    pitch = math.asin(float(np.clip(matrix[0, 2], -1, 1)))
    if abs(math.cos(pitch)) < 1e-7:
        return [math.atan2(matrix[2, 1], matrix[1, 1]), pitch, 0.]
    return [math.atan2(-matrix[1, 2], matrix[2, 2]), pitch, math.atan2(-matrix[0, 1], matrix[0, 0])]


def franka_xml(scene):
    folder = asset_folder()
    if not (folder / "panda.xml").exists():
        raise ValueError("Franka assets are missing. Run integrations/icrt/prepare_franka.py first")
    root = ET.fromstring(cartesian_xml(scene))
    panda = ET.parse(folder / "panda.xml").getroot()
    root.set("model", "Franka Panda manipulation research fixture")
    world = root.find("worldbody")
    for item in list(world):
        if item.get("name") in {"gantry_y", "rail_left", "rail_right", "post_left", "post_right"}:
            world.remove(item)
    # Keep the scene's own constraints, tendons, and actuators; gantry welds and motors are dropped with their sections.
    scene_constraints = {tag: [item for item in root.find(tag) if item.get("name", "").startswith(("scene_eq_", "scene_act_")) or tag in {"contact", "tendon"}]
        for tag in ["equality", "contact", "tendon", "actuator"]}
    for tag in scene_constraints:
        root.remove(root.find(tag))
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.attrib.update(angle="radian", autolimits="true", meshdir=str((folder / "assets").resolve()))
    root.find("option").set("timestep", str(1 / 600))
    for item in panda.find("default"):
        root.find("default").append(item)
    for item in panda.find("asset"):
        root.find("asset").append(item)
    for item in panda.find("worldbody"):
        if item.tag == "body":
            item.set("pos", numbers(BASE))
            world.append(item)
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="ee", pos="0 0 .1034", size=".004", rgba="0 0 0 0")
    # Aim at the fingertip site. The former parallel camera put the site
    # outside the vertical image and ICRT's square center crop.
    # Mount beside the hand rather than behind it, so the hand body does not
    # occlude the center ray. This remains a synthetic camera, not calibration.
    camera_position = np.array([.10, 0., .035])
    backward = camera_position - np.array([0., 0., .1034])
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    ET.SubElement(hand, "camera", name="wrist", pos=numbers(camera_position),
        xyaxes=numbers(np.r_[right, up]), fovy="65")
    for tag in ["tendon", "equality", "actuator", "contact"]:
        section = panda.find(tag)
        if section is not None:
            root.append(section)
    if scene.cloths:
        from studio.scene_graph import pinch_target
        scene_constraints["equality"].insert(0, ET.Element("connect", name="cloth_pinch", body1="hand",
            body2=pinch_target(scene.cloths), anchor="0 0 0", active="false"))
    for tag, items in scene_constraints.items():
        if items:
            section = root.find(tag)
            if section is None:
                section = ET.SubElement(root, tag)
            section.extend(items)
    # The tabletop fixture uses rubber fingertip pads and a firmer gripper
    # position servo. Contact forces remain bounded by Menagerie's 100 N limit.
    # These are explicit fixture/controller assumptions, not robot calibration.
    for finger in ["left_finger", "right_finger"]:
        for geom in root.findall(f".//body[@name='{finger}']/geom"):
            if geom.get("class", "").startswith("fingertip_pad"):
                geom.set("friction", "1.2 .01 .001")
                geom.set("priority", "2")
                geom.set("condim", str(scene.finger_contact_dim))
    gripper_actuator = root.find(".//actuator/general[@name='actuator8']")
    gripper_actuator.set("gainprm", ".07843137255 0 0")
    gripper_actuator.set("biasprm", "0 -500 -25")
    # The fixed mounting flange may overlap the tabletop at its base. Disable
    # contacts for that fixed body only; the seven moving links retain contact.
    base = world.find("body[@name='link0']")
    for geom in base.findall("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for index, geom in enumerate(world.findall(".//geom")):
        if not geom.get("name"):
            geom.set("name", f"panda_geom_{index}")
    return ET.tostring(root, encoding="unicode")


def legacy_cube(objects, grasps):
    """Single-object frame keys kept for block-to-tray consumers; absent when the scene has no cube."""
    if "cube" not in objects:
        return {"attached": any(grasps.values())}
    cube = objects["cube"]
    return {"cube": cube["position"], "cube_speed": cube["speed"], "cube_extent": cube["extent"], "attached": grasps["cube"]}


class FrankaSimulator:
    id = "mujoco-franka-v1"
    frequency = 15

    def build_source(self, scene):
        return franka_xml(scene)

    def __init__(self, scene):
        self.scene = scene
        self.source = self.build_source(scene)
        self.model = mujoco.MjModel.from_xml_string(self.source)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.arm_joints = [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        self.arm_qpos = [int(self.model.jnt_qposadr[joint]) for joint in self.arm_joints]
        self.arm_dofs = [int(self.model.jnt_dofadr[joint]) for joint in self.arm_joints]
        self.arm_limits = self.model.jnt_range[self.arm_joints]
        self.ee_id = self.model.site("ee").id
        self.left_id = self.model.body("left_finger").id
        self.right_id = self.model.body("right_finger").id
        self.closed = False
        self.ik_position_error = 0.
        self.ik_rotation_error = 0.
        self.objects = {"cube": {"body": "cube", "geom": "block", "joint": "cube_free"}} if scene.include_cube else {}
        for item in scene.objects:
            self.objects[item.id] = {"body": f"object_{item.id}", "geom": f"object_{item.id}_geom", "joint": f"object_{item.id}_free"}
        self.cube_body_id = self.model.body("cube").id if scene.include_cube else None
        self.cube_geom_id = self.model.geom("block").id if scene.include_cube else None
        self.belt_geom_id = self.model.geom("conveyor_belt").id if scene.conveyor else None
        self.data.qpos[self.arm_qpos] = [0, -.785, 0, -2.356, 0, 1.571, .785]
        self.data.joint("finger_joint1").qpos[0] = .04
        self.data.joint("finger_joint2").qpos[0] = .04
        apply_initial(self.model, self.data, scene.bodies)
        self.graph = GraphState(self.model, scene.bodies) if scene.bodies else None
        self.cloth = ClothState(self.model, scene.cloths, "hand") if scene.cloths else None
        self.drives = Drives(self.model, scene.actuators, scene.tendons) if scene.actuators or scene.tendons else None
        if self.drives:
            self.drives.apply(self.data)
        mujoco.mj_forward(self.model, self.data)
        initial = self.inverse_kinematics(np.array([-.22, -.08, .28]), [math.pi, 0, 0])
        self.data.qpos[self.arm_qpos] = initial
        self.data.ctrl[:7] = initial
        self.data.ctrl[7] = 255
        mujoco.mj_forward(self.model, self.data)
        for _ in range(120):
            self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        self.mesh_geometries = {}
        for i in range(self.model.ngeom):
            if self.model.geom_group[i] == 2:  # Omit duplicate high-detail visual surfaces in the browser trace.
                continue
            # CAD vertices never repeat in every robot frame. The browser uses
            # compact link bounds; the authoritative MuJoCo cameras retain the
            # original detailed meshes. User-created static mesh props stay
            # bounded by the studio's small OBJ import limits.
            # Scene-graph meshes (named geom_*) move with their bodies; robot CAD meshes stay out.
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and (self.model.geom_bodyid[i] == 0 or self.model.geom(i).name.startswith("geom_")):
                mid = self.model.geom_dataid[i]
                start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                fstart, fcount = self.model.mesh_faceadr[mid], self.model.mesh_facenum[mid]
                self.mesh_geometries[i] = {"mesh_vertices": self.model.mesh_vert[start:start+count].round(6).tolist(),
                    "mesh_faces": self.model.mesh_face[fstart:fstart+fcount].tolist()}

    def inverse_kinematics(self, target, orientation):
        self.ik_data.qpos[:] = self.data.qpos
        target_rotation = rotation_matrix(orientation)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(70):
            mujoco.mj_forward(self.model, self.ik_data)
            current_rotation = self.ik_data.site_xmat[self.ee_id].reshape(3, 3)
            position_error = target - self.ik_data.site_xpos[self.ee_id]
            rotation_error = .5 * sum(np.cross(current_rotation[:, i], target_rotation[:, i]) for i in range(3))
            if np.linalg.norm(position_error) < .0005 and np.linalg.norm(rotation_error) < .003:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.ee_id)
            jacobian = np.vstack([jp[:, self.arm_dofs], .4 * jr[:, self.arm_dofs]])
            error = np.concatenate([position_error, .4 * rotation_error])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * .001, error)
            maximum = np.max(np.abs(delta))
            if maximum > .12:
                delta *= .12 / maximum
            current = self.ik_data.qpos[self.arm_qpos] + delta
            self.ik_data.qpos[self.arm_qpos] = np.clip(current, self.arm_limits[:, 0] + .005, self.arm_limits[:, 1] - .005)
        self.ik_position_error = float(np.linalg.norm(position_error))
        self.ik_rotation_error = float(np.linalg.norm(rotation_error))
        return self.ik_data.qpos[self.arm_qpos].copy()

    def _contact_grasps(self, bodies=None):
        """Held means the gripper is closed and both finger bodies touch the object."""
        bodies = bodies or {name: self.model.body(item["body"]).id for name, item in self.objects.items()}
        touches = {name: set() for name in bodies}
        for contact in self.data.contact[:self.data.ncon]:
            pair = [int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])]
            for name, id in bodies.items():
                if id in pair:
                    touches[name].update(body for body in pair if body in {self.left_id, self.right_id})
        return {name: self.closed and fingers == {self.left_id, self.right_id} for name, fingers in touches.items()}

    def drive_conveyor(self):
        belt = self.scene.conveyor
        if self.cube_body_id is None:
            return
        self.data.xfrc_applied[self.cube_body_id, :3] = 0
        if not belt:
            return
        if not any({c.geom1, c.geom2} == {self.cube_geom_id, self.belt_geom_id} for c in self.data.contact[:self.data.ncon]):
            return
        target = np.zeros(2)
        target[0 if belt.direction == "x" else 1] = belt.speed
        mass = self.model.body_mass[self.cube_body_id]
        force = mass * (target - self.data.joint("cube_free").qvel[:2]) / .025
        norm, maximum = np.linalg.norm(force), mass * 9.81 * self.scene.friction
        self.data.xfrc_applied[self.cube_body_id, :2] = force * min(1, maximum / max(norm, 1e-9))

    def step(self, action):
        target = np.asarray([action[key] for key in ["x", "y", "z"]], dtype=float)
        orientation = np.asarray(action.get("euler_xyz", [math.pi, 0, 0]), dtype=float)
        if orientation.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(orientation).all():
            raise ValueError("Franka requires finite XYZ and Euler XYZ targets")
        self.data.ctrl[:7] = self.inverse_kinematics(target, orientation)
        grip = float(action.get("gripper", float(action["closed"])))
        if not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError("Franka gripper target must be between zero and one")
        self.closed = grip > .5
        self.data.ctrl[7] = 255 * (1 - grip)
        if self.cloth:
            self.cloth.update(self.data, self.closed, self.data.site("ee").xpos)
        if self.drives:
            self.drives.apply(self.data)
        for _ in range(40):
            self.data.qfrc_applied[self.arm_dofs] = self.data.qfrc_bias[self.arm_dofs]
            self.drive_conveyor()
            mujoco.mj_step(self.model, self.data)
        return self.snapshot()

    def snapshot(self):
        from studio.adapters.simulation import with_cloth
        return with_cloth(self._snapshot(), self.cloth, self.data, self.drives)

    def _snapshot(self):
        grasps = self._contact_grasps()
        objects = {}
        for name, item in self.objects.items():
            geom = self.data.geom(item["geom"])
            objects[name] = {"position": self.data.body(item["body"]).xpos.round(5).tolist(),
                "speed": round(float(np.linalg.norm(self.data.joint(item["joint"]).qvel[:3])), 5),
                "extent": geometry_extent(self.model, self.data, item["geom"]).round(5).tolist(),
                "attached": grasps[name]}
        geoms = []
        for i in range(self.model.ngeom):
            if self.model.geom_group[i] == 2:
                continue
            geoms.append({"name": self.model.geom(i).name, "type": geom_kind(self.model, i, self.mesh_geometries),
                "size": self.model.geom_size[i].round(5).tolist(), "color": self.model.geom_rgba[i].round(3).tolist(),
                "position": self.data.geom_xpos[i].round(5).tolist(), "matrix": self.data.geom_xmat[i].round(6).tolist(),
                **self.mesh_geometries.get(i, {})})
        return {"t": round(self.data.time, 5), "ee": self.data.site("ee").xpos.round(5).tolist(),
            "euler_xyz": np.round(euler_xyz(self.data.site("ee").xmat.reshape(3, 3)), 6).tolist(),
            "gripper": float(np.clip(1 - self.data.joint("finger_joint1").qpos[0] / .04, 0, 1)),
            "closed": self.closed, "embodiment": "franka-panda-v1", "robot_base": BASE.tolist(),
            **legacy_cube(objects, grasps), "objects": objects,
            "qpos": self.data.qpos.round(7).tolist(), "geoms": geoms,
            **(self.graph.snapshot(self.data, self._contact_grasps(self.graph.bodies)) if self.graph else {}),
            "ik_position_error": round(self.ik_position_error, 6), "ik_rotation_error": round(self.ik_rotation_error, 6),
            "grasp_kind": "two-finger contact", "robot_asset_commit": ASSET_COMMIT,
            "wrist_camera_version": "side-mounted-pinch-v3"}

    def image(self, camera="overview"):
        from studio.adapters.icrt import CameraClient
        renderer = CameraClient()
        try:
            result = renderer.render(self.source, [self.snapshot()])[0]
            return result["wrist" if camera == "wrist" else "side"]
        finally:
            renderer.close()
