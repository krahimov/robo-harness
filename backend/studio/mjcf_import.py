"""Import articulated objects from a sanitized subset of MuJoCo MJCF.

The upload is never simulated as-is. A whitelist check rejects file references,
includes, assets, actuators, sensors, constraints, plugins, and entity
declarations. MuJoCo then compiles the remaining bodies in memory, which
resolves defaults, classes, angle units, and fromto placement. The compiled
bodies, joints, primitive geoms, and sites are translated into the studio's own
scene graph, which is validated and compiled like any other scene.
"""
import math
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from studio.scene_graph import MAX_BODIES

MAX_XML_BYTES = 256 * 1024
MAX_ELEMENTS = 4000
KEEP = {"mujoco", "compiler", "default", "worldbody", "body", "frame", "joint", "freejoint", "geom", "site", "inertial", "asset", "mesh",
    "equality", "connect", "weld", "contact", "exclude", "tendon", "fixed", "spatial", "actuator", "motor", "position", "velocity"}
TRANSMISSIONS = {"site", "body", "jointinparent", "cranksite", "slidersite", "refsite", "cranklength"}
MESH_ATTRIBUTES = {"name", "class", "file", "vertex", "face", "scale", "refpos", "refquat", "inertia"}
MAX_COMPANION_FILES = 8
DROP = {"option", "size", "visual", "statistic", "camera", "light"}
COMPILER_KEEP = {"angle", "eulerseq", "autolimits", "inertiafromgeom", "boundmass", "boundinertia"}
FORBIDDEN_ATTRIBUTES = {"meshdir", "texturedir", "assetdir", "material", "hfield", "texture", "plugin",
    "instance", "fitscale", "content_type"}
SHAPES = {int(mujoco.mjtGeom.mjGEOM_MESH): "mesh", int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}
JOINTS = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}


class MjcfError(ValueError):
    """User-facing import failure; never includes the uploaded document."""


def _name(value, prefix, fallback, used):
    base = re.sub(r"[^A-Za-z0-9_]", "_", value or "") or fallback
    # A body named like the import itself ("cabinet" imported as "cabinet") keeps one copy of the name.
    name = (prefix.rstrip("_") if base == prefix.rstrip("_") else prefix + base)[:40]
    if not name[0].isalpha():
        name = ("m" + name)[:40]
    candidate, index = name, 2
    while candidate in used:
        suffix = f"_{index}"
        candidate, index = name[:40 - len(suffix)] + suffix, index + 1
    used.add(candidate)
    return candidate


def _companion_mesh(name, files):
    """Parse an uploaded OBJ or STL that a <mesh file=...> names; nothing is read from disk."""
    from studio.assets import MAX_FACES, MAX_VERTICES, parse_stl
    key = name.replace("\\", "/").rsplit("/", 1)[-1]
    if key not in files:
        raise MjcfError(f"Mesh file {key} was not uploaded with the MJCF. Add it as a companion file.")
    data = files[key]
    if key.lower().endswith(".stl"):
        try:
            return parse_stl(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MjcfError(f"{key}: {exc}") from None
    if key.lower().endswith(".obj"):
        vertices, faces = [], []
        for line in data.decode("utf-8", errors="strict").splitlines():
            parts = line.split("#", 1)[0].split()
            if parts and parts[0] == "v":
                vertices.append([float(v) for v in parts[1:4]])
            elif parts and parts[0] == "f":
                index = [int(v.split("/")[0]) for v in parts[1:]]
                index = [i - 1 if i > 0 else len(vertices) + i for i in index]
                faces += [[index[0], index[k], index[k + 1]] for k in range(1, len(index) - 1)]
        if len(vertices) > MAX_VERTICES or len(faces) > MAX_FACES:
            raise MjcfError(f"{key} exceeds {MAX_VERTICES} vertices or {MAX_FACES} triangles; simplify it first")
        return vertices, faces
    raise MjcfError(f"Mesh file {key} must be OBJ or STL")


def sanitize(xml: str, files=None):
    """Return the whitelisted MJCF text and warnings for removed decorative elements."""
    files = files or {}
    if len(files) > MAX_COMPANION_FILES:
        raise MjcfError(f"Upload at most {MAX_COMPANION_FILES} companion mesh files")
    if len(xml.encode()) > MAX_XML_BYTES:
        raise MjcfError(f"MJCF files can be at most {MAX_XML_BYTES // 1024} KiB")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", xml, re.I):
        raise MjcfError("MJCF with DOCTYPE or entity declarations is not accepted")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise MjcfError("The file is not well-formed XML") from None
    if root.tag != "mujoco":
        raise MjcfError("The root element must be <mujoco>")
    elements = list(root.iter())
    for spatial in root.iter("spatial"):
        if any(child.tag != "site" for child in spatial):
            raise MjcfError("Spatial tendons may only pass through sites; geometry wrapping and pulleys are not imported")
    if len(elements) > MAX_ELEMENTS:
        raise MjcfError(f"MJCF imports can contain at most {MAX_ELEMENTS} elements")
    unsupported = sorted({e.tag for e in elements} - KEEP - DROP)
    if unsupported:
        raise MjcfError("Unsupported MJCF elements: " + ", ".join(unsupported) +
            ". Imports accept bodies, joints, primitive and mesh geoms, sites, defaults, equality constraints, contact exclusions, "
            "fixed or site tendons, and motor/position/velocity actuators.")
    warnings = []
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in DROP:
                parent.remove(child)
                warnings.append(f"Ignored <{child.tag}>; the studio supplies its own cameras, lights, and solver options.")
    for element in root.iter():
        bad = sorted(set(element.attrib) & FORBIDDEN_ATTRIBUTES)
        if bad:
            raise MjcfError(f"<{element.tag}> uses unsupported attributes: {', '.join(bad)}. File, mesh, and material references are not imported.")
        if "file" in element.attrib and element.tag != "mesh":
            raise MjcfError(f"<{element.tag}> cannot reference files")
        if element.tag == "mesh":
            bad = sorted(set(element.attrib) - MESH_ATTRIBUTES)
            if bad:
                raise MjcfError("<mesh> uses unsupported attributes: " + ", ".join(bad))
            if "file" in element.attrib:
                vertices, faces = _companion_mesh(element.attrib.pop("file"), files)
                element.set("vertex", " ".join(str(v) for point in vertices for v in point))
                element.set("face", " ".join(str(i) for face in faces for i in face))
        if element.tag == "geom" and element.get("type", "sphere") not in {"box", "sphere", "capsule", "cylinder", "mesh"}:
            raise MjcfError(f"Geom type {element.get('type')} is unsupported; use box, sphere, capsule, cylinder, or mesh")
        if element.tag in {"motor", "position", "velocity"} and set(element.attrib) & TRANSMISSIONS:
            raise MjcfError(f"<{element.tag}> uses an unsupported transmission; actuators may drive a joint or a tendon")
        if element.tag == "compiler":
            for key in set(element.attrib) - COMPILER_KEEP:
                del element.attrib[key]
    return ET.tostring(root, encoding="unicode"), list(dict.fromkeys(warnings))


def _euler(quat):
    from studio.adapters.franka import euler_xyz
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return [round(float(v), 6) for v in euler_xyz(matrix.reshape(3, 3))]


def _volume(shape, size):
    if shape in {"box", "mesh"}:
        return 8 * size[0] * size[1] * size[2]
    if shape == "sphere":
        return 4 / 3 * math.pi * size[0] ** 3
    if shape == "cylinder":
        return math.pi * size[0] ** 2 * 2 * size[2]
    return math.pi * size[0] ** 2 * (2 * size[2] + 4 / 3 * size[0])


def preview_store(normalized):
    """Content hash in place of a stored asset, for previews that must not write records."""
    import hashlib, json
    return hashlib.md5(json.dumps(normalized["vertices"]).encode()).hexdigest()


def database_store(name):
    def store(normalized):
        from studio import storage as db
        return db.save("asset", {"name": name, "kind": "mesh", "source": "mjcf", **normalized})["id"]
    return store


def convert(xml: str, prefix: str, position=(0., 0., 0.), yaw=0., files=None, store=preview_store):
    """Translate sanitized MJCF into scene-graph root bodies placed at position/yaw on the table.

    `store` receives each normalized mesh and returns its asset ID.
    """
    clean, warnings = sanitize(xml, files)
    try:
        model = mujoco.MjModel.from_xml_string(clean)
    except ValueError as exc:
        # MuJoCo messages can quote element names but never file contents (files are rejected above).
        raise MjcfError("MuJoCo could not compile the MJCF: " + str(exc).splitlines()[0][:300]) from None
    if model.nbody - 1 > MAX_BODIES:
        raise MjcfError(f"MJCF imports can contain at most {MAX_BODIES} bodies")
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", prefix)[:16]
    if not prefix or not prefix[0].isalpha():
        raise MjcfError("Give the imported object a name starting with a letter")
    prefix += "_"
    used_bodies, used_joints, used_sites, used_drives = set(), set(), set(), set()
    nodes, joint_names, site_names = {}, {}, {}

    def name_of(kind, index):
        return mujoco.mj_id2name(model, kind, index)

    def geoms_for(body):
        ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
        total = sum(_volume(SHAPES.get(int(model.geom_type[i]), "box"), _size(i)) for i in ids) or 1
        result, used = [], set()
        for i in ids:
            shape = SHAPES.get(int(model.geom_type[i]))
            if shape is None:
                raise MjcfError("Only box, sphere, capsule, cylinder, and mesh geoms can be imported")
            size = _size(i)
            mass = float(model.body_mass[body]) * _volume(shape, size) / total if body else None
            geom = {"shape": shape, "size": size, "position": np.round(model.geom_pos[i], 6).tolist(),
                "euler_xyz": _euler(model.geom_quat[i]), "color": np.clip(model.geom_rgba[i], 0, 1).round(4).tolist(),
                "friction": float(np.clip(model.geom_friction[i][0], .05, 2)),
                "collidable": bool(model.geom_contype[i] or model.geom_conaffinity[i])}
            if mass is not None and mass > 0:
                if mass > 5:
                    warnings.append(f"Clamped a {mass:.1f} kg geom to the 5 kg per-geom limit.")
                geom["mass"] = float(np.clip(mass, .001, 5))
            if shape == "mesh":
                geom.update(_mesh_geom(i))
            if name_of(mujoco.mjtObj.mjOBJ_GEOM, i):
                geom["name"] = _name(name_of(mujoco.mjtObj.mjOBJ_GEOM, i), "", "geom", used)
            result.append(geom)
        return result

    def _mesh_geom(i):
        """Mesh vertices moved into the body frame, then stored normalized; the geom sits at their box center."""
        from studio.assets import normalize_mesh
        mid = model.geom_dataid[i]
        local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[i])
        vertices = (local @ rotation.reshape(3, 3).T + model.geom_pos[i]).astype(float)
        try:
            normalized = normalize_mesh(vertices.round(7).tolist(), faces.tolist())
        except ValueError as exc:
            raise MjcfError(f"Mesh {name_of(mujoco.mjtObj.mjOBJ_MESH, int(mid)) or mid}: {exc}") from None
        half = [max(v / 2, .002) for v in normalized["size"]]
        if any(v > .4 for v in half):
            raise MjcfError("A mesh exceeds the 0.4 m half-size limit; scale the object down before importing")
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        return {"asset_id": store(normalized), "size": [round(v, 6) for v in half],
            "position": np.round(center, 6).tolist(), "euler_xyz": [0, 0, 0]}

    def _size(i):
        size = model.geom_size[i]
        shape = SHAPES.get(int(model.geom_type[i]), "box")
        if shape == "mesh":
            half = model.geom_aabb[i][3:]
            return [round(float(max(v, .002)), 6) for v in half]
        values = size[:3] if shape == "box" else [size[0]] * 3 if shape == "sphere" else [size[0], size[0], size[1]]
        if any(v > .4 for v in values):
            raise MjcfError("A geom exceeds the 0.4 m half-size limit; scale the object down before importing")
        return [round(float(max(v, .002)), 6) for v in values]

    for body in range(1, model.nbody):
        joints = []
        for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
            kind = JOINTS[int(model.jnt_type[j])]
            dof, address = model.jnt_dofadr[j], model.jnt_qposadr[j]
            joint = {"name": _name(name_of(mujoco.mjtObj.mjOBJ_JOINT, j), prefix, f"joint{j}", used_joints), "type": kind}
            joint_names[j] = joint["name"]
            if kind in {"hinge", "slide"}:
                axis = model.jnt_axis[j]
                joint.update(axis=np.round(axis / np.linalg.norm(axis), 6).tolist(), position=np.round(model.jnt_pos[j], 6).tolist(),
                    stiffness=float(np.clip(model.jnt_stiffness[j], 0, 100)), spring_ref=float(model.qpos_spring[address]))
                if model.jnt_limited[j]:
                    joint["range"] = np.round(model.jnt_range[j], 6).tolist()
                if abs(model.qpos0[address]) > 1e-9:
                    joint["initial"] = float(model.qpos0[address])
            elif kind == "ball":
                joint["position"] = np.round(model.jnt_pos[j], 6).tolist()
            if kind != "free":
                joint.update(damping=float(np.clip(model.dof_damping[dof], 0, 50)),
                    friction_loss=float(np.clip(model.dof_frictionloss[dof], 0, 10)), armature=float(np.clip(model.dof_armature[dof], 0, 1)))
            joints.append(joint)
        geoms = geoms_for(body)
        if not geoms:
            # Pure frame bodies still need one geom in the studio graph; it is tiny and contact-free.
            geoms = [{"shape": "sphere", "size": [.002] * 3, "color": [0, 0, 0, 0], "mass": .001, "collidable": False}]
        sites = []
        for s in range(model.nsite):
            if model.site_bodyid[s] == body:
                site_names[s] = _name(name_of(mujoco.mjtObj.mjOBJ_SITE, s), prefix, f"site{s}", used_sites)
                sites.append({"name": site_names[s], "position": np.round(model.site_pos[s], 6).tolist(),
                    "size": float(np.clip(model.site_size[s][0], .002, .03))})
        nodes[body] = {"name": _name(name_of(mujoco.mjtObj.mjOBJ_BODY, body), prefix, f"body{body}", used_bodies),
            "position": np.round(model.body_pos[body], 6).tolist(), "euler_xyz": _euler(model.body_quat[body]),
            "joints": joints, "geoms": geoms, "sites": sites, "children": []}
    roots = []
    for body in range(1, model.nbody):
        parent = int(model.body_parentid[body])
        (roots if parent == 0 else nodes[parent]["children"]).append(nodes[body])
    static = geoms_for(0)
    world_sites = [s for s in range(model.nsite) if model.site_bodyid[s] == 0]
    if static or world_sites:
        roots.insert(0, {"name": _name("static", prefix, "static", used_bodies), "position": [0, 0, 0], "euler_xyz": [0, 0, 0],
            "joints": [], "geoms": static or [{"shape": "sphere", "size": [.002] * 3, "color": [0, 0, 0, 0], "collidable": False}],
            "sites": [{"name": _name(name_of(mujoco.mjtObj.mjOBJ_SITE, s), prefix, f"site{s}", used_sites),
                "position": np.round(model.site_pos[s], 6).tolist()} for s in world_sites], "children": []})
    if not roots:
        raise MjcfError("The MJCF has no bodies or geoms to import")
    body_names = {b: node["name"] for b, node in nodes.items()}
    if static or world_sites:
        body_names[0] = roots[0]["name"]
    equalities = []
    for e in range(model.neq):
        kind = int(model.eq_type[e])
        if not model.eq_active0[e]:
            warnings.append("Skipped an equality that starts inactive.")
            continue
        data, one, two = model.eq_data[e], int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        common = {"solref": np.round(model.eq_solref[e], 6).tolist(), "solimp": np.clip(model.eq_solimp[e], 0, 1).round(6).tolist()}
        if int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
            # Site pairs become body constraints anchored at site1; MuJoCo derives the rest from the start pose.
            if kind not in {int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)}:
                raise MjcfError("Only connect and weld equalities can reference sites")
            body1, body2 = int(model.site_bodyid[one]), int(model.site_bodyid[two])
            if body1 not in body_names or (body2 and body2 not in body_names):
                raise MjcfError("Site-based equalities must join sites on imported bodies")
            equalities.append({"type": "connect" if kind == int(mujoco.mjtEq.mjEQ_CONNECT) else "weld",
                "body1": body_names[body1], "body2": body_names.get(body2) if body2 else None,
                "anchor": np.round(model.site_pos[one], 6).tolist(), **common})
            continue
        if kind == int(mujoco.mjtEq.mjEQ_CONNECT):
            equalities.append({"type": "connect", "body1": body_names[one], "body2": body_names.get(two) if two else None,
                "anchor": np.round(data[0:3], 6).tolist(), **common})
        elif kind == int(mujoco.mjtEq.mjEQ_WELD):
            equalities.append({"type": "weld", "body1": body_names[one], "body2": body_names.get(two) if two else None,
                "anchor": np.round(data[0:3], 6).tolist(), "relpose": np.round(data[3:10], 6).tolist(), **common})
        elif kind == int(mujoco.mjtEq.mjEQ_JOINT):
            equalities.append({"type": "joint", "joint1": joint_names[one], "joint2": joint_names.get(two) if two >= 0 else None,
                "polycoef": np.round(data[0:5], 6).tolist(), **common})
        else:
            raise MjcfError("Only connect, weld, and joint equalities can be imported")
    excludes = []
    for signature in model.exclude_signature[:model.nexclude]:
        pair = [body_names.get(int(signature) >> 16), body_names.get(int(signature) & 0xFFFF)]
        if None in pair:
            warnings.append("Skipped a contact exclusion with the world body.")
        else:
            excludes.append(pair)
    # Place every root: rotate about the table's Z axis, then translate.
    turn = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    for root in roots:
        quat, combined = np.empty(4), np.empty(4)
        mujoco.mju_euler2Quat(quat, np.asarray(root["euler_xyz"], dtype=float), "xyz")
        mujoco.mju_mulQuat(combined, turn, quat)
        root["position"] = np.round(rotation @ np.asarray(root["position"]) + np.asarray(position), 6).tolist()
        root["euler_xyz"] = _euler(combined)
    summary = {"bodies": model.nbody - 1 + (1 if static or world_sites else 0), "joints": model.njnt,
        "geoms": model.ngeom, "sites": model.nsite}
    tendons, tendon_names = [], {}
    for t in range(model.ntendon):
        start, count = model.tendon_adr[t], model.tendon_num[t]
        kinds = {int(model.wrap_type[start + k]) for k in range(count)}
        tendon = {"name": _name(name_of(mujoco.mjtObj.mjOBJ_TENDON, t), prefix, f"tendon{t}", used_drives),
            "stiffness": float(np.clip(model.tendon_stiffness[t], 0, 1000)), "damping": float(np.clip(model.tendon_damping[t], 0, 100)),
            "width": float(np.clip(model.tendon_width[t], .0005, .02)), "color": np.clip(model.tendon_rgba[t], 0, 1).round(4).tolist()}
        if kinds == {int(mujoco.mjtWrap.mjWRAP_JOINT)}:
            ids = [int(model.wrap_objid[start + k]) for k in range(count)]
            if any(i not in joint_names for i in ids):
                raise MjcfError("A fixed tendon references a joint that cannot be imported")
            tendon.update(type="fixed", joints=[{"joint": joint_names[i], "coef": float(model.wrap_prm[start + k])} for k, i in enumerate(ids)])
        elif kinds == {int(mujoco.mjtWrap.mjWRAP_SITE)}:
            ids = [int(model.wrap_objid[start + k]) for k in range(count)]
            if any(i not in site_names for i in ids):
                raise MjcfError("Spatial tendons must run between sites on imported bodies")
            tendon.update(type="spatial", sites=[site_names[i] for i in ids])
        else:
            raise MjcfError("Only fixed tendons and spatial tendons through sites can be imported")
        spring = model.tendon_lengthspring[t]
        if spring[0] >= 0 and np.isclose(spring[0], spring[1]):
            tendon["rest_length"] = float(spring[0])
        if model.tendon_limited[t]:
            tendon["range"] = np.round(model.tendon_range[t], 6).tolist()
        tendon_names[t] = tendon["name"]
        tendons.append(tendon)
    actuators = []
    for a in range(model.nu):
        kind, target = int(model.actuator_trntype[a]), int(model.actuator_trnid[a][0])
        if int(model.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE):
            raise MjcfError("Actuators with activation dynamics (filters, integrators, muscles) are not imported")
        actuator = {"name": _name(name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, a), prefix, f"actuator{a}", used_drives)}
        if kind == int(mujoco.mjtTrn.mjTRN_JOINT) and target in joint_names:
            actuator["joint"] = joint_names[target]
        elif kind == int(mujoco.mjtTrn.mjTRN_TENDON) and target in tendon_names:
            actuator["tendon"] = tendon_names[target]
        else:
            raise MjcfError("Actuators may drive an imported hinge/slide joint or tendon only")
        gain, bias, gear = model.actuator_gainprm[a], model.actuator_biasprm[a], float(model.actuator_gear[a][0])
        if int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE):
            actuator.update(type="motor", gear=float(np.clip(gear * gain[0], -100, 100)))
        elif not np.isclose(gear, 1):
            raise MjcfError("Position and velocity actuators with a gear other than 1 are not imported")
        elif abs(bias[1]) > 0:
            actuator.update(type="position", kp=float(np.clip(gain[0], 0, 10000)), kv=float(np.clip(-bias[2], 0, 1000)))
        else:
            actuator.update(type="velocity", kv=float(np.clip(gain[0], 0, 1000)))
        if model.actuator_ctrllimited[a]:
            low, high = (float(v) for v in model.actuator_ctrlrange[a])
        else:
            low, high = -1., 1.
            warnings.append(f"Actuator {actuator['name']} had no control range; using -1..1.")
        actuator["ctrl_range"] = [low, high]
        if model.actuator_forcelimited[a]:
            actuator["force_range"] = np.round(model.actuator_forcerange[a], 6).tolist()
        # Hold the current setting: a position actuator keeps its joint where it starts.
        hold = float(model.qpos0[model.jnt_qposadr[target]]) if actuator["type"] == "position" and "joint" in actuator else 0.
        actuator["schedule"] = [[0, float(np.clip(hold, low, high))]]
        actuators.append(actuator)
    summary["equalities"], summary["contact_excludes"] = len(equalities), len(excludes)
    summary["tendons"], summary["actuators"] = len(tendons), len(actuators)
    return {"bodies": roots, "equalities": equalities, "contact_excludes": excludes, "tendons": tendons, "actuators": actuators,
        "warnings": list(dict.fromkeys(warnings)), "summary": summary}
