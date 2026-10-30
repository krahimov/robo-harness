"""Static CAD visuals for compact recorded robot traces.

The binary bundle contains no animation or new physics state. Every visual is
rigidly attached to a collision geom already present in the recorded frames.
"""
from functools import lru_cache
import gzip
import hashlib
import json
import struct

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
import mujoco
import numpy as np

from studio.adapters.franka import ASSET_COMMIT, asset_folder

router = APIRouter(prefix="/api/robot-visuals")
SOURCE = "https://github.com/google-deepmind/mujoco_menagerie/tree/" + ASSET_COMMIT + "/franka_emika_panda"


def geom_transform(model, index):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[index])
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = model.geom_pos[index]
    return transform


def visual_glb(model, license_text):
    """Export original surfaces, materials, and anchor-relative transforms."""
    recorded = [i for i in range(model.ngeom) if model.geom_group[i] != 2]
    if not recorded:
        raise ValueError("The robot has no recorded geometry anchors.")
    first = recorded[0]
    anchors = {}
    for i in recorded:
        anchors.setdefault(int(model.geom_bodyid[i]), i)
    metadata = {"embodiment": "franka-panda-v1", "robot_asset_commit": ASSET_COMMIT,
        "collision_offsets": [i - first for i in recorded], "source": SOURCE,
        "license": "Apache-2.0", "license_text": license_text,
        "description": "CAD display surfaces only. Motion comes from recorded MuJoCo geometry transforms."}
    nodes = []
    for i in range(model.ngeom):
        if model.geom_group[i] != 2:
            continue
        anchor = anchors.get(int(model.geom_bodyid[i]))
        if anchor is None:
            raise ValueError("A robot visual has no recorded anchor.")
        relative = np.linalg.inv(geom_transform(model, anchor)) @ geom_transform(model, i)
        nodes.append((i, f"panda_visual_{i}", relative, {"anchor_offset": anchor - first, "visual_geom_index": i}))
    return build_glb(model, nodes, metadata, license_text)


def build_glb(model, nodes, metadata, copyright):
    """Write one binary glTF holding each (geom, node name, transform, extras) visual mesh."""
    document = {"asset": {"version": "2.0", "generator": "Robot Harness Studio", "copyright": copyright},
        "scene": 0, "scenes": [{"nodes": [], "extras": metadata}], "nodes": [], "meshes": [],
        "materials": [], "accessors": [], "bufferViews": [], "buffers": []}
    binary = bytearray()
    materials = {}

    def accessor(values, bounds=False):
        values = np.asarray(values, dtype="<f4").reshape(-1, 3)
        start = len(binary)
        binary.extend(values.tobytes())
        view = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": start, "byteLength": values.nbytes, "target": 34962})
        index = len(document["accessors"])
        result = {"bufferView": view, "componentType": 5126, "count": len(values), "type": "VEC3"}
        if bounds:
            result.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        document["accessors"].append(result)
        return index

    total_vertices = 0
    for i, name, relative, extras in nodes:
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Unsupported native robot visual geometry.")
        mid = int(model.geom_dataid[i])
        vstart, vcount = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fstart, fcount = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        nstart, ncount = int(model.mesh_normaladr[mid]), int(model.mesh_normalnum[mid])
        faces = model.mesh_face[fstart:fstart + fcount]
        normal_faces = model.mesh_facenormal[fstart:fstart + fcount]
        positions = model.mesh_vert[vstart:vstart + vcount][faces.reshape(-1)]
        normals = model.mesh_normal[nstart:nstart + ncount][normal_faces.reshape(-1)]
        total_vertices += len(positions)
        if total_vertices > 2_000_000:
            raise ValueError("Robot display geometry exceeds the browser budget.")
        material_id = int(model.geom_matid[i])
        rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[i]
        material_key = tuple(float(value) for value in rgba)
        if material_key not in materials:
            materials[material_key] = len(document["materials"])
            document["materials"].append({"pbrMetallicRoughness": {"baseColorFactor": rgba.tolist(), "metallicFactor": .12, "roughnessFactor": .48}})
        mesh = len(document["meshes"])
        document["meshes"].append({"primitives": [{"attributes": {"POSITION": accessor(positions, True), "NORMAL": accessor(normals)},
            "material": materials[material_key]}]})
        node = len(document["nodes"])
        document["nodes"].append({"name": name, "mesh": mesh, "matrix": relative.T.reshape(-1).tolist(), "extras": extras})
        document["scenes"][0]["nodes"].append(node)
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\x00" * (-len(binary) % 4))
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, length) + struct.pack("<I4s", len(encoded), b"JSON") + encoded
        + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


@lru_cache(maxsize=2)
def visual_bundle(folder, xml_stamp, license_stamp):
    # Only the installed, fixed robot assets are read. Clients cannot supply XML
    # or file paths, and the response contains neither runtime nor user data.
    from pathlib import Path
    path = Path(folder)
    model = mujoco.MjModel.from_xml_path(str(path / "panda.xml"))
    payload = visual_glb(model, (path / "LICENSE").read_text())
    return payload, gzip.compress(payload, compresslevel=6, mtime=0), hashlib.sha256(payload).hexdigest()


@router.get("/franka-panda-v1.glb")
def panda_visuals(request: Request):
    folder = asset_folder().resolve()
    try:
        payload, compressed, digest = visual_bundle(str(folder), (folder / "panda.xml").stat().st_mtime_ns, (folder / "LICENSE").stat().st_mtime_ns)
    except (OSError, ValueError):
        raise HTTPException(503, "The optional Panda display assets are unavailable. Recorded geometry remains accessible.") from None
    headers = {"Cache-Control": "private, max-age=3600", "ETag": '"' + digest + '"', "Vary": "Accept-Encoding"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        payload = compressed
    return Response(payload, media_type="model/gltf-binary", headers=headers)


def arm_visual_glb(config):
    """Visual meshes of a registry arm, each placed relative to its link; frames carry link poses."""
    from studio.adapters.arms import MENAGERIE_COMMIT, arm_spec, robot_visual_bodies
    from studio.schemas import Scene
    model = arm_spec(Scene(embodiment=config.id, include_cube=False, include_tray=False), config).compile()
    bodies = robot_visual_bodies(model, config)
    nodes = [(i, f"{config.id}_visual_{i}", geom_transform(model, i), {"body": model.body(int(model.geom_bodyid[i])).name})
        for i in range(model.ngeom) if model.geom_group[i] == 2 and int(model.geom_bodyid[i]) in bodies]
    licenses = {part.folder: (folder / "LICENSE").read_text() for part, folder in zip(config.parts, config.asset_folders())}
    metadata = {"embodiment": config.id, "robot_asset_commit": MENAGERIE_COMMIT, "geom_prefix": config.parts[0].prefix,
        "source": "https://github.com/google-deepmind/mujoco_menagerie/tree/" + MENAGERIE_COMMIT, "license": "Apache-2.0",
        "license_text": "\n\n".join(f"{name}:\n{text}" for name, text in licenses.items()),
        "description": "CAD display surfaces only. Motion comes from recorded MuJoCo link poses."}
    return build_glb(model, nodes, metadata, metadata["license_text"])


@lru_cache(maxsize=4)
def arm_bundle(embodiment, stamp):
    from studio.adapters.arms import CONFIGS
    payload = arm_visual_glb(CONFIGS[embodiment])
    return payload, gzip.compress(payload, compresslevel=6, mtime=0), hashlib.sha256(payload).hexdigest()


@router.get("/arm/{embodiment}.glb")
def arm_visuals(embodiment: str, request: Request):
    from studio.adapters.arms import CONFIGS
    config = CONFIGS.get(embodiment)
    if config is None:
        raise HTTPException(404, "No display assets for that robot.")
    try:
        stamp = tuple((folder / part.file).stat().st_mtime_ns for part, folder in zip(config.parts, config.asset_folders()))
        payload, compressed, digest = arm_bundle(embodiment, stamp)
    except (OSError, ValueError):
        raise HTTPException(503, "The optional robot display assets are unavailable. Recorded geometry remains accessible.") from None
    headers = {"Cache-Control": "private, max-age=3600", "ETag": '"' + digest + '"', "Vary": "Accept-Encoding"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        payload = compressed
    return Response(payload, media_type="model/gltf-binary", headers=headers)
