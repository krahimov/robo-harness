"""Bounded OBJ geometry import. No file paths, external materials, or executable assets."""
import math

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, ValidationError

from studio import storage as db
from studio.schemas import StrictModel

router = APIRouter(prefix="/api")
MAX_OBJ_BYTES = 256 * 1024
MAX_VERTICES = 400
MAX_FACES = 800


class MeshUpload(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    obj: str = Field(min_length=1, max_length=MAX_OBJ_BYTES)


def parse_obj(text, *, max_vertices=MAX_VERTICES, max_faces=MAX_FACES, max_bytes=MAX_OBJ_BYTES):
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("OBJ files must be at most 256 KiB")
    vertices, faces = [], []
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        kind, values = parts[0], parts[1:]
        if kind == "v":
            if len(values) != 3 or len(vertices) >= max_vertices:
                raise ValueError("Use XYZ vertices only, at most 400")
            vertex = [float(v) for v in values]
            if not all(math.isfinite(v) and abs(v) <= 10000 for v in vertex):
                raise ValueError("Mesh coordinates must be finite and within ±10000 input units")
            vertices.append(vertex)
        elif kind == "f":
            if not 3 <= len(values) <= 8:
                raise ValueError("OBJ faces require 3 to 8 vertex indices")
            indexes = []
            for value in values:
                index = int(value.split("/", 1)[0])
                resolved = index - 1 if index > 0 else len(vertices) + index
                if index == 0 or not 0 <= resolved < len(vertices):
                    raise ValueError("OBJ face references a missing vertex")
                indexes.append(resolved)
            for i in range(1, len(indexes) - 1):
                if len({indexes[0], indexes[i], indexes[i+1]}) != 3:
                    raise ValueError("OBJ contains a degenerate face")
                faces.append([indexes[0], indexes[i], indexes[i+1]])
            if len(faces) > max_faces:
                raise ValueError("Meshes support at most 800 triangles")
        elif kind in {"o", "g", "s", "vn", "vt"}:
            continue
        else:
            raise ValueError("Upload geometry-only OBJ files without materials, textures, or external references")
    return normalize_mesh(vertices, faces, max_vertices=max_vertices, max_faces=max_faces)


def normalize_mesh(vertices, faces, *, max_vertices=MAX_VERTICES, max_faces=MAX_FACES):
    """Center each axis on the bounding box and scale it to ±1; `size` keeps the full extents."""
    if len(vertices) > max_vertices or len(faces) > max_faces:
        raise ValueError(f"Meshes support at most {max_vertices} vertices and {max_faces} triangles")
    if len(vertices) < 4 or len(faces) < 4:
        raise ValueError("A mesh needs at least four vertices and four triangles")
    low = [min(v[i] for v in vertices) for i in range(3)]
    high = [max(v[i] for v in vertices) for i in range(3)]
    size = [high[i] - low[i] for i in range(3)]
    if min(size) < 1e-6:
        raise ValueError("A mesh needs nonzero thickness in all three dimensions")
    normalized = [[round((v[i] - (low[i] + high[i]) / 2) / (size[i] / 2), 7) for i in range(3)] for v in vertices]
    return {"vertices": normalized, "faces": [[int(i) for i in face] for face in faces], "size": size,
        "vertex_count": len(vertices), "face_count": len(faces)}


def parse_stl(data: bytes):
    """Binary or ASCII STL triangles, deduplicated into indexed vertices; no file references."""
    if len(data) > MAX_OBJ_BYTES:
        raise ValueError("STL files must be at most 256 KiB")
    triangles = []
    if len(data) >= 84 and 84 + 50 * int.from_bytes(data[80:84], "little") == len(data):
        import struct
        for k in range(int.from_bytes(data[80:84], "little")):
            values = struct.unpack_from("<12f", data, 84 + 50 * k)
            triangles.append([values[3:6], values[6:9], values[9:12]])
    else:
        text = data.decode("ascii", errors="strict")
        points = [tuple(float(v) for v in line.split()[1:4]) for line in text.splitlines() if line.strip().startswith("vertex")]
        if len(points) % 3:
            raise ValueError("ASCII STL has an incomplete triangle")
        triangles = [points[i:i + 3] for i in range(0, len(points), 3)]
    index, vertices, faces = {}, [], []
    for triangle in triangles:
        face = []
        for point in triangle:
            if not all(math.isfinite(v) and abs(v) <= 10000 for v in point):
                raise ValueError("Mesh coordinates must be finite and within ±10000 input units")
            key = tuple(round(v, 7) for v in point)
            if key not in index:
                index[key] = len(vertices)
                vertices.append(list(key))
            face.append(index[key])
        if len(set(face)) == 3:
            faces.append(face)
        if len(vertices) > MAX_VERTICES or len(faces) > MAX_FACES:
            raise ValueError(f"Meshes support at most {MAX_VERTICES} vertices and {MAX_FACES} triangles; simplify the STL first")
    return vertices, faces


def public_asset(asset):
    return {key: value for key, value in asset.items() if key not in {"vertices", "faces"}}


@router.get("/assets")
def list_assets():
    return [public_asset(asset) for asset in db.all_records("asset") if asset.get("status") != "staging"]


@router.post("/assets", status_code=201)
async def upload_asset(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_OBJ_BYTES * 2 + 1024:
            raise HTTPException(413, "OBJ files must be at most 256 KiB")
    try:
        upload = MeshUpload.model_validate_json(bytes(body))
        parsed = parse_obj(upload.obj)
    except (ValidationError, ValueError, OverflowError):
        raise HTTPException(422, "Provide a geometry-only OBJ: at most 256 KiB, 400 vertices, 800 triangles, with no external files and nonzero XYZ thickness.") from None
    return public_asset(db.save("asset", {"name": upload.name.strip(), "kind": "mesh", **parsed}))


def mesh_data(id):
    if len(id) != 32 or any(c not in "0123456789abcdef" for c in id):
        raise ValueError("Invalid mesh asset ID")
    try:
        asset = db.get("asset", id)
    except KeyError:
        raise ValueError("Imported mesh asset was not found") from None
    if asset.get("kind") != "mesh":
        raise ValueError("Asset is not a mesh")
    return asset
