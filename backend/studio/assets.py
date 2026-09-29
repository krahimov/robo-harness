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
    if len(vertices) < 4 or len(faces) < 4:
        raise ValueError("A mesh needs at least four vertices and four triangles")
    low = [min(v[i] for v in vertices) for i in range(3)]
    high = [max(v[i] for v in vertices) for i in range(3)]
    size = [high[i] - low[i] for i in range(3)]
    if min(size) < 1e-6:
        raise ValueError("A mesh needs nonzero thickness in all three dimensions")
    normalized = [[round((v[i] - (low[i] + high[i]) / 2) / (size[i] / 2), 7) for i in range(3)] for v in vertices]
    return {"vertices": normalized, "faces": faces, "size": size, "vertex_count": len(vertices), "face_count": len(faces)}


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
