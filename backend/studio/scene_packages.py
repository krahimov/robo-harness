"""Portable, declarative scene packages. Archive code and MJCF are never executed on import."""
import base64
import binascii
from copy import deepcopy
import io
import json
import re
import shutil
import stat
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from studio import storage as db
from studio.assets import mesh_data, parse_obj, public_asset
from studio.authoring import SceneProgram, compile_program
from studio.robots import robot
from studio.images import MAX_IMAGE_BYTES, _normalize, image_path
from studio.schemas import StrictModel, task_for_scene
from studio.task_api import TaskCreate

router = APIRouter(prefix="/api")
FORMAT = "robot-harness-scene-package"
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 96 * 1024 * 1024
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_FILES = 128
MAX_BASE64 = 4 * ((MAX_ZIP_BYTES + 2) // 3)
ID = r"[a-f0-9]{32}"
ARCHIVE_ATTACHMENTS = {"source/scene.blend", "source/preview.png", "source/ROBOT_ASSETS_LICENSE.txt"}


class PackageAsset(StrictModel):
    id: str = Field(pattern="^" + ID + "$")
    name: str = Field(min_length=1, max_length=100)
    file: str = Field(pattern=r"^assets/[a-f0-9]{32}\.obj$")
    input_size: list[float] | None = Field(default=None, min_length=3, max_length=3)

    @model_validator(mode="after")
    def valid_size(self):
        if self.input_size and not all(0 < n <= 20000 for n in self.input_size):
            raise ValueError("Invalid mesh dimensions")
        return self


class PackageReference(StrictModel):
    id: str = Field(pattern="^" + ID + "$")
    name: str = Field(min_length=1, max_length=200)
    file: str = Field(pattern=r"^references/[a-f0-9]{32}\.jpg$")


class PackageMeasurement(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    value: float = Field(gt=0, le=100000)
    unit: Literal["m", "cm", "mm"]


class PackageManifest(StrictModel):
    format: Literal[FORMAT]
    version: Literal[1]
    name: str = Field(min_length=1, max_length=100)
    prompt: str = Field(default="Imported scene package", max_length=8000)
    source_scene_id: str = Field(pattern="^" + ID + "$")
    builder: Literal["mujoco", "blender"] = "mujoco"
    program_file: Literal["program.json"] = "program.json"
    task_file: Literal["task.json"] = "task.json"
    blender_program_file: Literal["source/blender-program.json"] | None = None
    assets: list[PackageAsset] = Field(default_factory=list, max_length=8)
    references: list[PackageReference] = Field(default_factory=list, max_length=4)
    measurements: list[PackageMeasurement] = Field(default_factory=list, max_length=8)
    attachments: list[str] = Field(default_factory=list, max_length=3)
    robot_files: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def unique_files(self):
        for records in (self.assets, self.references):
            if len({item.id for item in records}) != len(records) or len({item.file for item in records}) != len(records):
                raise ValueError("Duplicate package resource")
        if len(set(self.attachments)) != len(self.attachments) or not set(self.attachments) <= ARCHIVE_ATTACHMENTS:
            raise ValueError("Unsupported source attachment")
        if len(set(self.robot_files)) != len(self.robot_files):
            raise ValueError("Duplicate robot resource")
        for filename in self.robot_files:
            if filename not in {"robot/panda.xml", "robot/LICENSE", "robot/SOURCE.txt"} and not re.fullmatch(r"robot/assets/[A-Za-z0-9_-]+\.(?:obj|stl|png)", filename):
                raise ValueError("Unsupported robot resource")
        return self


class PackageUpload(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    data_base64: str = Field(min_length=4, max_length=MAX_BASE64)


def _json(data):
    return json.dumps(data, indent=2, allow_nan=False).encode("utf-8")


def _read_json(raw):
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Package JSON files must be at most 2 MiB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON property")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def _obj(asset):
    # Stored mesh geometry is normalized. Scene props supply its world position and scale.
    return ("\n".join([*("v " + " ".join(map(str, v)) for v in asset["vertices"]),
                        *("f " + " ".join(str(i + 1) for i in face) for face in asset["faces"])]) + "\n").encode()


def _asset_ids(program):
    from studio.scene_graph import scene_asset_ids
    return set(scene_asset_ids(program["scene"]))


def _portable_xml(xml, embodiment):
    """Copy only the installed, supplied robot's referenced files; never arbitrary host paths."""
    root = ET.fromstring(xml)
    if root.find("include") is not None:
        raise ValueError("Scene XML includes are unsupported")
    files = {}
    if embodiment == "franka-panda-v1":
        from studio.adapters.franka import ASSET_COMMIT, asset_folder
        folder = asset_folder().resolve()
        assets = (folder / "assets").resolve()
        compiler = root.find("compiler")
        if compiler is None or Path(compiler.get("meshdir", "")).resolve() != assets:
            raise ValueError("Unexpected robot asset directory")
        for element in root.iter():
            if "file" not in element.attrib:
                continue
            filename = element.get("file")
            if element.tag != "mesh" or not re.fullmatch(r"[A-Za-z0-9_-]+\.(?:obj|stl)", filename):
                raise ValueError("Unsupported scene file reference")
            path = assets / filename
            if not path.resolve().is_relative_to(assets) or not path.is_file():
                raise ValueError("Robot mesh is unavailable")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Robot mesh exceeds package limit")
            files["robot/assets/" + filename] = path.read_bytes()
        compiler.set("meshdir", "robot/assets")
        for name in ("panda.xml", "LICENSE"):
            path = folder / name
            if not path.resolve().is_relative_to(folder) or not path.is_file():
                raise ValueError("Robot source and license are required")
            files["robot/" + name] = path.read_bytes()
        files["robot/SOURCE.txt"] = ("MuJoCo Menagerie Franka Panda\nhttps://github.com/google-deepmind/mujoco_menagerie\nCommit: " + ASSET_COMMIT + "\n").encode()
    elif embodiment in _arm_configs():
        files.update(_portable_arm(root, _arm_configs()[embodiment]))
    elif any("file" in element.attrib for element in root.iter()):
        raise ValueError("Unexpected external scene asset")
    return ET.tostring(root, encoding="utf-8"), files


def _arm_configs():
    from studio.adapters.arms import CONFIGS
    return CONFIGS


def _portable_arm(root, config):
    """Menagerie arm parts: copy referenced meshes from the pinned part folders only."""
    from studio.adapters.arms import MENAGERIE_COMMIT
    folders = {folder.name: folder for folder in config.asset_folders()}
    files = {}
    for element in root.iter():
        if "file" not in element.attrib:
            continue
        path = Path(element.get("file")).resolve()
        folder = next((f for f in folders.values() if path.is_relative_to(f / "assets")), None)
        if element.tag != "mesh" or folder is None or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:obj|stl)", path.name, re.I):
            raise ValueError("Unsupported scene file reference")
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Robot mesh is unavailable or exceeds the package limit")
        relative = f"robot/{folder.name}/assets/{path.relative_to(folder / 'assets').as_posix()}"
        files[relative] = path.read_bytes()
        element.set("file", relative)
    for part in config.parts:
        folder = folders[part.folder]
        for name in (part.file, "LICENSE"):
            if not (folder / name).is_file():
                raise ValueError("Robot source and license are required")
            files[f"robot/{part.folder}/{name}"] = (folder / name).read_bytes()
    files["robot/SOURCE.txt"] = (f"MuJoCo Menagerie {config.name}\nhttps://github.com/google-deepmind/mujoco_menagerie\nCommit: {MENAGERIE_COMMIT}\n"
        "Parts: " + ", ".join(part.folder for part in config.parts) + "\n").encode()
    return files


def package_bytes(scene, include_references=True):
    if scene.get("status") != "ready" or not scene.get("artifact_id"):
        raise HTTPException(409, "Generate and preview a scene before exporting its package.")
    program = SceneProgram.model_validate(scene["spec"]).model_dump(exclude_none=True)
    artifact = db.read_artifact(scene["artifact_id"])
    files = {"program.json": _json(program), "task.json": _json(artifact["task"])}
    source = scene.get("package_source") or {}
    blender_program = (scene.get("blender") or {}).get("program") or source.get("blender_program")
    resources = _asset_ids(program)
    if blender_program:
        from studio.blender_schema import BlenderProgram
        blender_program = BlenderProgram.model_validate(blender_program).model_dump(exclude_none=True)
        files["source/blender-program.json"] = _json(blender_program)
        resources |= _asset_ids(blender_program)
    assets = []
    for id in sorted(resources):
        mesh = mesh_data(id)
        filename = f"assets/{id}.obj"
        files[filename] = _obj(mesh)
        assets.append({"id": id, "name": mesh["name"], "file": filename, "input_size": mesh.get("size")})
    references = []
    if include_references:
        for id in scene.get("image_ids", []):
            photo = db.get("reference", id)
            filename = f"references/{id}.jpg"
            files[filename] = image_path(id).read_bytes()
            references.append({"id": id, "name": photo["name"], "file": filename})
    attachments = []
    blender = scene.get("blender")
    attachment_folder = None
    if blender:
        from studio.blender_adapter import build_folder
        attachment_folder = build_folder(blender["build_id"])
    elif source.get("archive_id"):
        attachment_folder = _archive_folder(source["archive_id"]) / "source"
    if attachment_folder:
        for filename in ("scene.blend", "preview.png", "ROBOT_ASSETS_LICENSE.txt"):
            path = attachment_folder / filename
            if path.is_file() and path.resolve().is_relative_to(attachment_folder.resolve()):
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("Blender source exceeds package limit")
                name = "source/" + filename
                files[name] = path.read_bytes()
                attachments.append(name)
    xml, robot_files = _portable_xml(artifact["xml"], program["scene"]["embodiment"])
    files["scene.xml"] = xml
    files.update(robot_files)
    manifest = PackageManifest(format=FORMAT, version=1, name=scene["name"], prompt=scene.get("prompt", ""),
        source_scene_id=scene["id"], builder=scene.get("builder", "mujoco"),
        blender_program_file="source/blender-program.json" if blender_program else None,
        assets=assets, references=references, measurements=scene.get("measurements", []),
        attachments=attachments, robot_files=sorted(robot_files))
    files["package.json"] = _json(manifest.model_dump(exclude_none=True))
    files["README.txt"] = (
        "Robot Harness Studio scene package v1\n\n"
        "Edit program.json to change the runnable scene, then import the ZIP into the studio.\n"
        "Mesh OBJ files contain normalized geometry; program props set placement and size.\n"
        "scene.xml references bundled robot assets and can be loaded by MuJoCo from this directory.\n"
        "Import recompiles program.json with the studio's installed robot runtime and measures a new rollout.\n"
        "Imported XML, robot source files and Blender files are archival; the importer never executes them.\n"
        "source/blender-program.json preserves the original geometry design, when present.\n"
        "Editing scene.blend does not update the simulation. Edit program.json or re-author its geometry.\n"
        "Reference photos are included unless export used include_references=false.\n"
        "This package contains no provider credentials or generation logs. Keep bundled asset licenses.\n"
    ).encode()
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_EXPANDED_BYTES:
        raise ValueError("Scene exceeds the portable package limit")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Scene file exceeds the portable package limit")
            archive.writestr(name, data)
    result = output.getvalue()
    if len(result) > MAX_ZIP_BYTES:
        raise ValueError("Compressed scene package exceeds 64 MiB")
    return result


@router.get("/scenes/{id}/package")
def export_package(id: str, include_references: bool = True):
    data = package_bytes(db.get("scene", id), include_references)
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="scene-{id}.zip"', "X-Content-Type-Options": "nosniff"})


def _archive_folder(id):
    if not isinstance(id, str) or not re.fullmatch(ID, id):
        raise KeyError("Package archive not found")
    path = db.ROOT / "packages" / id
    if not path.resolve().is_relative_to(db.ROOT.resolve()):
        raise ValueError("Package storage directory is invalid")
    return path


@router.get("/scene-packages/{id}/source/{filename}")
def source_attachment(id: str, filename: str):
    if "source/" + filename not in ARCHIVE_ATTACHMENTS:
        raise KeyError("Package source attachment not found")
    folder = _archive_folder(id)
    path = folder / "source" / filename
    if not (folder / "complete").is_file() or not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
        raise KeyError("Package source attachment not found")
    return FileResponse(path, filename=filename, headers={"X-Content-Type-Options": "nosniff"})


def _unpack(raw):
    if not raw or len(raw) > MAX_ZIP_BYTES:
        raise ValueError("Scene packages must be at most 64 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(item.file_size for item in entries) > MAX_EXPANDED_BYTES:
                raise ValueError("Package exceeds the expanded size or file count limit")
            names, directories = set(), set()
            for item in entries:
                directory = item.is_dir()
                filename = item.filename[:-1] if directory else item.filename
                path = PurePosixPath(filename)
                mode = item.external_attr >> 16
                if (not filename or len(filename) > 160 or "\\" in filename or "\x00" in filename or path.is_absolute()
                    or any(p in {"", ".", ".."} for p in filename.split("/"))
                    or item.filename != item.orig_filename or item.filename in names or item.filename in directories
                    or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFDIR if directory else stat.S_IFREG}
                    or item.flag_bits & 1 or item.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or item.file_size > MAX_FILE_BYTES or (directory and item.file_size)):
                    raise ValueError("Package contains an unsupported archive entry")
                (directories if directory else names).add(item.filename)
            if "package.json" not in names:
                raise ValueError("Provide a versioned Robot Harness Studio scene package")
            manifest = PackageManifest.model_validate(_read_json(archive.read("package.json")))
            expected = {"package.json", "program.json", "task.json", "scene.xml", "README.txt", *manifest.robot_files,
                        *manifest.attachments, *(a.file for a in manifest.assets), *(r.file for r in manifest.references)}
            if manifest.blender_program_file:
                expected.add(manifest.blender_program_file)
            if names != expected or any(not any(name.startswith(directory) for name in expected) for directory in directories):
                raise ValueError("Package contents do not match its manifest")
            return manifest, {item.filename: archive.read(item) for item in entries if not item.is_dir()}
    except (zipfile.BadZipFile, UnicodeError, KeyError, RuntimeError, NotImplementedError, EOFError) as exc:
        raise ValueError("Invalid scene package archive") from None


def _remap(program, mapping):
    program = deepcopy(program)
    from studio.scene_graph import mesh_geoms
    for item in [*program["scene"].get("props", []), *mesh_geoms(program["scene"])]:
        if item.get("asset_id"):
            if item["asset_id"] not in mapping:
                raise ValueError("A scene mesh is missing from the package")
            item["asset_id"] = mapping[item["asset_id"]]
    return program


def _task_snapshot(raw, scene):
    if not isinstance(raw, dict):
        raise ValueError("Package task must be an object")
    if raw.get("schema_version") != 2:
        # Legacy requirements refer to the original tray. Validate those semantics
        # before rebinding the same task to a deliberately edited layout.
        parameters = raw.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Invalid legacy task parameters")
        previous = scene.model_validate({**scene.model_dump(), "tray_x": parameters.get("tray_x", scene.tray_x),
                                         "tray_y": parameters.get("tray_y", scene.tray_y)})
        canonical = task_for_scene(previous)
        if raw.get("requirements") != canonical["requirements"]:
            raise ValueError("Unsupported legacy task requirements")
        return task_for_scene(scene), None
    version = raw.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= 1000000:
        raise ValueError("Invalid source task version")
    spec = TaskCreate.model_validate({key: raw[key] for key in ("name", "description", "requirements", "operations") if key in raw})
    id = uuid.uuid4().hex
    snapshot = {**spec.model_dump(), "id": id, "root_id": id, "version": 1, "schema_version": 2,
                "owner": "human", "source": "imported", "embodiment": scene.embodiment,
                "success": "All required outcomes pass. Unknown required outcomes need review.",
                "suggested_scene": scene.model_dump(), "imported_from": {"id": str(raw.get("id", ""))[:100], "version": version}}
    return snapshot, id


def import_package(raw, name=None):
    manifest, files = _unpack(raw)
    program = SceneProgram.model_validate(_read_json(files[manifest.program_file]))
    blender_program = None
    if manifest.blender_program_file:
        from studio.blender_schema import BlenderProgram
        blender_program = BlenderProgram.model_validate(_read_json(files[manifest.blender_program_file])).model_dump(exclude_none=True)
    if not robot(program.scene.embodiment).available():
        raise HTTPException(409, f"Install the supplied {robot(program.scene.embodiment).name} robot runtime before importing this scene. The package also contains standalone MuJoCo XML and robot assets.")
    # Parse every resource before creating files or records. Imported Python, XML and blend files are never run.
    prepared_assets = []
    for asset in manifest.assets:
        parsed = parse_obj(files[asset.file].decode("utf-8"), max_vertices=1200, max_faces=2400, max_bytes=512 * 1024)
        if asset.input_size:
            parsed["size"] = asset.input_size
        prepared_assets.append((asset.id, {"id": uuid.uuid4().hex, "created_at": db.now(), "name": asset.name, "kind": "mesh", "source": "imported", "status": "staging", **parsed}))
    mapping = {old: asset["id"] for old, asset in prepared_assets}
    if set(mapping) != (_asset_ids(program.model_dump()) | (_asset_ids(blender_program) if blender_program else set())):
        raise ValueError("Package assets must match the scene and its source design")
    program = SceneProgram.model_validate(_remap(program.model_dump(exclude_none=True), mapping))
    if name:
        program = program.model_copy(update={"name": name.strip() or program.name})
    if blender_program:
        blender_program = _remap(blender_program, mapping)
    task, task_id = _task_snapshot(_read_json(files[manifest.task_file]), program.scene)
    photos = []
    for reference in manifest.references:
        content = files[reference.file]
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("Package photos must be at most 8 MiB")
        normalized, width, height = _normalize("image/jpeg", content)
        id = uuid.uuid4().hex
        photos.append(({"id": id, "name": reference.name, "url": f"/api/references/{id}/image", "width": width,
                        "height": height, "mime_type": "image/jpeg", "size_bytes": len(normalized)}, normalized))
    created_assets, paths = [], []
    archive_id = uuid.uuid4().hex
    archive_folder = _archive_folder(archive_id)
    try:
        for _, asset in prepared_assets:
            db.save("asset", asset)
            created_assets.append(asset["id"])
        resolved, artifact, verification = compile_program(program, task)
        source = {"format": FORMAT, "version": 1, "source_scene_id": manifest.source_scene_id,
                  "builder": manifest.builder, "blender_program": blender_program,
                  "archive_id": archive_id if manifest.attachments else None,
                  "attachments": {Path(path).name: f"/api/scene-packages/{archive_id}/{path}" for path in manifest.attachments},
                  "note": "Imported the editable simulation program and recorded a new rollout. Original Blender files remain source attachments; editing them does not change this simulation."}
        if manifest.attachments:
            (archive_folder / "source").mkdir(parents=True)
            for filename in manifest.attachments:
                (archive_folder / filename).write_bytes(files[filename])
            (archive_folder / "complete").write_text("ready")
        if photos:
            from studio.images import _folder
            folder = _folder()
            folder.mkdir(parents=True, exist_ok=True)
            for photo, content in photos:
                path = folder / (photo["id"] + ".jpg")
                paths.append(path)
                with path.open("xb") as output:
                    output.write(content)
        artifact["package_source"] = source
        artifact_id = db.write_artifact(artifact)
        paths.append(db.ROOT / "artifacts" / (artifact_id + ".json"))
        scene_id = uuid.uuid4().hex
        scene = {"id": scene_id, "created_at": db.now(), "name": program.name, "prompt": manifest.prompt,
                 "image_ids": [p["id"] for p, _ in photos], "provider": "codex", "model": None, "source": "imported",
                 "builder": "mujoco", "parent_id": None, "root_id": None, "status": "ready", "task_id": task_id,
                 "task": artifact["task"], "measurements": [measurement.model_dump() for measurement in manifest.measurements], "asset_ids": created_assets,
                 "assets": [public_asset({**asset, "status": "ready"}) for _, asset in prepared_assets],
                 "embodiment": program.scene.embodiment, "package_source": source,
                 "progress": [{"message": "Imported package, recompiled geometry and recorded a fresh MuJoCo rollout."}],
                 "spec": resolved, "artifact_id": artifact_id, "verification": verification,
                 "summary": "Imported the editable scene and recorded a new simulation. Review the measured checks before saving a demonstration. Blender source attachments are preserved separately."}
        records = [("scene", scene), *[("reference", photo) for photo, _ in photos]]
        if task_id:
            records.append(("task", artifact["task"]))
        with db.connection() as connection:
            for kind, body in records:
                body.setdefault("created_at", db.now())
                connection.execute("INSERT INTO records VALUES (?, ?, ?)", (body["id"], kind, json.dumps(body, allow_nan=False)))
            for _, asset in prepared_assets:
                asset["status"] = "ready"
                connection.execute("UPDATE records SET body=? WHERE id=? AND kind='asset'", (json.dumps(asset, allow_nan=False), asset["id"]))
        return scene
    except BaseException:
        for id in created_assets:
            db.delete("asset", id)
        for path in paths:
            path.unlink(missing_ok=True)
        shutil.rmtree(archive_folder, ignore_errors=True)
        raise


@router.post("/scenes/import", status_code=201)
async def upload_package(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BASE64 + 2048:
            raise HTTPException(413, "Scene packages must be at most 64 MiB")
    try:
        upload = PackageUpload.model_validate_json(bytes(body))
        raw = base64.b64decode(upload.data_base64, validate=True)
        return await run_in_threadpool(import_package, raw, upload.name)
    except HTTPException:
        raise
    except (ValueError, ValidationError, binascii.Error, OverflowError, RecursionError, OSError):
        # Do not echo uploaded document contents, local paths, or executable text in errors.
        raise HTTPException(422, "Invalid scene package. Use a version 1 studio ZIP containing its declared program, bounded geometry, task and photo files. Import requires the supplied robot runtime; archive code and Blender files are never executed.") from None
