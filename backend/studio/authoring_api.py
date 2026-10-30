"""Persistent scene drafts, photo references, generation jobs, and demo promotion."""
import json
import threading
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import Field, ValidationError, model_validator

from studio import storage as db
from studio.authoring import MODEL, AuthorError, AuthorCancelled, SceneProgram, author_scene, compile_program
from studio.images import MAX_IMAGE_BYTES, MAX_IMAGES, save_image, image_path
from studio.schemas import StrictModel, Promotion, Scene
from studio.author_providers import Provider, connection_status, default_provider, model_for, model_name, provider_status, refresh_codex_status
from studio.task_api import get_task
from studio.robots import Embodiment
from studio.scene_graph import scene_asset_ids
from studio.assets import mesh_data, public_asset
from studio import scene_editing

router = APIRouter(prefix="/api")
pool: ThreadPoolExecutor | None = None
jobs: dict[str, threading.Event] = {}
job_lock = threading.RLock()


def startup():
    global pool
    with job_lock:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-author")
        jobs.clear()
        for scene in db.all_records("scene"):
            if scene["status"] in {"queued", "running"}:
                db.update("scene", scene["id"], status="interrupted", error="The server restarted. Your brief and references are saved.")


def shutdown():
    global pool
    with job_lock:
        for event in jobs.values():
            event.set()
        current, pool = pool, None
    if current:
        current.shutdown(wait=False, cancel_futures=True)


class ReferenceUpload(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    data_url: str = Field(max_length=4 * ((MAX_IMAGE_BYTES + 2) // 3) + 100)


class Measurement(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    value: float = Field(gt=0, le=100000)
    unit: str = Field(pattern=r"^(m|cm|mm)$")


class SceneDraft(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    prompt: str = Field(default="", max_length=8000)
    image_ids: list[str] = Field(default_factory=list, max_length=MAX_IMAGES)
    parent_id: str | None = None
    provider: Provider | None = None
    builder: Literal['mujoco','blender'] = 'mujoco'
    embodiment: Embodiment = "cartesian-gripper-v1"
    task_id: str | None = Field(default=None, max_length=100)
    measurements: list[Measurement] = Field(default_factory=list, max_length=8)
    asset_ids: list[str] = Field(default_factory=list, max_length=4)
    selected_object_id: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def meaningful(self):
        if not self.name.strip() or (len(self.prompt.strip()) < 3 and not self.image_ids):
            raise ValueError("Enter a scene name and add a photo or describe the operation")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("Use each photo reference only once")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("Use each mesh asset only once")
        return self


@router.get("/author/status")
def author_status(refresh: bool = False):
    if refresh:
        refresh_codex_status()
    from studio.blender_adapter import status
    return {"model": MODEL, **connection_status(), "max_images": MAX_IMAGES, "max_image_bytes": MAX_IMAGE_BYTES, "blender":status()}


@router.get("/references")
def references():
    return db.all_records("reference")


@router.post("/references", status_code=201)
async def upload_reference(request: Request):
    # Bound the raw request before JSON/Pydantic processing, including chunked bodies.
    limit = 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 2048
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Each photo must be at most 8 MiB.")
    try:
        upload = ReferenceUpload.model_validate_json(bytes(body))
    except ValidationError:
        raise HTTPException(422, "Provide an image name and a JPEG, PNG, or WebP data URL, at most 8 MiB.") from None
    return save_image(upload.name, upload.data_url)


@router.get("/references/{id}/image")
def reference_image(id: str):
    return FileResponse(image_path(id), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400", "X-Content-Type-Options": "nosniff"})


@router.get("/scenes")
def scenes():
    return db.all_records("scene")


@router.post("/scenes", status_code=201)
def create_scene(spec: SceneDraft):
    for image_id in spec.image_ids:
        db.get("reference", image_id)
    parent = db.get("scene", spec.parent_id) if spec.parent_id else None
    task = get_task(spec.task_id) if spec.task_id else None
    locked_task = bool(parent and parent.get("task") and (spec.task_id is None or spec.task_id == parent.get("task_id")))
    if locked_task:
        task = parent["task"]
    if spec.selected_object_id:
        if parent is None:
            raise HTTPException(422, "Select an existing scene before referring to an object.")
        if spec.selected_object_id not in {o["id"] for o in scene_editing.scene_objects(parent)["objects"]}:
            raise HTTPException(422, "The selected object is not in the parent revision.")
    assets = [public_asset(mesh_data(id)) for id in spec.asset_ids]
    provider = spec.provider or default_provider()
    return db.save("scene", {**spec.model_dump(), "name": spec.name.strip(), "prompt": spec.prompt.strip(),
        "task": task, "task_snapshot_locked": locked_task, "assets": assets,
        "task_id": parent.get("task_id") if locked_task else spec.task_id,
        "provider": provider,
        "root_id": (parent.get("root_id") or parent["id"]) if parent else None,
        "parent_program": ((parent.get('blender') or {}).get('program') if spec.builder=='blender' else None) or parent.get("spec") or parent.get("parent_program") if parent else None,
        "status": "draft", "model": model_for(provider), "progress": [], "source": "astra"})


def conveyor_program():
    return SceneProgram(name="Conveyor pick and place", description="One box travels along a short conveyor. A Cartesian gripper transfers it into the tray.",
        assumptions=["Built-in template; no photo was interpreted and no AI model was called.",
            "Machinery uses simple collision geometry at tabletop scale.",
            "One box per recording, approximate conveyor drive, and an idealized gripper attachment."],
        scene=Scene(conveyor={"x": -.22, "y": -.08, "length": .38, "width": .14, "height": .04, "speed": .025, "direction": "x"},
            props=[{"name": "control_cabinet", "shape": "box", "position": [-.44, .24, .12], "size": [.075, .07, .12], "color": [.57, .64, .60, 1], "collidable": True}]))


@router.post("/scenes/template", status_code=201)
def template(task_id: str | None = None, embodiment: Embodiment = "cartesian-gripper-v1", builder: Literal['mujoco','blender']='mujoco'):
    task = get_task(task_id) if task_id else None
    program = SceneProgram(name=task["name"], description=task["description"], assumptions=["Built-in teacher motion for the selected human task. No model was called."], scene=Scene.model_validate(task.get("suggested_scene", {}))) if task else conveyor_program()
    if builder=='blender':
        from studio.blender_adapter import compile_blender
        from studio.blender_example import conveyor_design
        program=conveyor_design()
        spec,artifact,verification=compile_blender(program,task,embodiment)
    else:
        spec, artifact, verification = compile_program(program, task, embodiment)
    blender=artifact.get('blender')
    return db.save("scene", {"name": spec["name"], "prompt": "Built-in example: " + spec["name"], "image_ids": [],
        "parent_id": None, "root_id": None, "status": "ready", "source": "template", "model": None, "task_id": task_id, "task": artifact["task"], "measurements": [], "asset_ids": blender['asset_ids'] if blender else [], "embodiment": embodiment, 'builder':builder,'blender':blender,
        "progress": [{"message": "Ran the built-in teacher for " + spec["name"] + " in MuJoCo."}], "spec": spec,
        "artifact_id": db.write_artifact(artifact), "verification": verification,
        "summary": spec["name"] + ". Recorded in MuJoCo without an AI call. Review its motion and assumptions."})


@router.get("/scenes/{id}")
def scene_detail(id: str):
    return db.get("scene", id)


@router.get("/scenes/{id}/objects")
def scene_objects(id: str):
    return scene_editing.scene_objects(db.get("scene", id))


@router.get("/scenes/{id}/conversation")
def scene_conversation(id: str):
    return scene_editing.scene_conversation(db.get("scene", id))


@router.get("/scene-components")
def scene_components(scene_id: str | None = None):
    return scene_editing.component_catalog(db.get("scene", scene_id) if scene_id else None)


def run_scene_edit(id, program, event):
    artifact, artifact_id, published = None, None, False
    def progress(message):
        with job_lock:
            if event.is_set():
                raise AuthorCancelled("Scene edit cancelled.")
            current = db.get("scene", id)
            db.update("scene", id, progress=[*current["progress"], {"message": message}])
    try:
        with job_lock:
            if event.is_set():
                return
            child = db.update("scene", id, status="running")
        progress("Compiling your edit and recording the teacher in MuJoCo.")
        scene_editing.task_is_preserved(child, program)
        if child.get("builder") == "blender":
            from studio.blender_adapter import compile_blender
            spec, artifact, verification = compile_blender(program, child.get("task"), child.get("embodiment"), event=event, progress=progress)
        else:
            spec, artifact, verification = compile_program(program, child.get("task"), child.get("embodiment"))
        if child.get("task") and artifact["task"] != child["task"]:
            raise ValueError("The compiler changed the saved task. The parent revision is unchanged.")
        asset_ids = scene_asset_ids(spec["scene"])
        with job_lock:
            if event.is_set():
                raise AuthorCancelled("Scene edit cancelled.")
            artifact_id = db.write_artifact(artifact)
            db.update("scene", id, spec=spec, task=artifact["task"], verification=verification,
                artifact_id=artifact_id, blender=artifact.get("blender"),
                asset_ids=asset_ids, assets=[public_asset(mesh_data(key)) for key in asset_ids],
                status="ready", error=None, summary=child["prompt"] + " Recorded a new teacher demonstration. Task requirements are unchanged; review the state checks.")
            published = True
    except AuthorCancelled:
        db.update("scene", id, status="cancelled", error="Scene edit cancelled. The parent revision is unchanged.")
    except Exception as exc:
        safe_error = scene_editing.validation_message(exc) if isinstance(exc, (ValueError, AuthorError)) else "The scene edit could not compile. The parent revision is unchanged."
        db.update("scene", id, status="cancelled" if event.is_set() else "failed", error=safe_error)
    finally:
        try:
            if not published:
                scene_editing.discard_edit_candidate(artifact)
                if artifact_id:
                    (db.ROOT / "artifacts" / (artifact_id + ".json")).unlink(missing_ok=True)
        finally:
            with job_lock:
                if jobs.get(id) is event:
                    jobs.pop(id, None)


def queue_scene_edit(parent, program, summary, selected_object_id=None):
    """Publish a validated child under the same queue lock as its worker."""
    with job_lock:
        if parent["status"] != "ready":
            raise HTTPException(409, "Finish generating this scene before making a direct edit.")
        if pool is None:
            raise HTTPException(503, "The scene author is not running. Restart the backend.")
        if len(jobs) >= 4:
            raise HTTPException(409, "The scene author has four pending jobs. Wait for one to finish.")
        if parent.get("builder") == "blender":
            from studio.blender_adapter import status
            ready = status()
            if not ready["configured"]:
                raise HTTPException(409, ready["message"])
        inherited = {key: parent.get(key) for key in ("name", "provider", "builder", "task_id", "task", "embodiment")}
        child = db.save("scene", {**inherited, "builder": parent.get("builder", "mujoco"),
            "image_ids": parent.get("image_ids", []), "measurements": parent.get("measurements", []),
            "asset_ids": parent.get("asset_ids", []), "assets": parent.get("assets", []),
            "parent_id": parent["id"], "root_id": parent.get("root_id") or parent["id"],
            "parent_program": program.model_dump(exclude_none=True), "selected_object_id": selected_object_id,
            "prompt": summary, "summary": summary, "status": "queued", "source": "manual", "model": None,
            "progress": [{"message": "Queued your scene edit."}]})
        event = threading.Event()
        jobs[child["id"]] = event
        try:
            pool.submit(run_scene_edit, child["id"], program, event)
        except RuntimeError:
            jobs.pop(child["id"], None)
            db.delete("scene", child["id"])
            raise HTTPException(503, "The scene worker is stopping. Retry after the backend restarts.") from None
        return child


@router.post("/scenes/{id}/edit", status_code=202)
def edit_scene(id: str, edit: scene_editing.SceneEdit):
    parent = db.get("scene", id)
    program, summary = scene_editing.edited_program(parent, edit)
    return queue_scene_edit(parent, program, summary, edit.object_id)


@router.post("/mjcf/preview")
def preview_mjcf(request: scene_editing.MjcfInsert):
    """Convert without saving, so the UI can show bodies, joints, and warnings first."""
    from studio.mjcf_import import convert
    from studio.schemas import Scene
    converted = convert(request.xml, request.name, request.position, request.yaw, request.decoded_files())
    Scene(bodies=converted["bodies"], equalities=converted["equalities"], contact_excludes=converted["contact_excludes"],
        tendons=converted["tendons"], actuators=converted["actuators"])
    return converted


@router.post("/scenes/{id}/mjcf", status_code=202)
def insert_scene_mjcf(id: str, request: scene_editing.MjcfInsert):
    parent = db.get("scene", id)
    program, summary, warnings = scene_editing.insert_mjcf(parent, request)
    return {**queue_scene_edit(parent, program, summary), "import_warnings": warnings}


@router.post("/scenes/{id}/components", status_code=202)
def insert_scene_component(id: str, request: scene_editing.ComponentInsert):
    parent = db.get("scene", id)
    program, summary = scene_editing.insert_component(parent, request)
    return queue_scene_edit(parent, program, summary)


def run_author(id, event):
    def progress(message):
        with job_lock:
            if event.is_set():
                raise AuthorCancelled("Scene generation cancelled.")
            scene = db.get("scene", id)
            db.update("scene", id, progress=[*scene["progress"], {"message": message}])
    try:
        with job_lock:
            if event.is_set():
                return
            draft = db.update("scene", id, status="running")
        context = scene_editing.author_context(draft)
        if context:
            # Keep the user's saved message untouched; only the provider input
            # receives the bounded lineage and concrete selected-object data.
            draft = {**draft, "prompt": draft["prompt"] + context}
        options={}
        if draft.get('builder')=='blender':
            from studio.blender_adapter import compile_blender
            from studio.blender_schema import BlenderProgram, BLENDER_RULES
            options={'program_model':BlenderProgram,'extra_instructions':BLENDER_RULES,
                'compiler':lambda program,task=None,embodiment=None:compile_blender(program,task,embodiment,event=event,progress=progress)}
        if draft.get("task_snapshot_locked"):
            compiler = options.get("compiler") or compile_program
            def compile_revision(program, task=None, embodiment=None):
                selected = program.model_copy(update={"scene": Scene.model_validate({**program.scene.model_dump(), "embodiment": embodiment or program.scene.embodiment})})
                try:
                    scene_editing.task_is_preserved(draft, selected)
                except ValueError as exc:
                    # Both author transports send structured validation errors
                    # back to Astra, allowing a corrected candidate next turn.
                    raise ValidationError.from_exception_data("Saved task", [{"type": "value_error", "loc": ("scene",),
                        "input": None, "ctx": {"error": ValueError(str(exc))}}]) from None
                result = compiler(selected, task=draft["task"], embodiment=embodiment)
                if result[1]["task"] != draft["task"]:
                    scene_editing.discard_edit_candidate(result[1])
                    raise AuthorError("The compiler changed the saved task. Revise the scene without changing its destination or requirements.")
                return result
            options["compiler"] = compile_revision
        if (draft.get("provider") or "openai") in {"codex", "claude"}:
            from studio.codex_author import author_scene_codex
            result = author_scene_codex(draft, draft.get("parent_program"), progress, event, provider=draft["provider"], **options)
        else:
            result = author_scene(draft, draft.get("parent_program"), progress, event,**options)
        with job_lock:
            if event.is_set():
                return
            if draft.get("task_snapshot_locked") and result["artifact"]["task"] != draft["task"]:
                scene_editing.discard_edit_candidate(result["artifact"])
                raise AuthorError("The generated scene changed the saved task. Its parent revision is unchanged.")
            result["task"] = result["artifact"]["task"]
            if result['artifact'].get('blender'):
                result['blender']=result['artifact']['blender']
                asset_ids=scene_asset_ids(result['spec']['scene'])
                result.update(asset_ids=asset_ids,assets=[public_asset(mesh_data(key)) for key in asset_ids])
            artifact_id = db.write_artifact(result.pop("artifact"))
            db.update("scene", id, **result, artifact_id=artifact_id, status="ready", error=None)
    except AuthorCancelled:
        db.update("scene", id, status="cancelled", error="Generation cancelled. Your brief and references are saved.")
    except Exception as exc:
        # Never store/log provider exception bodies, credentials, or reference image bytes.
        error = str(exc) if isinstance(exc, AuthorError) else "Scene generation failed. Your brief and references are saved; retry or revise the request."
        db.update("scene", id, status="cancelled" if event.is_set() else "failed", error=error)
    finally:
        with job_lock:
            if jobs.get(id) is event:
                jobs.pop(id, None)


@router.post("/scenes/{id}/generate", status_code=202)
def generate_scene(id: str):
    with job_lock:
        draft = db.get("scene", id)
        connection = provider_status(draft.get("provider") or "openai")
        if not connection["configured"]:
            raise HTTPException(409, connection["message"])
        if draft.get('builder')=='blender':
            from studio.blender_adapter import status
            if not status()['configured']:
                raise HTTPException(409,status()['message'])
        if len(draft["prompt"].strip()) < 3:
            raise HTTPException(422, "Describe the operation before generating the scene.")
        if id in jobs:
            raise HTTPException(409, "The previous job is still stopping. Wait a moment before retrying.")
        if draft["status"] in {"ready", "running", "queued"}:
            raise HTTPException(409, "Create a revision of this scene before generating again.")
        if pool is None:
            raise HTTPException(503, "The scene author is not running. Restart the backend.")
        if len(jobs) >= 4:
            raise HTTPException(409, "The scene author has four pending jobs. Wait for one to finish.")
        event = threading.Event()
        jobs[id] = event
        provider = draft.get("provider") or "openai"
        scene = db.update("scene", id, status="queued", model=model_for(provider), source="astra", error=None,
            progress=[{"message": f"Queued for {model_name(provider)} through {connection['name']}."}])
        pool.submit(run_author, id, event)
        return scene


@router.post("/scenes/{id}/cancel")
def cancel_scene(id: str):
    with job_lock:
        scene = db.get("scene", id)
        if id in jobs and scene["status"] in {"queued", "running"}:
            jobs[id].set()
            return db.update("scene", id, status="cancelled", error="Generation cancelled. Your brief and references are saved.")
        return scene


@router.post("/scenes/{id}/preview", status_code=201)
def preview_revision(id: str, program: SceneProgram):
    parent = db.get("scene", id)
    if parent["status"] in {"running", "queued"}:
        raise HTTPException(409, "Wait for generation to finish before editing this scene.")
    scene_editing.task_is_preserved(parent, program)
    spec, artifact, verification = compile_program(program, parent.get("task"))
    if parent.get("task") and artifact["task"] != parent["task"]:
        raise HTTPException(422, "The edited program changed the saved task. Its parent revision is unchanged.")
    asset_ids = list(dict.fromkeys([*scene_asset_ids(program.scene.model_dump()), *parent.get("asset_ids", [])]))
    return db.save("scene", {"name": spec["name"], "prompt": "Manual scene revision", "image_ids": parent["image_ids"],
        "parent_id": id, "root_id": parent.get("root_id") or id, "status": "ready", "source": "manual", "model": None, 'builder':'mujoco',
        "task_id": parent.get("task_id"), "task": artifact["task"], "measurements": parent.get("measurements", []), "asset_ids": asset_ids, "assets": [public_asset(mesh_data(id)) for id in asset_ids], "embodiment": program.scene.embodiment,
        "progress": [{"message": "Compiled and recorded your edited scene program."}], "spec": spec,
        "artifact_id": db.write_artifact(artifact), "verification": verification,
        "summary": "Manual revision recorded in MuJoCo. Inspect the state checks before saving a demonstration."})


@router.post("/scenes/{id}/record", status_code=201)
def record_scene(id: str, spec: Promotion):
    from studio.main import store_demo
    with job_lock:
        scene = db.get("scene", id)
        if scene["status"] != "ready" or not scene.get("artifact_id"):
            raise HTTPException(409, "Generate and preview the scene before recording a demonstration.")
        if scene.get("demo_id"):
            return db.get("demo", scene["demo_id"])
        artifact = db.read_artifact(scene["artifact_id"])
        demo = store_demo(spec.name, scene["spec"]["scene"], artifact,
            "scene_author" if scene["source"] == "astra" else scene["source"],
            waypoints=scene["spec"]["waypoints"], source_scene_id=id, author_model=scene.get("model"),
            author_provider=scene.get("provider"), reference_image_ids=scene["image_ids"], task_id=scene.get("task_id"), task=artifact.get("task"))
        db.update("scene", id, demo_id=demo["id"])
        return demo


@router.get("/scenes/{id}/export")
def export_scene(id: str):
    scene = db.get("scene", id)
    if not scene.get("artifact_id"):
        raise HTTPException(409, "Generate a scene before exporting it.")
    artifact = db.read_artifact(scene["artifact_id"])
    return {"format": "robot-harness-scene-v1", "scene_id": id, "source": scene["source"], "model": scene.get("model"), "provider": scene.get("provider"),
        "spec": scene["spec"], "references": [db.get("reference", key) for key in scene["image_ids"]],
        "xml": artifact["xml"], "actions": artifact["actions"], "frequency": artifact["frequency"],
        "verification": scene["verification"], "task": artifact["task"], 'builder':scene.get('builder','mujoco'),'blender':scene.get('blender'),
        "measurements": scene.get("measurements", []), "assets": [public_asset(mesh_data(id)) for id in scene.get("asset_ids", [])]}
