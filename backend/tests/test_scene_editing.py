import copy
import json
import threading
import time
import zipfile

import mujoco
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from studio import authoring_api, blender_adapter, scene_editing, storage
from studio.authoring import AuthorError, SceneProgram, compile_program
from studio.blender_schema import BlenderProgram
from studio.main import app
from studio.schemas import SceneObject
from studio.task_api import get_task


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)


@pytest.fixture
def client():
    with TestClient(app) as session:
        yield session


def finished(client, id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = client.get("/api/scenes/" + id).json()
        if result["status"] not in {"queued", "running"} and id not in authoring_api.jobs:
            return result
        time.sleep(.01)
    pytest.fail("Scene edit worker did not finish")


def make_scene(program=None, task=None, **overrides):
    program = program or SceneProgram(name="Editable factory", description="Transfer the box into its tray.")
    spec, artifact, verification = compile_program(program, task)
    return storage.save("scene", {"name": program.name, "prompt": "Build a factory", "source": "template", "model": None,
        "status": "ready", "image_ids": [], "parent_id": None, "root_id": None, "builder": "mujoco", "provider": "codex",
        "task_id": task["id"] if task else None, "task": artifact["task"], "embodiment": "cartesian-gripper-v1", "measurements": [],
        "asset_ids": [], "assets": [], "progress": [], "spec": spec, "verification": verification,
        "artifact_id": storage.write_artifact(artifact), **overrides})


def test_direct_edit_is_immutable_records_new_motion_and_preserves_task(client):
    parent = make_scene(task=get_task("block-to-tray"))
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "cube", "position": [-.30, -.10, .026]})
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"
    child = finished(client, response.json()["id"])
    assert child["status"] == "ready", child
    assert child["source"] == "manual" and child["provider"] == "codex" and child["model"] is None
    assert child["parent_id"] == child["root_id"] == parent["id"]
    assert child["task"] == parent["task"]
    assert child["artifact_id"] != parent["artifact_id"]
    assert child["spec"]["scene"]["cube_x"] == -.30
    assert child["verification"]["status"] == "pass"
    assert storage.get("scene", parent["id"]) == parent
    artifact = storage.read_artifact(child["artifact_id"])
    assert artifact["task"] == parent["task"]
    assert abs(artifact["frames"][0]["cube"][0] + .30) < .01
    assert child["prompt"] == "Moved Task box."


@pytest.mark.parametrize("edit", [
    {"object_id": "../../secrets", "position": [0, 0, .1]},
    {"object_id": "cube", "position": [0, 0, .4]},
    {"object_id": "cube", "size": [.03, .03, .03]},
    {"object_id": "cube", "position": [4, 0, .026]},
    {"object_id": "tray", "position": [.12, .05, .025]},
    {"object_id": "cube", "size": [-.1, .1, .1]},
    {"object_id": "cube"},
])
def test_invalid_edits_do_not_publish_children_or_change_parent(client, edit):
    parent = make_scene(task=get_task("block-to-tray"))
    response = client.post(f"/api/scenes/{parent['id']}/edit", json=edit)
    assert response.status_code == 422, response.text
    assert storage.all_records("scene") == [parent]


def test_objects_map_to_actual_mujoco_geometry(client):
    program = authoring_api.conveyor_program()
    program.scene.objects = [SceneObject(id="blue", position=[-.25, .20, .026])]
    parent = make_scene(program)
    result = client.get(f"/api/scenes/{parent['id']}/objects").json()
    objects = {o["id"]: o for o in result["objects"]}
    assert {"cube", "tray", "conveyor", "prop:control_cabinet", "object:blue"} == set(objects)
    assert objects["cube"]["position_axes"] == [0, 1]
    assert not objects["tray"]["editable"]
    model = mujoco.MjModel.from_xml_string(storage.read_artifact(parent["artifact_id"])["xml"])
    for obj in objects.values():
        for name in obj["geom_names"]:
            assert model.geom(name).id >= 0


def test_conveyor_resize_keeps_bottom_on_table(client):
    parent = make_scene(authoring_api.conveyor_program())
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "conveyor", "size": [.19, .08, .03]})
    child = finished(client, response.json()["id"])
    assert child["status"] == "ready", child
    belt = child["spec"]["scene"]["conveyor"]
    assert belt["height"] == .06 and belt["width"] == .16
    obj = next(o for o in scene_editing.scene_objects(child)["objects"] if o["id"] == "conveyor")
    assert obj["position"][2] == obj["size"][2] == .03


@pytest.fixture
def fake_blender(monkeypatch):
    monkeypatch.setattr(blender_adapter, "status", lambda: {"configured": True, "version": "test"})
    def worker(mode, folder, event):
        if mode == "preview":
            (folder / "preview.png").write_bytes(b"test-png")
            return
        design = BlenderProgram.model_validate_json((folder / "design.json").read_text())
        meshes = []
        for assembly in design.assemblies:
            center, size = scene_editing._assembly_bounds(assembly)
            meshes.append({"name": assembly.name, "color": assembly.color, "position": center, "size": size,
                "collisions": [{"position": p.position, "size": p.size} for p in assembly.parts if p.collision]})
            (folder / (assembly.name + ".obj")).write_text("v -1 -1 -1\nv 1 -1 -1\nv -1 1 -1\nv -1 -1 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
        (folder / "scene.blend").write_bytes(b"test-blend")
        (folder / "manifest.json").write_text(json.dumps({"format": "studio-blender-v1", "blender_version": "test",
            "collision_method": "separate part AABBs", "meshes": meshes}))
    monkeypatch.setattr(blender_adapter, "run_worker", worker)


def blender_parent():
    program = BlenderProgram.model_validate({"name": "Blender machine", "description": "Editable metal assembly.", "assemblies": [
        {"name": "BlueSteel", "parts": [{"position": [-.42, .20, .08], "size": [.03, .03, .08]},
            {"position": [-.34, .20, .08], "size": [.03, .03, .08]}]}]})
    spec, artifact, verification = blender_adapter.compile_blender(program, get_task("block-to-tray"))
    return storage.save("scene", {"name": program.name, "prompt": "Make blue machinery", "status": "ready", "image_ids": [],
        "source": "astra", "provider": "codex", "model": "gpt-6-astra", "parent_id": None, "root_id": None,
        "builder": "blender", "blender": artifact["blender"], "spec": spec, "task": artifact["task"], "task_id": "block-to-tray",
        "embodiment": "cartesian-gripper-v1", "measurements": [{"label": "Housing", "value": 20, "unit": "cm"}],
        "asset_ids": artifact["blender"]["asset_ids"], "progress": [], "verification": verification, "artifact_id": storage.write_artifact(artifact)})


def test_blender_edit_updates_source_and_regenerates_matching_meshes_and_colliders(client, fake_blender):
    parent = blender_parent()
    source = copy.deepcopy(parent["blender"]["program"])
    selected = next(o for o in scene_editing.scene_objects(parent)["objects"] if o["id"] == "assembly:BlueSteel")
    assert selected["geom_names"] == ["prop_bv_BlueSteel", "prop_bc_BlueSteel_0", "prop_bc_BlueSteel_1"]
    position = [selected["position"][0], selected["position"][1] + .05, selected["position"][2]]
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": selected["id"], "position": position, "color": [.2, .4, .6, 1]})
    assert response.status_code == 202, response.text
    child = finished(client, response.json()["id"])
    assert child["status"] == "ready", child
    assert child["builder"] == "blender" and child["task"] == parent["task"]
    assert child["measurements"] == parent["measurements"]
    assert child["asset_ids"] != parent["asset_ids"]
    parts = child["blender"]["program"]["assemblies"][0]["parts"]
    assert [p["position"][1] for p in parts] == pytest.approx([.25, .25])
    assert parent["blender"]["program"] == source
    assert storage.get("scene", parent["id"]) == parent
    props = child["spec"]["scene"]["props"]
    assert len(props) == 3 and len({p["name"] for p in props}) == 3
    assert all(p["position"][1] == pytest.approx(.25) for p in props)
    assert not child["blender"]["program"]["scene"]["props"]
    model = mujoco.MjModel.from_xml_string(storage.read_artifact(child["artifact_id"])["xml"])
    for name in selected["geom_names"]:
        assert model.geom(name).id >= 0
    folder = blender_adapter.build_folder(child["blender"]["build_id"])
    with zipfile.ZipFile(folder / "bundle.zip") as bundle:
        assert json.loads(bundle.read("design.json")) == child["blender"]["program"]
        assert json.loads(bundle.read("simulation.json"))["task"] == parent["task"]


def test_components_retain_open_geometry_and_offer_budget_constraints(client, fake_blender):
    parent = blender_parent()
    response = client.post(f"/api/scenes/{parent['id']}/components", json={"component_id": "bin"})
    child = finished(client, response.json()["id"])
    assert child["status"] == "ready", child
    assembly = child["blender"]["program"]["assemblies"][-1]
    assert assembly["name"] == "bin" and len(assembly["parts"]) == 5
    assert child["blender"]["collision_shapes"] == 7
    program = scene_editing.source_program(child).model_dump()
    program["assemblies"] += [{**copy.deepcopy(program["assemblies"][0]), "name": name} for name in ["SpareA", "SpareB"]]
    full = {**child, "blender": {"program": program}}
    choices = scene_editing.component_catalog(full)["components"]
    assert all(not c["available"] for c in choices)
    assert all("assemblies" in c["constraint"] or "four" in c["constraint"] for c in choices)
    before = storage.all_records("scene")
    bad = client.post(f"/api/scenes/{child['id']}/components", json={"component_id": "unknown"})
    assert bad.status_code == 422 and storage.all_records("scene") == before


def test_component_insert_in_plain_mujoco_keeps_individual_colliders(client):
    parent = make_scene()
    response = client.post(f"/api/scenes/{parent['id']}/components", json={"component_id": "guard"})
    child = finished(client, response.json()["id"])
    assert child["status"] == "ready", child
    assert len(child["spec"]["scene"]["props"]) == 3
    assert all(p["collidable"] for p in child["spec"]["scene"]["props"])
    assert child["task"] == parent["task"]
    catalog = client.get("/api/scene-components", params={"scene_id": child["id"]}).json()
    assert len(catalog["components"]) == 4


def test_failed_edit_keeps_parent_and_has_no_artifact(client, monkeypatch):
    parent = make_scene()
    monkeypatch.setattr(authoring_api, "compile_program", lambda *args: (_ for _ in ()).throw(AuthorError("Scene does not compile.")))
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "cube", "position": [-.30, -.10, .026]})
    child = finished(client, response.json()["id"])
    assert child["status"] == "failed" and "artifact_id" not in child
    assert "does not compile" in child["error"]
    assert storage.get("scene", parent["id"]) == parent


def test_cancelled_edit_never_publishes_ready_artifact(client, monkeypatch):
    parent = make_scene()
    entered, release = threading.Event(), threading.Event()
    original = authoring_api.compile_program
    def blocked(*args):
        entered.set()
        assert release.wait(4)
        return original(*args)
    monkeypatch.setattr(authoring_api, "compile_program", blocked)
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "cube", "position": [-.30, -.10, .026]})
    id = response.json()["id"]
    try:
        assert entered.wait(3)
        assert client.post(f"/api/scenes/{id}/cancel").json()["status"] == "cancelled"
    finally:
        release.set()
    child = finished(client, id)
    assert child["status"] == "cancelled" and "artifact_id" not in child
    assert storage.get("scene", parent["id"]) == parent


def test_conversation_is_ordered_bounded_and_selection_context_is_concrete(client):
    parent = make_scene()
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "cube", "position": [-.30, -.10, .026]})
    child = finished(client, response.json()["id"])
    draft = client.post("/api/scenes", json={"name": "Factory", "prompt": "Make this orange", "parent_id": child["id"], "selected_object_id": "cube"}).json()
    lineage = client.get(f"/api/scenes/{draft['id']}/conversation").json()
    assert [s["id"] for s in lineage["scenes"]] == [parent["id"], child["id"], draft["id"]]
    assert not lineage["truncated"]
    assert scene_editing.scene_conversation(draft, limit=1)["truncated"]
    context = scene_editing.author_context(draft)
    assert '"selected_object": {"id": "cube"' in context and "Moved Task box" in context
    assert draft["prompt"] == "Make this orange"
    assert draft["task"] == parent["task"]
    bad = client.post("/api/scenes", json={"name": "Factory", "prompt": "Move it", "parent_id": child["id"], "selected_object_id": "missing"})
    assert bad.status_code == 422


def test_blender_rotation_requires_uniform_scale(fake_blender):
    parent = blender_parent()
    parent["blender"]["program"]["assemblies"][0]["parts"][0]["euler_xyz"] = [0, 0, .3]
    selected = next(o for o in scene_editing.scene_objects(parent)["objects"] if o["kind"] == "assembly")
    with pytest.raises(ValueError, match="same factor"):
        scene_editing.edited_program(parent, scene_editing.SceneEdit(object_id=selected["id"], size=[selected["size"][0] * 1.1, *selected["size"][1:]]))
    with pytest.raises(ValidationError):
        scene_editing.SceneEdit(object_id="cube", position=[float("nan"), 0, 0])


def test_cancellation_after_blender_compile_removes_only_new_candidate_assets(client, fake_blender, monkeypatch):
    parent = blender_parent()
    assets_before = storage.all_records("asset")
    original = blender_adapter.compile_blender
    candidate = {}
    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        candidate.update(result[1]["blender"])
        kwargs["event"].set()
        return result
    monkeypatch.setattr(blender_adapter, "compile_blender", cancelled)
    response = client.post(f"/api/scenes/{parent['id']}/edit", json={"object_id": "assembly:BlueSteel", "position": [-.38, .25, .08]})
    id = response.json()["id"]
    child = finished(client, id)
    assert child["status"] == "cancelled" and "artifact_id" not in child
    assert storage.all_records("asset") == assets_before
    assert not blender_adapter.build_folder(candidate["build_id"]).exists()
    assert blender_adapter.build_folder(parent["blender"]["build_id"]).exists()


def test_author_receives_selected_object_history_without_changing_saved_prompt(client, monkeypatch):
    parent = make_scene()
    received = []
    monkeypatch.setattr(authoring_api, "provider_status", lambda _: {"configured": True, "name": "Test connection"})
    def author(draft, parent_program, progress, event, **options):
        received.append(draft)
        program = SceneProgram.model_validate(parent_program)
        spec, artifact, verification = options["compiler"](program, draft["task"])
        return {"spec": spec, "artifact": artifact, "verification": verification, "summary": "Kept the selected box."}
    monkeypatch.setattr(authoring_api, "author_scene", author)
    draft = client.post("/api/scenes", json={"name": "Factory", "prompt": "Keep this here", "provider": "openai", "parent_id": parent["id"], "selected_object_id": "cube"}).json()
    assert client.post(f"/api/scenes/{draft['id']}/generate").status_code == 202
    child = finished(client, draft["id"])
    assert child["status"] == "ready", child
    assert '"selected_object": {"id": "cube"' in received[0]["prompt"]
    assert '"request": "Build a factory"' in received[0]["prompt"]
    assert child["prompt"] == "Keep this here"


@pytest.mark.parametrize('provider', ['codex', 'openai'])
@pytest.mark.parametrize('builder', ['mujoco', 'blender'])
def test_conversational_revisions_reject_changed_task_before_compile_and_can_repair(client, fake_blender, monkeypatch, provider, builder):
    from studio import codex_author
    parent = blender_parent() if builder == 'blender' else make_scene(task=get_task('block-to-tray'))
    monkeypatch.setattr(authoring_api, 'provider_status', lambda _: {'configured': True, 'name': 'Test author'})
    reports = []
    def author(draft, parent_program, progress, event, **options):
        assert draft['task_snapshot_locked']
        model = BlenderProgram if builder == 'blender' else SceneProgram
        original = model.model_validate(parent_program)
        invalid = original.model_dump()
        invalid['scene']['tray_x'] -= .15
        with pytest.raises(ValidationError, match='saved task') as error:
            options['compiler'](model.model_validate(invalid), task=draft['task'], embodiment=draft['embodiment'])
        reports.append(error.value.errors(include_context=False, include_input=False))
        spec, artifact, verification = options['compiler'](original, task=draft['task'], embodiment=draft['embodiment'])
        return {'spec': spec, 'artifact': artifact, 'verification': verification, 'summary': 'Repaired layout without moving the destination.'}
    monkeypatch.setattr(authoring_api, 'author_scene', author)
    monkeypatch.setattr(codex_author, 'author_scene_codex', author)
    draft = client.post('/api/scenes', json={'name': 'Revised factory', 'prompt': 'Make this blue', 'parent_id': parent['id'],
        'task_id': parent['task_id'], 'builder': builder, 'provider': provider, 'selected_object_id': 'cube'}).json()
    assert draft['task'] == parent['task'] and draft['task_snapshot_locked']
    assert client.post(f"/api/scenes/{draft['id']}/generate").status_code == 202
    child = finished(client, draft['id'])
    assert child['status'] == 'ready', child
    assert reports and 'destination bounds' in reports[0][0]['msg']
    assert child['task'] == parent['task']
    assert storage.get('scene', parent['id']) == parent


def test_new_human_selected_task_is_allowed_to_bind_a_new_scene(client, monkeypatch):
    parent = make_scene(task=get_task('block-to-tray'))
    monkeypatch.setattr(authoring_api, 'provider_status', lambda _: {'configured': True, 'name': 'Test author'})
    def author(draft, parent_program, progress, event, **options):
        assert not draft['task_snapshot_locked']
        assert 'compiler' not in options
        program = SceneProgram.model_validate(parent_program)
        program.scene.tray_x = .12
        program.waypoints = None
        spec, artifact, verification = compile_program(program, draft['task'])
        return {'spec': spec, 'artifact': artifact, 'verification': verification, 'summary': 'Generated the newly selected lift task.'}
    monkeypatch.setattr(authoring_api, 'author_scene', author)
    draft = client.post('/api/scenes', json={'name': 'Lift task', 'prompt': 'Lift and hold the block', 'parent_id': parent['id'],
        'task_id': 'lift-and-hold', 'provider': 'openai'}).json()
    assert draft['task_id'] == 'lift-and-hold' and not draft['task_snapshot_locked']
    assert client.post(f"/api/scenes/{draft['id']}/generate").status_code == 202
    child = finished(client, draft['id'])
    assert child['status'] == 'ready', child
    assert child['task']['id'] == 'lift-and-hold'
    assert storage.get('scene', parent['id']) == parent


def test_raw_program_preview_also_preserves_saved_task_destination(client):
    parent = make_scene(task=get_task('block-to-tray'))
    program = copy.deepcopy(parent['spec'])
    program['scene']['tray_x'] = .15
    response = client.post(f"/api/scenes/{parent['id']}/preview", json=program)
    assert response.status_code == 422 and 'saved task' in response.json()['detail']
    assert storage.all_records('scene') == [parent]
