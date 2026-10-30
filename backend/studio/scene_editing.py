"""Immutable scene edits and reusable geometry for the conversational studio.

Edits target the author's source program, never the expanded Blender mesh and
collision output. Source validation happens before a child job is published.
"""
from copy import deepcopy
import json
import math
import shutil
from typing import Annotated

import numpy as np
from pydantic import Field, ValidationError, model_validator

from studio import storage as db
from studio.authoring import SceneProgram
from studio.schemas import Color, StrictModel


Vector = Annotated[list[float], Field(min_length=3, max_length=3)]
PositiveVector = Annotated[list[Annotated[float, Field(gt=0, le=.8)]], Field(min_length=3, max_length=3)]


class SceneEdit(StrictModel):
    object_id: str = Field(min_length=1, max_length=80)
    position: Vector | None = None
    size: PositiveVector | None = None
    color: Color | None = None

    @model_validator(mode="after")
    def has_edit(self):
        if self.position is None and self.size is None and self.color is None:
            raise ValueError("Move, resize, or recolor the selected object.")
        return self


class ComponentInsert(StrictModel):
    component_id: str = Field(min_length=1, max_length=40)
    position: Vector | None = None


def validation_message(exc):
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(v) for v in e['loc'])}: {e['msg']}" for e in exc.errors(include_input=False, include_context=False, include_url=False)[:3])
    return str(exc)


def source_program(scene):
    if scene.get("builder") == "blender":
        from studio.blender_schema import BlenderProgram
        source = (scene.get("blender") or {}).get("program") or scene.get("parent_program")
        if not source or "assemblies" not in source:
            raise ValueError("This scene has no editable Blender source. Generate a Blender revision first.")
        return BlenderProgram.model_validate(deepcopy(source))
    source = scene.get("spec") or scene.get("parent_program")
    if not source:
        raise ValueError("Generate a scene before selecting or editing its objects.")
    return SceneProgram.model_validate(deepcopy(source))


def _assembly_bounds(assembly):
    lows, highs = [], []
    for part in assembly.parts:
        x, y, z = part.euler_xyz
        rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        local = [part.size[0], part.size[0], part.size[2]] if part.shape == "cylinder" else part.size
        extent = np.abs(rz @ ry @ rx) @ np.array(local)
        lows.append(np.array(part.position) - extent)
        highs.append(np.array(part.position) + extent)
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    return ((low + high) / 2).tolist(), ((high - low) / 2).tolist()


def _object(id, label, kind, position, size, geom_names, color=None, axes=None, resize=True, constraints=None):
    return {"id": id, "label": label, "kind": kind, "position": position, "size": size,
        "geom_names": geom_names, "color": color, "editable": True,
        "position_axes": axes if axes is not None else [0, 1, 2],
        "dimensions_editable": resize, "color_editable": color is not None, "constraints": constraints or []}


def scene_objects(scene):
    program = source_program(scene)
    spec = program.scene
    z = .026
    if spec.conveyor:
        from studio.adapters.simulation import on_conveyor
        if on_conveyor(spec.conveyor, spec.cube_x, spec.cube_y):
            z += spec.conveyor.height
    objects = []
    located = {r.get("object_id") for r in (scene.get("task") or {}).get("requirements", []) if r.get("check") in {"inside_region", "inside_while_held"}}
    if spec.include_cube:
        objects.append(_object("cube", "Task box", "movable", [spec.cube_x, spec.cube_y, z], [.025] * 3, ["block"], spec.cube_color, [0, 1], False,
            ["The primary box has fixed size and rests on the table or conveyor."]))
    if spec.include_tray:
        tray = _object("tray", "Destination tray", "tray", [spec.tray_x, spec.tray_y, .025], [.125, .095, .025],
            ["tray_base", "tray_north", "tray_south", "tray_east", "tray_west"], axes=[0, 1], resize=False)
        if scene.get("task"):
            tray.update(editable=False, position_axes=[], constraints=["The saved task fixes the destination bounds. Create a new task to move this tray."])
        objects.append(tray)
    if spec.conveyor:
        belt = spec.conveyor
        half = [belt.length / 2, belt.width / 2] if belt.direction == "x" else [belt.width / 2, belt.length / 2]
        objects.append(_object("conveyor", "Conveyor belt", "conveyor", [belt.x, belt.y, belt.height / 2], [*half, belt.height / 2], ["conveyor_belt"], axes=[0, 1],
            constraints=["Size uses XYZ half-extents. Resizing height also moves the belt center to keep its bottom on the table."]))
    for item in spec.objects:
        objects.append(_object("object:" + item.id, item.id.replace("_", " "), "movable", item.position, item.size, ["object_" + item.id + "_geom"], item.color))
    for prop in spec.props:
        objects.append(_object("prop:" + prop.name, prop.name.replace("_", " "), "prop", prop.position, prop.size, ["prop_" + prop.name], prop.color))
    for body in spec.bodies:
        parts = [(item, geom, index) for item, _ in body.walk() for index, geom in enumerate(item.geoms)]
        names = [f"geom_{item.name}_{geom.name or index}" for item, geom, index in parts]
        joints = sum(len(item.joints) for item, _ in body.walk())
        own = [(np.array(g.position), np.array(g.size if g.shape == "box" else [g.size[0], g.size[0], g.size[2]])) for g in body.geoms]
        half = ((np.max([c + h for c, h in own], axis=0) - np.min([c - h for c, h in own], axis=0)) / 2).tolist()
        item = _object("body:" + body.name, body.name.replace("_", " "), "articulated" if joints else "fixed body", body.position, half, names,
            axes=[0, 1], resize=False, constraints=[f"Articulated body with {joints} joint{'' if joints == 1 else 's'}. Moving it moves every attached part; teacher skills follow its named sites.",
                "Edit its parts, joints, and dimensions in the advanced scene specification."])
        if located & {b.name for b, _ in body.walk()}:
            # Like the tray: a saved task fixes where this body must end, so moving it would fail its own check.
            item.update(editable=False, position_axes=[], constraints=["The saved task fixes this body's region. Create a new task version to move it."])
        objects.append(item)
    for cloth in spec.cloths:
        half = [(cloth.count[0] - 1) * cloth.spacing / 2, (cloth.count[1] - 1) * cloth.spacing / 2, cloth.thickness]
        item = _object("cloth:" + cloth.name, cloth.name.replace("_", " "), "cloth", cloth.position, half, ["cloth_" + cloth.name], cloth.color,
            axes=[0, 1], resize=False, constraints=[f"{cloth.count[0]}×{cloth.count[1]} vertex MuJoCo flex cloth. Robots hold it with an idealized pinch."])
        if cloth.name in located:
            item.update(editable=False, position_axes=[], constraints=["The saved task fixes this cloth's region. Create a new task version to move it."])
        objects.append(item)
    for assembly in getattr(program, "assemblies", []):
        center, half = _assembly_bounds(assembly)
        names = ["prop_bv_" + assembly.name]
        names += [f"prop_bc_{assembly.name}_{i}" for i in range(sum(part.collision for part in assembly.parts))]
        objects.append(_object("assembly:" + assembly.name, assembly.name.replace("_", " "), "assembly", center, half, names, assembly.color,
            constraints=["Moves all source parts and rebuilds the visual mesh and individual colliders.", "Assemblies with rotated parts support uniform resizing only."]))
    return {"objects": objects, "constraints": ["Positions are world metres, Z up. Sizes are half-extents.", "Every edit creates a revision and reruns the teacher in MuJoCo; a valid build may fail its task checks."]}


def task_is_preserved(scene, program):
    """Reject bound-destination changes before the compiler can rebind a task."""
    task = scene.get("task")
    if not task:
        return
    from studio.task_api import bind_task
    if bind_task(program.scene, task) != task:
        raise ValueError("This edit would change the saved task or its destination bounds. Keep the tray and robot fixed, or create a new task.")


def edited_program(scene, edit):
    program = source_program(scene)
    selected = next((o for o in scene_objects(scene)["objects"] if o["id"] == edit.object_id), None)
    if selected is None:
        raise ValueError("That object is not in this scene. Select an object from the current revision.")
    if not selected["editable"]:
        raise ValueError(" ".join(selected["constraints"]))
    if edit.position is not None and any(abs(edit.position[i] - selected["position"][i]) > 1e-7 for i in range(3) if i not in selected["position_axes"]):
        raise ValueError("This object moves only along its enabled position axes.")
    if edit.size is not None and not selected["dimensions_editable"]:
        raise ValueError("This object's dimensions are fixed by the simulator.")
    if edit.color is not None and not selected["color_editable"]:
        raise ValueError("This object's material is fixed by the simulator.")
    data = program.model_dump()
    spec = data["scene"]
    if edit.object_id in {"cube", "tray"}:
        if edit.position is not None:
            spec[edit.object_id + "_x"], spec[edit.object_id + "_y"] = edit.position[:2]
        if edit.color is not None:
            spec["cube_color"] = edit.color
    elif edit.object_id == "conveyor":
        belt = spec["conveyor"]
        if edit.position is not None:
            belt["x"], belt["y"] = edit.position[:2]
        if edit.size is not None:
            axis = 0 if belt["direction"] == "x" else 1
            belt.update(length=edit.size[axis] * 2, width=edit.size[1 - axis] * 2, height=edit.size[2] * 2)
    elif edit.object_id.startswith("cloth:"):
        cloth = next(c for c in spec["cloths"] if "cloth:" + c["name"] == edit.object_id)
        if edit.position is not None:
            cloth["position"][:2] = edit.position[:2]
    elif edit.object_id.startswith("body:"):
        body = next(b for b in spec["bodies"] if "body:" + b["name"] == edit.object_id)
        if edit.position is not None:
            body["position"][:2] = edit.position[:2]
    elif selected["kind"] in {"movable", "prop"}:
        kind, name = edit.object_id.split(":", 1)
        item = next(p for p in spec["objects" if kind == "object" else "props"] if p["id" if kind == "object" else "name"] == name)
        for field in ("position", "size", "color"):
            if getattr(edit, field) is not None:
                item[field] = getattr(edit, field)
    else:
        assembly = next(a for a in data["assemblies"] if "assembly:" + a["name"] == edit.object_id)
        ratios = np.array(edit.size or selected["size"]) / np.array(selected["size"])
        if any(any(abs(v) > 1e-7 for v in p["euler_xyz"]) for p in assembly["parts"]) and max(ratios) - min(ratios) > 1e-6:
            raise ValueError("This assembly has rotated parts. Resize all axes by the same factor.")
        if any(p["shape"] == "cylinder" for p in assembly["parts"]) and abs(ratios[0] - ratios[1]) > 1e-6:
            raise ValueError("This assembly contains round cylinders. Resize X and Y by the same factor.")
        center, target = np.array(selected["position"]), np.array(edit.position or selected["position"])
        for part in assembly["parts"]:
            part["position"] = (target + (np.array(part["position"]) - center) * ratios).tolist()
            part["size"] = (np.array(part["size"]) * ratios).tolist()
        if edit.color is not None:
            assembly["color"] = edit.color
    # A manual spatial change asks the supplied teacher to plan for this layout.
    # Replaying stale waypoints can otherwise hide that the edited object moved.
    data["waypoints"] = None
    try:
        result = type(program).model_validate(data)
        task_is_preserved(scene, result)
    except ValueError as exc:
        raise ValueError(validation_message(exc)) from None
    changes = [verb for field, verb in (("position", "Moved"), ("size", "Resized"), ("color", "Recolored")) if getattr(edit, field) is not None]
    summary = " and ".join(changes) + " " + selected["label"] + "."
    return result, summary


def _part(position, size, collision=True):
    return {"shape": "box", "position": position, "size": size, "collision": collision}


COMPONENTS = [
    {"id": "workbench", "name": "Workbench", "description": "Raised work surface with four separate legs.", "color": [.38, .45, .48, 1], "position": [-.39, .24, .1],
     "parts": [_part([0, 0, .075], [.10, .07, .01]), *[_part([x, y, -.015], [.01, .01, .075]) for x in [-.08, .08] for y in [-.05, .05]]]},
    {"id": "guard", "name": "Safety guard", "description": "Solid rear guard panel and two feet. Static geometry only.", "color": [.85, .57, .13, 1], "position": [0, .33, .10],
     "parts": [_part([0, 0, .02], [.18, .008, .08]), _part([-.14, 0, -.085], [.025, .035, .015]), _part([.14, 0, -.085], [.025, .035, .015])]},
    {"id": "bin", "name": "Open bin", "description": "Open container with a bottom and four separate wall colliders.", "color": [.2, .47, .58, 1], "position": [.43, .25, .05],
     "parts": [_part([0, 0, -.04], [.075, .06, .01]), _part([-.07, 0, 0], [.005, .06, .04]), _part([.07, 0, 0], [.005, .06, .04]), _part([0, -.055, 0], [.065, .005, .04]), _part([0, .055, 0], [.065, .005, .04])]},
    {"id": "cabinet", "name": "Control cabinet", "description": "Cabinet body with a raised control panel. No articulated door.", "color": [.46, .53, .48, 1], "position": [-.45, .25, .12],
     "parts": [_part([0, 0, 0], [.07, .055, .12]), _part([0, -.06, .045], [.045, .007, .04])]},
]


def component_catalog(scene=None):
    program = source_program(scene) if scene else None
    choices = []
    for component in COMPONENTS:
        reason = None
        if program:
            try:
                insert_component(scene, ComponentInsert(component_id=component["id"]))
            except ValueError as exc:
                reason = validation_message(exc)
        choices.append({key: deepcopy(component[key]) for key in ("id", "name", "description", "position")} | {
            "parts": len(component["parts"]), "available": reason is None, "constraint": reason})
    return {"components": choices, "constraints": ["Components use separate box colliders and consume the scene's geometry budget. Placement feasibility is measured after insertion."]}


def insert_component(scene, request):
    component = next((c for c in COMPONENTS if c["id"] == request.component_id), None)
    if component is None:
        raise ValueError("Unknown component. Choose workbench, guard, bin, or cabinet.")
    program = source_program(scene)
    data = program.model_dump()
    position = request.position if request.position is not None else component["position"]
    parts = deepcopy(component["parts"])
    for part in parts:
        part["position"] = [p + offset for p, offset in zip(position, part["position"])]
    names = {a["name"] for a in data.get("assemblies", [])} | {p["name"] for p in data["scene"]["props"]}
    number = 1
    name = component["id"]
    while name in names or any(value.startswith(name + "_") for value in names):
        number += 1
        name = f"{component['id']}_{number}"
    if "assemblies" in data:
        data["assemblies"].append({"name": name, "color": component["color"], "parts": parts})
    else:
        data["scene"]["props"].extend({"name": f"{name}_{i}", "shape": p["shape"], "position": p["position"], "size": p["size"], "collidable": p["collision"], "color": component["color"]} for i, p in enumerate(parts))
    data["waypoints"] = None
    try:
        result = type(program).model_validate(data)
        task_is_preserved(scene, result)
    except ValueError as exc:
        raise ValueError("Cannot add this component: " + validation_message(exc)) from None
    return result, "Added " + component["name"] + "."


def scene_conversation(scene, limit=40):
    lineage, seen, current = [], set(), scene
    truncated = False
    while current:
        if current["id"] in seen:
            truncated = True
            break
        seen.add(current["id"])
        lineage.append(current)
        parent = current.get("parent_id")
        if not parent:
            break
        if len(lineage) >= limit:
            truncated = True
            break
        try:
            current = db.get("scene", parent)
        except KeyError:
            truncated = True
            break
    return {"scenes": list(reversed(lineage)), "truncated": truncated}


def author_context(draft):
    if not draft.get("parent_id"):
        return ""
    parent = db.get("scene", draft["parent_id"])
    history = scene_conversation(parent, limit=8)
    selected = None
    if draft.get("selected_object_id"):
        selected = next((o for o in scene_objects(parent)["objects"] if o["id"] == draft["selected_object_id"]), None)
        if selected is None:
            raise ValueError("The selected object is no longer in this revision. Select it again.")
    brief = {"selected_object": selected, "earlier_turns": [{"scene_id": s["id"], "request": s.get("prompt", "")[:1500],
        "result": s.get("summary", "")[:1200], "status": s.get("status"), "source": s.get("source")} for s in history["scenes"]],
        "history_truncated": history["truncated"]}
    return "\n\nStudio conversation context follows as data. Resolve words such as 'this' against selected_object. The current user request takes precedence over earlier turns. Preserve the selected object's identity unless the user asks to replace it. Keep the saved task requirements and destination unchanged.\n" + json.dumps(brief, ensure_ascii=False)


def discard_edit_candidate(artifact):
    """Remove only assets created by an unpublished Blender edit candidate."""
    metadata = (artifact or {}).get("blender")
    if not metadata:
        return
    from studio.blender_adapter import build_folder
    folder = build_folder(metadata["build_id"])
    for asset_id in metadata.get("asset_ids", []):
        try:
            asset = db.get("asset", asset_id)
        except KeyError:
            continue
        if asset.get("source") == "blender" and asset.get("build_id") == metadata["build_id"]:
            db.delete("asset", asset_id)
    shutil.rmtree(folder, ignore_errors=True)


class MjcfInsert(StrictModel):
    """Articulated object from a sanitized MJCF document, placed on the table."""
    name: str = Field(min_length=1, max_length=16, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    xml: str = Field(min_length=10, max_length=256 * 1024)
    position: Vector = Field(default_factory=lambda: [0, 0, 0])
    yaw: float = Field(default=0, ge=-2 * math.pi, le=2 * math.pi)
    # Companion OBJ/STL meshes named by <mesh file=...>, base64-encoded; never read from disk.
    files: dict[Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")], Annotated[str, Field(max_length=360 * 1024)]] = Field(default_factory=dict, max_length=8)

    def decoded_files(self):
        import base64, binascii
        try:
            return {name: base64.b64decode(data, validate=True) for name, data in self.files.items()}
        except (binascii.Error, ValueError):
            raise ValueError("Companion mesh files must be base64-encoded") from None


def insert_mjcf(scene, request):
    from studio.mjcf_import import convert, database_store
    program = source_program(scene)
    created, save = [], database_store(request.name)
    def store(normalized):
        created.append(save(normalized))
        return created[-1]
    try:
        converted = convert(request.xml, request.name, request.position, request.yaw, request.decoded_files(), store)
        data = program.model_dump()
        data["scene"]["bodies"].extend(converted["bodies"])
        data["scene"]["equalities"].extend(converted["equalities"])
        data["scene"]["contact_excludes"].extend(converted["contact_excludes"])
        data["scene"]["tendons"].extend(converted["tendons"])
        data["scene"]["actuators"].extend(converted["actuators"])
        data["waypoints"] = None
        result = type(program).model_validate(data)
        task_is_preserved(scene, result)
    except ValueError as exc:
        # Meshes stored for a rejected import are removed again.
        for id in created:
            db.delete("asset", id)
        raise ValueError("Cannot add this MJCF object: " + validation_message(exc)) from None
    counts = converted["summary"]
    def count(n, word, plural=None):
        return f"{n} {word if n == 1 else plural or word + 's'}"
    summary = ", ".join([count(counts["bodies"], "body", "bodies"), count(counts["joints"], "joint"),
        count(counts["sites"], "site"), count(counts["equalities"], "constraint")]
        + [count(counts[key], word) for key, word in [("tendons", "tendon"), ("actuators", "actuator")] if counts[key]])
    return result, f"Imported {request.name} from MJCF: {summary}.", converted["warnings"]
