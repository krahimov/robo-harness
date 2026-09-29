#!/usr/bin/env python3
"""Run deterministic scene-edit fixtures through real MuJoCo; never call an AI model."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from studio import storage
from studio.authoring import SceneProgram, compile_program
from studio.scene_editing import ComponentInsert, SceneEdit, edited_program, insert_component
from studio.schemas import Scene
from studio.task_api import bind_task, get_task


def fixtures():
    """Deliberately chosen layouts, including one solvable-build / failed-task case."""
    return [
        {"id": "plain_transfer", "seed": 7},
        {"id": "source_back_left", "seed": 17, "scene": {"cube_x": -.35, "cube_y": .11, "tray_x": .22, "tray_y": -.1}},
        {"id": "lift_with_cabinet", "seed": 29, "task": "lift-and-hold", "components": ["cabinet"]},
        {"id": "two_box_manual_move", "seed": 41, "task": "two-box-sort", "edits": [{"object_id": "object:blue", "position": [-.3, .22, .026]}]},
        {"id": "conveyor_x_move_source", "seed": 59, "scene": {"conveyor": {"direction": "x", "speed": .025}}, "edits": [{"object_id": "cube", "position": [-.27, -.08, .066]}]},
        {"id": "conveyor_y", "seed": 71, "scene": {"conveyor": {"direction": "y", "speed": .02}}},
        {"id": "cabinet_resize", "seed": 83, "components": ["cabinet"], "edits": [{"object_id": "prop:cabinet_0", "position": [-.45, .25, .1], "size": [.06, .045, .1]}]},
        {"id": "guard_and_recolor", "seed": 97, "components": ["guard"], "edits": [{"object_id": "cube", "color": [.2, .65, .35, 1]}]},
        {"id": "bin_and_source_move", "seed": 101, "components": ["bin"], "edits": [{"object_id": "cube", "position": [-.26, -.14, .026]}]},
        {"id": "workbench_seed_variation", "seed": 113, "scene": {"cube_x": -.3, "cube_y": -.16, "variation": .025}, "components": ["workbench"]},
        {"id": "two_box_bin_and_guard", "seed": 127, "task": "two-box-sort", "components": ["bin", "guard"]},
        {"id": "destination_obstructed", "seed": 131, "expected_status": "fail",
         "description": "Solid machinery fills the destination. Compilation should work, while task measurement should detect failed placement.",
         "scene": {"props": [{"name": "destination_obstruction", "position": [.28, .05, .15], "size": [.125, .095, .15], "color": [.7, .3, .15, 1]}]}},
    ]


def blender_fixtures():
    return [
        {"id": "blender_add_open_bin", "seed": 149, "builder": "blender", "components": ["bin"]},
        {"id": "blender_move_resize_machine", "seed": 163, "builder": "blender",
         "edits": [{"object_id": "assembly:machine", "position": [-.42, .26, .10], "size": [.06, .045, .1]}]},
    ]


def prepare(case, embodiment):
    task = get_task(case.get("task", "block-to-tray"))
    scene = Scene.model_validate({**task.get("suggested_scene", {}), **case.get("scene", {}), "seed": case["seed"], "embodiment": embodiment})
    data = {"name": case["id"], "description": case.get("description", "Deterministic scene editing benchmark fixture."),
            "assumptions": ["Fixture geometry and scripted teacher only. No photo inference, actor model or hardware validation."],
            "scene": scene.model_dump(), "waypoints": None}
    builder = case.get("builder", "mujoco")
    if builder == "blender":
        from studio.blender_schema import BlenderProgram
        data["assemblies"] = [{"name": "machine", "color": [.2, .35, .5, 1],
            "parts": [{"position": [-.45, .25, .12], "size": [.07, .055, .12]}]}]
        program = BlenderProgram.model_validate(data)
    else:
        program = SceneProgram.model_validate(data)
    task = bind_task(program.scene, task)
    def parent():
        result = {"builder": builder, "spec": program.model_dump(exclude_none=True), "task": task}
        if builder == "blender":
            result["blender"] = {"program": program.model_dump(exclude_none=True)}
        return result
    actions = []
    for id in case.get("components", []):
        program, summary = insert_component(parent(), ComponentInsert(component_id=id))
        actions.append(summary)
    for edit in case.get("edits", []):
        program, summary = edited_program(parent(), SceneEdit.model_validate(edit))
        actions.append(summary)
    return program, task, actions


def run_case(case, embodiment):
    started = time.perf_counter()
    result = {"id": case["id"], "fixture": deepcopy(case), "builder": case.get("builder", "mujoco"),
              "embodiment": embodiment, "expected_status": case.get("expected_status", "pass"), "compiled": False}
    try:
        program, task, edits = prepare(case, embodiment)
        result.update(program=program.model_dump(exclude_none=True), edits=edits, task_id=task["id"])
        compiler = compile_program
        if result["builder"] == "blender":
            from studio.blender_adapter import compile_blender
            compiler = compile_blender
        _, artifact, verification = compiler(program, task, embodiment)
        result.update(compiled=True, task_status=verification["status"], outcomes=verification["outcomes"],
                      frames=len(artifact["frames"]), simulation_seconds=artifact["frames"][-1]["t"],
                      final_objects=artifact["frames"][-1].get("objects", {}), simulator_version=artifact.get("simulator_version"))
        if artifact.get("blender"):
            result["blender"] = {key: artifact["blender"][key] for key in ("version", "visual_meshes", "collision_shapes", "collision_method")}
        result["matches_expectation"] = verification["status"] == result["expected_status"]
    except Exception as error:
        result.update(task_status="not_run", matches_expectation=False, error=f"{type(error).__name__}: {error}"[:800])
    result["wall_seconds"] = round(time.perf_counter() - started, 4)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", action="store_true", help="Also run two real Blender geometry/render fixtures. Requires the local Blender runtime.")
    parser.add_argument("--embodiment", choices=["cartesian-gripper-v1", "franka-panda-v1"], default="cartesian-gripper-v1",
                        help="Robot fixture for every case. Franka requires its normal installed assets.")
    parser.add_argument("--output", type=Path, default=ROOT / ".data/validation/scene-edit-benchmark.json")
    args = parser.parse_args(argv)
    cases = fixtures() + (blender_fixtures() if args.blender else [])
    original_root = storage.ROOT
    started = time.perf_counter()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="scene-edit-benchmark-") as temporary:
            storage.ROOT = Path(temporary)
            for case in cases:
                result = run_case(case, args.embodiment)
                results.append(result)
                print(f"{result['id']}: compiled={result['compiled']} task={result['task_status']} expected={result['expected_status']} {result['wall_seconds']:.3f}s", flush=True)
    finally:
        storage.ROOT = original_root
    summary = {"cases": len(results), "compiled": sum(r["compiled"] for r in results),
               "compile_failures": sum(not r["compiled"] for r in results),
               "task_pass": sum(r["task_status"] == "pass" for r in results),
               "task_fail": sum(r["task_status"] == "fail" for r in results),
               "task_unknown": sum(r["task_status"] == "unknown" for r in results),
               "expected_results": sum(r["matches_expectation"] for r in results),
               "wall_seconds": round(time.perf_counter() - started, 4)}
    report = {"format": "studio-scene-edit-benchmark-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Deterministic fixture coverage of structured scene authoring and direct edits. This is not a benchmark of general photo authoring, robot ICL or hardware transfer.",
              "ai_calls": 0, "isolated_temporary_store": True, "embodiment": args.embodiment, "includes_blender": args.blender,
              "python": platform.python_version(), "platform": platform.platform(), "summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".tmp")
    temporary_output.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary_output.replace(args.output)
    print(json.dumps(summary), flush=True)
    print(f"Evidence: {args.output.resolve()}", flush=True)
    print("Deterministic fixtures and scripted teachers only; no model call or learned actor result.", flush=True)
    return 0 if summary["expected_results"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
