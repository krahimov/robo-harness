# Deterministic scene-edit benchmark

Run the local batch from the repository root:

```sh
uv run --project backend python scripts/scene_author_benchmark.py
```

The script exercises structured scene authoring and the same object-edit/component functions used by the studio. Each case runs real MuJoCo physics and measures the supplied task. It makes no AI calls and requires no model credentials or running studio server.

It uses an isolated temporary data directory, then writes only its report to `.data/validation/scene-edit-benchmark.json`. The temporary scene data is removed after the batch. Existing projects, demonstrations and task versions remain untouched.

## Cases and measurements

The default batch has twelve fixed cases:

- Block transfer from different source and tray placements.
- Lift-and-hold with a control cabinet.
- Two-box sorting with a manually moved box.
- Conveyors moving along X and Y, including a moved source box.
- Cabinet resizing, a guard and box recoloring, and an open bin with a moved source.
- A workbench with seeded initial-position variation.
- Two-box sorting with both a bin and guard.
- An obstructed destination that compiles but should fail its placement requirement.

All cases use explicit seeds, and the report includes each final declarative program and edit requests. Seeded variation is exercised in the workbench case. The obstruction case checks that a runnable scene is not counted as a successful task merely because it compiles.

The report records compile count, task pass/fail/unknown counts, expected-result count, elapsed time, simulator version, frame count, simulation duration, final object states and individual task outcomes. Each case's wall time includes program preparation, direct edits, teacher path construction and physics recording. Process startup and Python import time are excluded. The script exits with status 1 if a case fails to compile or its measured task result differs from the declared expectation.

The default uses `cartesian-gripper-v1`, whose grasp is an idealized proximity weld. Its teacher has direct access to scene state. This is **deterministic fixture coverage**, not a measure of general photo-authoring quality, autonomous planning, learned actor performance, ICL improvement or hardware transfer.

## Optional Blender and robot runs

Add two real Blender cases with:

```sh
uv run --project backend python scripts/scene_author_benchmark.py --blender \
  --output .data/validation/scene-edit-benchmark-blender.json
```

These add an open bin to a Blender design, then move and resize a source assembly in a separate case. The normal Blender adapter builds visual meshes and separate collision boxes, runs MuJoCo, and renders the static layout. Blender files are temporary; the persistent report records Blender version and geometry counts. The installed Blender runtime is required, including normal native graphics access on macOS. A missing or failed Blender runtime counts as a failed case, not a skipped success.

`--embodiment franka-panda-v1` runs the batch with the supplied native Panda fixture when its normal assets are installed. This is an additional fixture investigation. The default Cartesian expectations may expose different reach or contact failures under Panda; inspect the outcome details rather than treating the default success count as a promised native-arm result.

Use `--output <path>` to keep different runs for comparison. Timing differences depend on the machine and render workload; the script applies no performance threshold.

## Local result

On September 6, 2026, the default batch compiled all 12 cases. Eleven passed their task and the obstructed-destination case failed as intended. All 12 matched their declared outcomes in 1.36 seconds. Evidence is stored in `.data/validation/scene-edit-benchmark.json`.

The optional Blender batch also ran locally. All 14 cases compiled; 13 tasks passed and the expected obstruction failed. Both Blender cases passed, producing real meshes, collider geometry and layout renders. The 14-case batch took 6.40 seconds. Evidence is stored in `.data/validation/scene-edit-benchmark-blender.json`.
