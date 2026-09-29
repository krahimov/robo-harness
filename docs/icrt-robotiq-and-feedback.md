# ICRT camera, gripper, and action-history investigation

September 13, 2026. Two integration defects were fixed, but ICRT still has no task success in these new tests. Weights, task thresholds, and judge rules remained unchanged. All trials were local development diagnostics, not independent holdouts.

## Fixes

The original Panda wrist mount put the fingertip site outside the model's square center crop: a 6 cm offset on each image axis versus a crop half-width of about 3.94 cm at that depth. A first aimed-camera correction still looked through the hand body. Rendered-image inspection caught that occlusion. The final `side-mounted-pinch-v3` camera sits beside the hand and aims between the fingertips. A regression test checks projection and ray-cast visibility at two arm poses. This does not guarantee visibility throughout every task or establish real-camera calibration.

The worker also cached raw predictions even when the adapter clipped them before execution. `icrt-runtime-v3` accepts `previous_action` and converts that executed command into its cached representation relative to the previous observation. `icrt-studio-v4` negotiates support, sends the actual prior command, and records feedback counts. Measured robot pose remains a separate observation. Older workers remain usable; runtimes that do not advertise support reject feedback. The experimental temporal runtime does not advertise it.

Old artifacts retain their original XML, images, and results. New metadata records context/query camera versions and flags differences. For a matched-camera trial, re-record the demonstration on the current simulator. `production_probe.py` re-executes saved teacher commands, saves a new artifact, and requires it to pass the frozen task before testing ICRT.

## Robotiq control experiment

The separate `franka-robotiq-2f85-research-v1` fixture attaches the official Menagerie 2F-85 to the Panda arm using assets pinned at `8161bba264d7fa7c99ca301e91e7fb44737676ad`. It retains articulated fingers, coupling, and native contacts without a grasp weld. Mounting, cameras, and controller are approximations, not calibrated DROID hardware. This fixture is available through research scripts, not the production embodiment selector. Asset licenses remain with the downloaded files.

The initial scripted teacher slipped. Aligning the cube faces with its grasp produced a passing teacher: 63 held frames and maximum cube height 18.6 cm. That scene was fixed for subsequent model trials. Teacher actions use object state; ICRT receives only images, robot pose, and context. Teacher success is not ICRT success.

Each Robotiq query used seed 72000, one full 137-action demonstration from seed 71, and 220 predictions:

| Intervention | Result | Held frames | Closest distance | Clipped predictions |
| --- | --- | ---: | ---: | ---: |
| Initial research cameras | Fail | 0 | 27.44 cm | 149 |
| Aimed camera behind hand | Fail | 0 | 27.72 cm | 183 |
| Reference-like starting flange pose | Fail | 0 | 22.00 cm | 141 |
| Same setup, executed-command feedback | Fail | 0 | 22.00 cm | 132 |
| Final side camera mount with feedback | Fail | 0 | 21.96 cm | 149 |

The paired feedback test reduced clipping but did not produce an approach or grasp. It does not establish task-performance improvement.

The original pen task also failed both production-adapter retests. Each used 415 steps and 414 executed-command feedback updates, with a newly recorded passing teacher and seed 62000. The first, obstructed-camera run reached 23.66 cm with 349 clipped predictions. The final side-mounted-camera run reached 24.18 cm with 367 clipped predictions. Both had zero grasp frames and no execution error. Long-horizon context was sampled to 95 frames; the earlier full-context offline diagnostic also failed to resolve the synthetic prediction gap.

All raw states, predictions, XML, teacher verdicts, and plans are under `.data/validation/icrt-robotiq-1/`. Final results are in `unoccluded-feedback/` and `production-pen-unoccluded/`. Intermediate failures remain preserved.

## Next useful work

1. **Observation audit:** show the exact cropped tensors the model receives; test target visibility in teacher recordings; compare camera versions and command distributions before launching trials.
2. **Matched short task:** reconstruct one released real demonstration's camera geometry, gripper, appearance, and control scale as closely as available calibration permits. Start with approach/grasp/lift and compare zero/one context on fresh seeds. Merely replacing the gripper did not supply that calibration.
3. **Separate error sources:** compare predictions on fixed teacher observations with predicted history versus recorded past commands, then run the same setup in closed loop. Prior real-data controls and synthetic-data failures suggest a sensorimotor distribution mismatch, but do not isolate appearance as its sole cause.
4. **Keep Astra useful:** use bounded skills and camera/state checkpoints for recovery while ICRT remains experimental. Judges can propose demonstration or camera revisions during development; task criteria and final holdouts stay fixed. Short skill sessions reduce context pressure, but history resets must be measured as an intervention.

Changing weights through simulator adaptation is a separate track, not evidence of fixed-weight in-context learning.

## Reproduction and verification

After preparing the existing pinned Panda assets, add the Robotiq directory:

```sh
git -C .data/models/menagerie sparse-checkout add robotiq_2f85
```

With the current worker on port 8790, run from the repository root:

```sh
PYTHONPATH=backend backend/.venv/bin/python integrations/icrt/robotiq_trial.py \
  --output .data/validation/robotiq-new-trial --executed-feedback --seeds 72000

PYTHONPATH=backend backend/.venv/bin/python integrations/icrt/production_probe.py \
  --output .data/validation/pen-new-trial \
  --experiment de8c6487b6f44edeb759ecd80aea468c \
  --demo f72fc7aee300431f9823dbbf3bc11f5e --seed 62000 --steps 415
```

The pen command requires this workspace's saved experiment/demo. Commands refuse to overwrite existing output directories. No paid GPU was started. Backend tests cover camera visibility, contact teachers, transforms, protocol behavior, and adapter feedback. Runtime tests independently verify the previous-pose action conversion. These tests are distinct from the failed checkpoint-driven task trials.
