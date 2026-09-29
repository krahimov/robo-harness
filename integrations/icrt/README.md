# Run the pretrained ICRT actor

This integration runs the released compact [ICRT model](https://github.com/Max-Fu/icrt). It loads camera and action demonstrations into the model's context, then predicts actions from fresh camera images and robot proprioception. Weights remain fixed. There is no training step, replay fallback, or language input to this actor.

The scene-authoring agent and judge are separate from this worker. They can use GPT-6 Astra while the robot actor uses ICRT.

## Install and start

Run these commands from the repository root. The worker has its own Python 3.12 environment so the studio backend does not depend on PyTorch.

```sh
uv sync --project integrations/icrt
uv run --project integrations/icrt python integrations/icrt/prepare.py
uv run --project integrations/icrt python integrations/icrt/prepare_franka.py
uv run --project integrations/icrt python integrations/icrt/worker.py --port 8790
```

The prepare scripts download pinned source, weights, and robot assets into `.data/models/`, outside Git. The two model weight files total about 835 MiB. Python dependencies and robot meshes require additional disk space. The first worker startup loads the weights before its health endpoint becomes available.

CPU inference was exercised on an Apple M4 Pro with 24 GiB of memory. CUDA is available through `--device cuda` on a host with a working CUDA PyTorch installation. MPS is rejected because the upstream complex rotary embeddings have not been ported to that device.

The backend uses `http://127.0.0.1:8790` by default. Set `STUDIO_ICRT_ENDPOINT` on the backend for another worker address. For a remote worker, set the same `STUDIO_ICRT_TOKEN` in the backend and worker environments. The launcher requires a token before binding beyond localhost. Tokens are never part of experiments or artifacts.

```sh
curl http://127.0.0.1:8790/health
```

Select ICRT in the studio and choose demonstrations recorded with the same embodiment as the rollout. Select Franka Panda to execute all pose dimensions. Older recordings without a wrist camera need to be recorded again.

## What the actor receives

Each observation contains two RGB camera images and seven proprioception values:

| Field | Convention |
| --- | --- |
| Side camera | MuJoCo camera `overview`, rendered at 320 by 180 pixels |
| Wrist camera | MuJoCo camera `wrist`, rendered at 320 by 180 pixels |
| Position | Native: absolute `panda_link8` flange XYZ in meters relative to the robot base. Cartesian: projected TCP XYZ. |
| Rotation | Intrinsic Euler XYZ angles in radians |
| Gripper | Zero means open, one means closed |

The upstream wrapper applies its visual transforms, converts rotation to its six-dimensional representation, and converts absolute demonstrated actions to the checkpoint's delta-action representation. Returned actions are converted back to absolute XYZ, Euler XYZ, and gripper. The actor receives no object positions, success flags, judge results, or task text.

The adapter requests predictions at the upstream 15 Hz control rate. Franka runs at 15 Hz; the Cartesian fixture holds actions between prediction ticks. Demonstrations preserve their selected order and observation/action alignment. Recordings are sampled at 15 Hz and then evenly reduced if necessary to fit a shared budget of 256 context frames across at most four demonstrations. Artifacts record the exact selected frame indices and original frequency. Reducing a long demonstration changes its temporal spacing; treat that as an experimental variable.

The worker reserves the rest of the 512-frame sequence for rollout history. It refuses to overrun capacity. Only one session may use its model cache at a time. A session owns that cache until it closes or a later session replaces it after ten minutes without a request. Zero-demo runs are supported as ablations, but have no task instruction because this checkpoint does not condition on language.

Raw actions and executed actions are both recorded. Position commands are limited to the studio workspace and at most 4 cm of movement per axis per prediction. Each position intervention is marked in the trace. Native Franka retains predicted orientation and clips the gripper target to its physical range. Cartesian projection discards predicted rotation and explicitly records that loss.

## Franka simulation

The native fixture uses the articulated Panda from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), pinned to commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`. Seven arm joints move through bounded Jacobian inverse kinematics and MuJoCo position actuators. Finger contacts move the boxes; this fixture has no proximity weld grasp.

The robot base is at world coordinates `[-0.45, -0.15, 0]`. Snapshots include that transform, the actual forward-kinematics pose, gripper opening, contact state, and IK residuals. The model uses the actual pose, not the requested target. The MuJoCo timestep is 1/600 second with 40 substeps per control command.

The tabletop fixture changes fingertip friction to 1.2 and increases the gripper position stiffness to 500 N/m while retaining the original 100 N force limit. It adds gravity compensation for the arm. These controller and camera settings are research assumptions; they have not been calibrated against DROID hardware. Reachability and contact results should be judged per rollout.

MuJoCo camera images use the original robot meshes. Compatible browser recordings load the original CAD from a shared visual asset, with compact bounds as a fallback. Raw recording XML refers to installed Menagerie files by absolute path. Portable scene packages now include the robot assets and license; the [Runpod trial transfer](../runpod/README.md) instead rebuilds XML against the target's pinned robot installation while preserving demonstration states and actions. `STUDIO_FRANKA_ASSETS` can select a separately installed Panda asset directory.

Cartesian scenes remain available for testing the adapter plumbing. Their idealized gripper and fixed orientation are a different embodiment from Franka. A Cartesian demo cannot be used as a native Franka demonstration.

## Checkpoints, source changes, and licenses

The [author's model repository](https://huggingface.co/mlfu7/ICRT) supplies the compact policy and Cross-MAE visual encoder. `prepare.py` pins model revision `5cb0391`, checks the SHA-256 of both weights and the configuration, and writes a manifest. ICRT source is pinned to commit `e0c95887d577c95ef9d0eef5ee23960043d6cb5e`.

The local clone receives three portability changes: construct tensors on CPU, allocate KV caches on the model's selected device, and disable CUDA autocast when running on CPU. No weights change. Inference uses evaluation mode with gradients disabled. The runtime reads the tagged upstream YAML as plain mappings instead of instantiating arbitrary YAML objects.

Runtime `icrt-runtime-v2` also corrects a pose-reference error in the pinned upstream evaluator. After its first prediction, that evaluator converts a delta predicted for the current observation through the previous observation's pose. The studio recovers the predicted local delta and composes it onto the current pose. This correction preserves the upstream cached delta and leaves the weights unchanged. Model metadata records `action_reference=current_observation` and `upstream_pose_reference_correction=true`.

Backend adapter `icrt-studio-v2` accounts for the five-decimal timestamps stored in simulation snapshots. Its prediction tolerance prevents a 15 Hz timestamp such as `0.13333` from skipping a prediction. Worker and backend adapter versions are saved with each rollout so older traces remain distinguishable.

Backend adapter `icrt-studio-v3` also converts native fingertip poses to DROID flange poses for observations and demonstrations, then converts predicted actions back before execution. The two frames differ by 10.34 cm and a local Z rotation of 45 degrees. Four additional diagnostic episodes still failed, including both corrected-frame conditions. See the [failure investigation](../../docs/icrt-diagnosis.md) for kinematics tests, temporal ensembling, checkpoint audits, and remaining uncertainties about training-stage provenance and transfer.

Keep the upstream notices with any redistribution. The ICRT repository advertises Apache 2.0 at its top level, but its [`llama.py`](https://github.com/Max-Fu/icrt/blob/e0c95887d577c95ef9d0eef5ee23960043d6cb5e/icrt/models/policy/llama.py) carries a GPL-3.0 header. The model card does not declare a separate model license. This integration preserves the checkout and notices instead of claiming a single license for all downloaded material. The Panda assets include their own Apache 2.0 license.

## Verification and observed behavior

Run the adapter and native-physics checks with:

```sh
cd backend
uv run pytest -q tests/test_icrt.py tests/test_franka.py
```

From the repository root, check the runtime correction with:

```sh
uv run --project integrations/icrt pytest -q integrations/icrt/test_action_reference.py
```

That regression uses the pinned evaluator with a stubbed zero-delta prediction. It verifies that a moved or rotated robot holds its current pose, that the local cached action stays unchanged, and that resetting a session clears the previous pose. It does not load model weights.

The native physics tests skip when the separately downloaded robot assets are missing. Tests exercise camera/action alignment, ordered context sampling, zero-demo sessions, mismatched embodiments, rotation conventions, native joint movement, contact-only scripted placement, and retained raw/executed actions. Worker HTTP tests use a fake model for transport checks; they do not claim pretrained model performance.

Real checkpoint inference was also run locally on September 6, 2026. The loaded model had 177,389,052 parameters. A native Franka run with one demonstration executed 60 predictions in 9.609 seconds of wall time. Its end effector moved and changed orientation, with a maximum IK position residual of 0.000477 m. It did not grasp the box during those four simulated seconds. The contact-based scripted teacher placed the box and passed the required outcome checks. Short Cartesian zero-demo and one-demo trials also produced real predictions without grasping the box.

Those initial trials used the earlier adapter before the pose-reference and timestamp corrections. They establish working fixed-weight inference and context insertion, but their failures cannot be attributed solely to camera or simulation differences. They do not establish task success, improvement from demonstrations, or sim-to-real transfer. Compare full rollouts with the corrected adapters, the same scene seeds, and the same evaluation criteria before making those claims. Local evidence, including model hashes, context indices, and prediction traces, is saved in `.data/validation/icrt-franka-live.json` and `.data/validation/icrt-cartesian-live.json`.

A corrected matched comparison then ran three seeds each with zero, one, and two passing native demonstrations, using 150 predictions per trial. All nine trials completed with no task successes. The adapter corrections did not produce an observed success-rate gain. See the [current validation record](../../docs/validation.md#corrected-icrt-comparison) for configuration, evidence IDs, and limits.

## Worker protocol

The protocol identifier is `robot-harness-icrt-v1`.

| Request | Purpose |
| --- | --- |
| `GET /health` | Read readiness, model provenance, and interface conventions |
| `POST /sessions` | Start with an ordered list of demonstrations, or an empty list |
| `POST /sessions/{id}/act` | Submit one side/wrist/proprio observation and receive seven action values |
| `DELETE /sessions/{id}` | Release the cache for the next experiment |

Images are base64 JPEG or PNG fields named `side` and `wrist`; proprioception is a seven-element list named `proprio`. Each demonstration has an `id`, a list of observation `frames`, and an equally long list of paired `actions`. The studio adapter performs all scene-specific rendering and coordinate conversion before calling the worker.

## Feedback-driven experiments

The experimental `iterate.py`, `improve.py`, `teacher_probe.py`, and `tool_actor.py` scripts and their results are documented in [the iteration report](../../docs/iteration-results.md). Astra can propose bounded demonstration/camera/controller changes from judge feedback, while the task, actor, and holdout seeds remain fixed. The separate Astra tool-controller passed 3/3 factory holdout seeds; the additional ICRT tests did not succeed. These commands are not yet connected to an autonomous-run UI.

Run the experiment-boundary tests from the repository root with:

```sh
PYTHONPATH=backend:. backend/.venv/bin/python -m pytest integrations/icrt/test_iteration.py -q
```

## Complete reference-episode diagnostics

`fetch_reference.py` downloads four fixed complete episodes from a pinned ICRT-MT revision using bounded range reads. `reference_eval.py` compares zero/one context with production predicted-action history, corrected recorded past history, and the upstream notebook procedure. It verifies sample hashes, disjoint episode IDs, and complete frame indices, and refuses to overwrite results.

The [completed reference evaluation](../../docs/icrt-reference-evaluation.md) found a context benefit on all three query episodes with the current runtime. This establishes offline action-prediction improvement on these samples, not closed-loop robot success. A secondary truncated-context test lost the benefit. Reproduction commands and limitations are in the report.

The latest [simulation transfer diagnosis](../../docs/icrt-transfer-diagnosis.md) separates action inference from physics, tests pose-frame and wrist-image changes, and retains a failed 415-step camera-intervention rollout. These diagnostic scripts leave production inference unchanged. Run `uv run --project integrations/icrt pytest -q integrations/icrt/test_transfer_probe.py integrations/icrt/test_reference_eval.py integrations/icrt/test_action_reference.py integrations/icrt/test_temporal.py` to verify their transformation and inference contracts.


## Camera visibility and executed-command feedback

Current runtime `icrt-runtime-v3` and backend `icrt-studio-v4` correct cached action history after execution limits. A session whose metadata advertises `supports_executed_action_feedback=true` accepts optional seven-value `previous_action` alongside `observation` in an act request. It is the command actually sent after the previous prediction, expressed in the same absolute pose convention. Omit it on the first prediction. The adapter negotiates support and records feedback counts. Unsupported runtimes reject that field; legacy clients continue to work without it.

New Panda recordings use the `side-mounted-pinch-v3` wrist camera, aimed between the fingers from beside the hand. This corrects an out-of-crop target and an obstructed intermediate mounting. Old recordings retain their cameras; metadata records mismatches. Re-record passing teachers for matched-camera trials.

The [Robotiq and feedback investigation](../../docs/icrt-robotiq-and-feedback.md) records five additional failed Robotiq trials and two failed production pen trials, including the final camera configuration. The new research fixture has passing contact-based teachers, but ICRT still does not grasp. Its approximate hardware and cameras do not establish DROID calibration or sim-to-real transfer.
