# Validation record

The current ICL, native-robot, and judge checks are in the September 6 section below. Earlier sections document what was available at each stage.

## Initial local validation

Validated September 4, 2026, on macOS ARM64 with Python 3.12, MuJoCo 3.12, and Node.js 25.

- Backend: 11 tests passed. Two deprecation warnings come from the installed Starlette test client; no test failures.
- Frontend: TypeScript and Vite production build passed. Vite reports a bundle-size advisory for the Three.js-containing JavaScript bundle.
- Browser: launched a four-attempt simulation batch, inspected outcomes, saved a human review, promoted the accepted recording, displayed an actual MuJoCo camera frame, recorded a custom keyframe path, and saved a new judge version.
- Responsive layout: no horizontal document overflow at 390 × 844.
- Baseline check: four of four attempts passed at the demonstrated object position.
- Variation check: three of eight replay attempts passed with ±5 cm placement variation using seeds 42–49.

These are trajectory-replay fixture results, not robotics foundation-model results. The model and VLM HTTP integrations were tested against controlled protocol fixtures; no live robotics foundation model or external VLM provider was configured.

Browser validation created example runs, demonstrations, and a judge version in the local `.data/` store. They are excluded from Git. A clean checkout generates only the initial demonstration and state judge on first startup.

## Scene builder validation, September 5, 2026

- 67 backend tests passed across the runtime, scene simulation, photo normalization, author protocol, and author API suites. The original 11 runtime tests still pass.
- TypeScript and Vite production build passed. The existing Three.js bundle-size advisory remains.
- Scene simulation tests cover shifted tray targets, real contact with authored props, and one-box conveyor motion and pickup in X, Y, and reverse directions.
- Photo tests cover JPEG, PNG, WebP, EXIF orientation, transparent backgrounds, image byte/pixel limits, animation rejection, and safe storage paths.
- Author tests verify the exact `gpt-6-astra` Responses request, photo input, tool continuation, rendered-frame feedback, bounded programs, provider error handling, cancellation, and preservation of a runnable scene after a failed repair.
- API tests verify photo-only drafts, missing-key errors, reference persistence, scene revisions, restart recovery, idempotent demonstration recording, source lineage, and bundle export.
- Actual MuJoCo rendering produced initial and final camera images for the conveyor template. Its state checks passed.
- Browser verification covered a photo upload, saved-photo reattachment, draft persistence across reload, conveyor template playback, a manually edited scene revision, saving its demonstration, and a completed replay experiment using that scene.
- The edited conveyor demonstration passed one of one replay attempts. The mobile layout had no horizontal document overflow at 390 pixels.
- Camera capture was tested with an artificial browser media stream. It saved a JPEG and stopped its media track when closed. A simulated permission denial displayed the file fallback. No physical camera was used in testing.

At this stage, the Astra protocol tests used mocked provider responses with real MuJoCo execution. No OpenAI key was configured. The subsequent live Codex test is recorded below. No learned robotics model was connected.

## Codex subscription author validation, September 5, 2026

- 93 backend tests passed. Coverage includes provider selection, preserved draft connections, legacy API drafts, saved-login probing, exact model selection, reference attachments, simulator feedback, cancellation, deadlines, sanitized errors, and excluding backend credentials from the Codex process environment.
- TypeScript and Vite production build passed. The existing Three.js bundle-size advisory remains; backend tests still report two Starlette deprecation warnings.
- A live run used Codex CLI 0.153.0 with its saved ChatGPT login and exactly `gpt-6-astra`. No API key was configured or used.
- Astra received an uploaded MuJoCo camera render and a brief requesting a conveyor pick-and-place scene. It produced one runnable candidate, received actual rendered frames and numerical state checks, and finished with an evidence-based summary.
- The recorded nine-second operation passed the placement checks. The final box position was approximately `(0.281, 0.050, 0.041)` metres, with release, settling, and no-drop checks passing. The configured conveyor moved along X at 0.025 m/s.
- The live scene is retained in the local data store as `9a22492aa52c49acb76b792505799f12`, with its source image, program, recording, connection, model, usage, and checks.
- Browser checks confirmed the saved-login status, connection switching, missing-API-key guidance, preserved scene selection, and playback of all 181 frames through 9 seconds. Desktop and 390-pixel mobile layouts had no horizontal overflow. These checks triggered no additional model calls.

This verifies subscription authentication and the complete authoring loop with a known simulator reference. It does not measure factory-photo reconstruction, demonstrate learned robot control, or validate hardware execution. The OpenAI API connection remains covered by controlled protocol fixtures rather than a live API call.


## ICL harness milestone, September 6, 2026

### Runtime and release checks

- The backend passes 147 tests with the optional Panda assets installed. A fresh source copy with locked dependencies passes 145 tests and skips the two native-asset tests; it has no model weights or provider credentials. Both runs report the two existing Starlette test-client deprecation warnings.
- The frontend passes TypeScript checking and its Vite production build in both the working tree and the fresh source copy. The existing bundle-size advisory remains.
- Browser checks cover desktop and 390-pixel mobile views, context comparison controls, judge labeling and calibration views, task version editing, robot selection, measurements, OBJ upload, draft persistence, native demonstration handoff, and scene playback. No human outcome labels were fabricated during validation.
- Supplied block-to-tray, lift-and-hold, and two-box sorting teachers pass their state checks on both the Cartesian fixture and native Franka. The Franka pickup uses finger contact and joint dynamics, without a proximity weld. An imported mesh was rendered and compiled into a passing native scene.
- Open-source preparation includes the studio's MIT license, contribution guide, issue/PR templates, locked dependencies, a local smoke script, and GitHub Actions configuration. GitHub-hosted CI and public release have not been executed. Downloaded model/source/robot assets retain their separate upstream terms.

### Factory-photo authoring

The live reference was a real [factory conveyor photo by falco, released under CC0](https://commons.wikimedia.org/wiki/File:Conveyor_system_in_a_factory.jpg). No user-owned factory photo was supplied. The brief explicitly requested a tabletop reinterpretation with supplied dimensions; this is not a calibrated reconstruction of the photographed factory.

- The exact `gpt-6-astra` model used the saved Codex ChatGPT login to author the native Franka scene. Its first candidate passed numerical task checks after actual MuJoCo execution.
- The saved teacher contains 134 frames and 133 actions over 8.867 seconds at 15 Hz. The final box position was approximately `(0.248, 0.049, 0.041)` metres. Placement, release, settling, and the task's off-table-drop check passed.
- The native scene is `af1456203a2445afb27694a6ece979a2`; its demonstration is `dbcc8746f79e46d8aba134ad3a1e807b`. A second teacher at a shifted pickup position, `b34135649d994ab99ae80c03c57431bc`, also passed.
- Known measurements, human task requirements, selected embodiment, source image, program, rendered feedback, and model connection remain attached to the saved scene. The browser showed the photo and completed Franka scene, then opened the same demonstration in Workspace.
- Scene authoring is bounded by the documented scene language. Curved machinery, scale, unseen geometry, camera calibration, and hardware behavior remain assumptions. The photo and recordings are local artifacts excluded from Git.

### Matched ICRT comparison

The released compact DROID-pretrained ICRT checkpoint ran with fixed weights and 177,389,052 parameters. Its native adapter supplies side/wrist images and actual robot proprioception; it executes position, orientation, and gripper predictions through the Panda controller. The setup records checkpoint hashes, context frame indices, raw predictions, executed commands, and command interventions. See the [integration guide](../integrations/icrt/README.md) for precise coordinate and controller conventions.

Comparison `ee04cebafd124274b2ef37f3179c674b` used the factory-photo Franka scene, the same task and state judge, seeds 2000–2002 with 2 cm scene variation, and 150 control steps (10 simulated seconds) per trial. The selected demonstration seeds differ from those evaluation seeds. Context order was fixed; only demonstration context changed.

| Demonstration context | Pass | Fail | Unknown/error |
| --- | ---: | ---: | ---: |
| None | 0 | 3 | 0 |
| One passing teacher | 0 | 3 | 0 |
| Two passing teachers | 0 | 3 | 0 |

All nine trials produced real model predictions and completed. A subsequent audit identified two adapter defects in this initial comparison: the pinned upstream evaluator converted local motion against the previous pose, and five-decimal timestamp rounding skipped one prediction in each 150-step episode. Version `icrt-runtime-v2` corrects the pose reference while retaining the checkpoint and upstream cached delta; `icrt-studio-v2` corrects timestamp tolerance. Four runtime regression tests exercise the real pinned evaluator with stubbed network outputs and local-motion composition, and a backend regression verifies exactly 150 prediction calls. The corrected comparison is recorded separately below. There was no measured success-rate gain. Three trials per condition are insufficient for a reliable general performance estimate; the 95% Wilson interval for each observed 0/3 success rate extends from 0 to about 56%. ICRT has no language conditioning, so the zero-demo condition has no task instruction. These results do not establish transfer to DROID hardware or diagnose the cause of failure.

The exported comparison is retained locally at `.data/validation/factory-icrt-comparison.json`, with scene and demonstration IDs in `.data/validation/live-milestone.json`. This initial comparison remains immutable if later adapters or judges change. Earlier shorter native/Cartesian inference evidence is in `.data/validation/icrt-franka-live.json` and `.data/validation/icrt-cartesian-live.json`.

### Corrected ICRT comparison

Comparison `c483d132b2a4478fafb04d3a6e345e30` reran the same scene, teachers, seeds, task, judge, and 150-step horizon after both fixes. All nine trials completed. Each of the zero-, one-, and two-demonstration conditions again scored **0 passes and 3 failures**, with no unknowns or provider errors. The corrections did not produce a measured task-success gain in this small experiment.

Every saved trace was checked for exactly 150 predictions and 150 executed actions, `icrt-studio-v2` and `icrt-runtime-v2` metadata, the current-observation pose reference, the same policy hash, fixed weights with zero gradient updates, preserved context order, and retained native orientation. The exported comparison is `.data/validation/factory-icrt-comparison-corrected.json`; `.data/validation/factory-icrt-corrected-audit.json` records the per-trial audit. The earlier comparison and Astra evaluation remain intact.

This completes the live photo-to-demonstration-to-learned-actor-to-evaluation workflow. Reliable learned task execution remains unresolved. Camera, scene, controller, context representation, and model compatibility require further controlled investigation; these runs do not isolate one cause or demonstrate hardware transfer.

### Live judge and human calibration

- A live Codex Astra judge inspected a successful recorded attempt using both sampled numerical states and actual MuJoCo camera frames. It returned all required outcomes as pass with human review required. The trace used four turns and recorded 40,892 input and 556 output tokens.
- That evaluation is `5713d186bb3f4e1c9123cd5bed4e35b2`, attached to episode `8d5f3e84f7f24a57ba65dce1327c5175`; its original state evaluation `4beaf86d4ebd4f1e87cb6d3ff83f7736` remains preserved.
- Astra also judged the one-demo ICRT attempt at seed 2000: placement failed, release passed, settling failed, and the off-table-drop check passed. All four outcomes matched the original state evaluation. It inspected 40 sampled state frames with complete-interval statistics and four actual camera images, using 46,378 input and 428 output tokens. Evaluation `d1bba151d1ee47b9913150dd038359fb` requires human review; its local evidence is `.data/validation/icrt-franka-astra-judge.json`. The comparison response remained unchanged after rejudging, preserving its original evaluation ID. Agreement with one state evaluation is not human calibration.
- The judge lab records per-outcome human labels against an exact judge version, calculates agreement from resolved labels, samples uncertain results and clear passes, and supports draft rubric proposals from corrections. Tests exercise those mechanics. No human calibration dataset or measured human agreement is claimed.
- State evidence distinguishes sampled frame coverage from complete-interval aggregate statistics. A VLM verdict is retrospective; it does not supervise live motion or authorize physical hardware operation.

## Blender authoring, September 6, 2026

- The backend passes 158 tests. Eleven Blender integration tests cover declarative geometry limits, visual/collision separation, native mesh conversion, failed-build cleanup, cancellation, credential filtering, provider feedback, and download access. These tests use a controlled Blender transport with real MuJoCo execution. The two existing Starlette deprecation warnings remain.
- TypeScript checking and the Vite production build pass. Browser verification covered builder selection and readiness, the Blender layout image, playback of the MuJoCo recording, saving a demonstration, and running it in Workspace. No browser errors or horizontal overflow were observed at the current 577-pixel viewport.
- Live generation used the official `bpy` 5.2.1 LTS package in its separate Python 3.13 environment. A procedural conveyor example compiled and passed the native Franka task. CPU Cycles rendered the layout, including the original Panda geometry at its initial MuJoCo pose.
- A second live run used exactly `gpt-6-astra` through the saved Codex login with the same CC0 factory reference described above. Its first candidate generated two machinery meshes and eight separate collision boxes. The native Franka teacher passed placement, release, settling, and the off-table-drop check over 134 frames and 8.867 seconds at 15 Hz. The run recorded 29,110 input and 1,282 output tokens.
- The photo-guided scene is `de69517757b44feeab77fdd368e10231`, with Blender build `5f89841d72104d0d9c9b06c5ac46e9f2`. Browser saving created demonstration `1a019f62b128463f9a6d096152401216` and preserved the Blender build metadata in its source recording.
- A fresh replay experiment, `21ce55bde5dd49aa94ecb0bd55021cf3`, passed one of one attempts at seed 42 with zero scene variation and the existing state judge. This checks that generated geometry and the saved demonstration survive the author-to-experiment handoff. Replay does not measure in-context learning.
- The preview, editable `.blend`, and ZIP downloads returned HTTP 200 through the frontend proxy and matched their local files byte for byte. The ZIP passed its integrity check and contains the scene, render, design, manifest, generated OBJ meshes, simulation evidence, and Panda asset license. The `.blend` also embeds that license as a text block. Local evidence is retained in `.data/validation/blender-photo-live.json`, `blender-handoff.json`, and `blender-downloads.json`.

The photo guided a tabletop approximation with supplied dimensions. Blender generated static machinery and a layout render; MuJoCo executed the robot dynamics and recorded the motion. Curved conveyors, unobserved dimensions, and physical material properties remain assumptions. The generated machinery uses bounded box/cylinder assemblies and conservative box colliders. The package's MuJoCo XML references separately installed Panda assets, so it is not a portable standalone runtime. This work adds scene authoring and leaves the earlier ICRT results unchanged. The OpenAI API author protocol remains covered by controlled fixtures; the complete Blender path was exercised live through Codex only.

## Conversational workspace and direct editing, September 6, 2026

- The backend passes 208 tests. New coverage checks immutable scene edits, object constraints, task preservation, conversation context, component insertion, cancellation cleanup, portable package validation and rollback, measurement preservation, and the robot visual asset. TypeScript checking and the Vite production build pass. The existing bundle-size advisory and two Starlette deprecation warnings remain.
- Browser checks covered the conversation and simulator at 1440 by 900 pixels, a narrow layout at 390 by 844 pixels without document overflow, photo references, persistent unsent drafts, numeric object movement, a real translation-handle drag, selected-object prompting, component insertion, cancellation, package import, and saving a demonstration.
- The live selected-object request used exactly `gpt-6-astra` through the saved Codex login. It changed the BlueSteel assembly to the requested dark navy color while preserving its geometry, the edited box position, robot, tray, and human task. The run recorded 35,041 input and 1,775 output tokens. The resulting native Franka recording passed all four task checks. Evidence is `.data/validation/conversational-scene-live.json`.
- Adding a guard created final scene `c16e6f046273441d9dde361e891846ed`. Its five-revision conversation retains the original photo request, two direct moves, the Astra edit, and component insertion. Canceling a subsequent component build left this parent ready, removed staged assets, and published no recording. Browser saving created demonstration `73abf744fd834e6481fc519086491e7f`, named `Factory conveyor / interactive authoring`, with 134 frames over 8.867 seconds.
- Compatible recordings now display the original Franka CAD meshes in the browser. Tests compare their transforms with MuJoCo at two poses and different scene geometry offsets, and check compressed delivery, caching, and fallback behavior. This changes rendering, not robot dynamics.
- The structured scene-edit benchmark compiled all 14 cases with Blender enabled. Thirteen passed their tasks and one deliberately obstructed case failed as expected. This measures compiler, component, and teacher behavior across known layouts; it makes no AI calls. Evidence is `.data/validation/scene-edit-benchmark-blender.json`.
- Export/import into an isolated data store rebuilt a native Franka scene with fresh mesh, reference-photo, and task identifiers and passed all four task checks. Browser import also rebuilt a passing scene and exposed the preserved Blender source attachments. The final workspace export is a valid 10,550,324-byte ZIP with 83 files and both supplied scale measurements. It is retained as `.data/validation/agent-workspace-portable.zip`; other browser and runtime evidence is `.data/validation/agent-workspace-ui.json`.

Portable packages now include the robot assets and license alongside the standalone MuJoCo XML, resolving the earlier Blender export limitation above. Studio import recompiles the declarative scene using the installed canonical robot runtime. It preserves Blender source as attachments; it does not execute archive scripts or infer physics from arbitrary desktop Blender edits. Authoring still uses bounded static assemblies. These checks establish the editing and packaging workflow, not general factory reconstruction or an improvement in learned robot task success. The earlier ICRT comparison remains unchanged.

## Runpod preparation and isolated local trial, September 7, 2026

- Added a source/fixture exporter, manifest verifier, Linux bootstrap with CUDA/EGL preflight, isolated trial restoration, a private model/API supervisor, and a resumable comparison runner. The cloud studio serves the prebuilt frontend through an SSH tunnel. Runpod provisioning and a remote job launcher in the original local UI remain separate work.
- Eleven new tests pass. They cover frozen context selection, exclusion of photo references, preserved trajectories and tasks, failed-restore rollback, refusing existing data directories, source exclusions, loopback-only connections, opt-in browser origins, and rejecting CPU fallback or missing predictions. The full backend run passed 218 checks and hit one existing eight-second Blender test deadline during concurrent CPU inference. That test passed in 0.76 seconds on an isolated retry. The frontend production build passes with its existing bundle advisory.
- The actual corrected factory comparison exported as two frozen demonstrations with unchanged states/actions, the human task and state judge, model provenance, and seeds 2000–2002. Restoration into `.data/runpod/local-validation` rebuilt XML against the installed Panda assets. The transfer archive extracted cleanly and passed its source/fixture hashes; a deliberately changed source file was rejected.
- Using explicit `--allow-cpu` for local protocol validation, the isolated sandbox completed comparison `2a5666d941aa43fb92966d777807b797` in 387.299 seconds. All nine episodes completed with exactly 150 predictions and actions each. Zero-, one-, and two-demonstration conditions each had zero passes and three failures, with no unknowns or provider errors. The audit verified model/adapter provenance, fixed weights, seeds, ordered context, and unchanged task/judge snapshots. This reproduces the prior observed outcome counts without claiming a task-success gain or bitwise-equivalent trajectories.
- The browser displayed the isolated comparison through the bundled frontend. Evidence and all episode artifacts downloaded to `.data/runpod/local-trial-results/`; the upload bundle is `.data/runpod/upload/`.

No GPU Pod was launched, no model was uploaded, and no paid compute was provisioned during this preparation. The Runpod console required user sign-in. Linux dependency installation, actual NVIDIA EGL rendering, pretrained CUDA inference, SSH transfer, and remote outcome reproduction remain unverified until the first GPU run. Stopping the model processes does not stop Pod billing; the deployment guide documents explicit shutdown.

## First Runpod GPU trial, September 7, 2026

The frozen comparison ran on an RTX 4090 in Runpod Secure Cloud, using the official `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image in EU-RO-1. NVIDIA reported 24,564 MiB VRAM and driver 580.178.04; the locked worker used PyTorch 2.8.0 with CUDA 12.8. Actual CUDA tensor execution and EGL-rendered side/wrist images at 320×180 passed before model startup. The private studio was accessed through authenticated SSH at local port 8789; only SSH was exposed on the Pod.

Two deployment issues were resolved without changing the experiment or model: GNU tar needed `--no-same-owner` for archives created on macOS, and slow package copying onto `/workspace` was avoided by installing the model environment on local container disk at `/opt/studio-icrt-venv`, linked from `integrations/icrt/.venv`. The deployment guide records both options. Checkpoints and Franka assets were downloaded from their pinned revisions and verified. Both frozen demonstration trajectory hashes matched on restoration.

Comparison `bf55493faa71417d9cb3b1ad04920f7c` completed all nine episodes: three each with zero, one, and two demonstrations, using seeds 2000–2002 and 150 control steps per episode. Every condition scored 0/3 passes, with no errors, unknowns, or pending judgments. The audit passed for all 1,350 predictions and actions, CUDA execution, unchanged model hash and adapter versions, fixed weights with zero gradient updates, task/judge snapshots, seeds, and ordered demonstration context. These are the same categorical outcomes as the corrected local CPU reference; they do not establish identical trajectories or explain the task failures. Three attempts per condition remain too few for a precise performance estimate.

The runner's measured run-and-download interval was 92.993 seconds, versus 387.299 seconds in the earlier isolated CPU trial. This includes API polling and artifact transfer and is not a controlled hardware speed benchmark. Setup time is excluded.

Evidence is saved locally under `.data/runpod/gpu-trial-1/`: the comparison export, nine full episode artifacts, passing audit, bootstrap/service logs, database snapshot, fixture receipt, and a SHA-256 download manifest. `two-demo-seed-2000.gif` re-renders the recorded two-demo episode through the original EGL camera at every fifth simulator frame; it is a sampled replay of the failed attempt, not a generated successful demonstration.

Pod `kvqg04tzkanu75` was terminated after downloading and verifying the evidence. Follow-up Runpod listings confirmed no remaining Pods or standalone network volumes. Its approximately 15-minute lifetime at the reported $0.74/hour compute rate implies roughly $0.19 compute plus disk charges; the billing-history API had not posted this session yet, so this is an estimate rather than an invoice. The lifecycle receipt is `.data/runpod/pod-trial-1.json`.

## Feedback-driven experiments, September 7, 2026

See [the full iteration report](iteration-results.md). An Astra-authored, simulator-state-assisted motion program passed the moving-conveyor development episode and 3/3 disjoint holdout seeds with native Franka contact physics. A separate Astra judge agreed with the numerical checks on one held-out rollout. This is a tool-controller result, not an ICRT success. Ten additional fixed-weight ICRT candidate tests, including three Astra-proposed revisions and a new successful demonstration, all failed without a grasp. All evidence and earlier results are preserved. Backend tests: 219 passed; new experiment-boundary tests: 3 passed; frontend build passed. The second GPU pod was terminated after evidence download.

## ICRT pose-frame investigation, September 8, 2026

Adapter `icrt-studio-v3` corrects the native fingertip/flange mismatch in observation, demonstration, and executed-action transforms. Independent MuJoCo forward-kinematics tests verify the conversion. Four additional local CPU diagnostic episodes, crossing old/corrected frames with one-step/temporal inference, all failed without a grasp. Weight loading and a context-streaming check passed. The [investigation report](icrt-diagnosis.md) separates this confirmed defect from unresolved transfer and checkpoint-provenance questions. Backend: 221 passed; runtime tests: 5 passed; experiment-boundary tests: 3 passed; frontend build passed. Earlier evaluations remain unchanged.

## Complete ICRT reference-data evaluation, September 8, 2026

The [reference-data report](icrt-reference-evaluation.md) records 18 completed offline conditions using a full 117-frame demonstration and three distinct complete queries. With fixed weights and predicted action history, mean position error improved from 3.53 cm to 0.95 cm and gripper agreement from 72.0% to 98.7%. The upstream notebook procedure also benefited from context. Three secondary tests with truncated context lost that improvement. No simulated success verdicts changed, and training overlap of the reference episodes is unknown. Eight integration tests passed. All computation ran on local CPU; no production code or frontend changed.
