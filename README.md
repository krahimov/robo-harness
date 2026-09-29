# Robot Harness Studio

A local studio for building robot demonstrations, supplying them as model context, and measuring the resulting behavior with configurable judges and human review. The focus is inference with fixed model weights.

The scene author, robot actor, and judge have separate jobs:

| Role | Available implementation |
| --- | --- |
| Scene author | GPT-6 Astra takes photos, measurements, and instructions; proposes a scene; runs MuJoCo; inspects feedback; and revises it. |
| Robot actor | Astra generates motion programs with bounded revisions and frozen holdouts. The optional pretrained ICRT worker predicts actions from demonstration context and camera/proprioception observations. Replay and external HTTP adapters are also available. |
| Judge | Local numerical task checks, GPT-6 Astra through a saved Codex login, or a configurable VLM endpoint, followed by per-outcome human review. |

Live Astra authoring and judging, pretrained ICRT inference, and a contact-based Franka teacher have been exercised locally. After correcting two action/timing integration defects, the matched ICRT comparison still scored 0/3 successful attempts with each of zero, one, and two demonstrations. The generated teachers passed; learned task execution remains unresolved. See [validation results](docs/validation.md) and the [ICRT integration report](integrations/icrt/README.md) for the exact tests and limits. A later [feedback-driven experiment](docs/iteration-results.md) produced an Astra-authored, simulator-state-assisted conveyor controller that passed 3/3 holdout seeds. Ten additional ICRT candidates still failed; the actors and observation contracts are reported separately. A subsequent [ICRT investigation](docs/icrt-diagnosis.md) fixed a native flange/fingertip frame mismatch in adapter v3; corrected retests still failed, so the remaining failure cause is unresolved.

A [complete reference-data evaluation](docs/icrt-reference-evaluation.md) now demonstrates a context benefit with the released ICRT checkpoint. On three separate recorded query episodes, one complete demonstration reduced mean position error from 3.53 cm to 0.95 cm and raised gripper agreement from 72.0% to 98.7%. These are offline predictions, not successful simulated or physical robot executions.

## Run locally

Requires Python 3.12 or 3.13, Node.js 22+, and [uv](https://docs.astral.sh/uv/).

```sh
python3 scripts/dev.py
```

Open [the studio](http://127.0.0.1:5178) or the backend's [API documentation](http://127.0.0.1:8788/docs). The launcher installs locked dependencies and starts both processes. Ctrl+C stops them. Templates, manual editing, trajectory replay, and state checks need no model account.

Export optional settings from [.env.example](.env.example) before launching. The application does not automatically load `.env` files. The server binds to localhost and is intended for one user's development machine; multi-user authentication and production hosting are not included.

For separate terminals, run `uv sync --frozen` and `uv run uvicorn studio.main:app --host 127.0.0.1 --port 8788` from `backend/`, then `npm ci` and `npm run dev` from `web/`.

## Build a demonstration

1. Open **Scene builder**. The agent conversation sits beside the simulator; narrow windows use **Agent** and **Scene** tabs.
2. Attach photos to the composer and describe the operation. **Scene settings** selects the robot, task, builder, known measurements, mesh assets, and Codex subscription or API connection. Examples work without a model call.
3. Send the request with Enter. Astra builds, simulates, inspects feedback, and records the result. Follow-up messages preserve the scene and conversation context.
4. Select an object in **Edit scene**. Drag its permitted axes, edit its position or dimensions, or choose **Ask agent about this**. Each applied edit creates a revision and reruns the teacher. Add reusable machinery from the object inspector's **+** control.
5. Inspect the recording, **Details** task checks, and optional **Blender render**. Scene history opens previous revisions. **Save demonstration** sends the recorded operation to Workspace for an independent actor experiment.

The [workspace guide](docs/agent-workspace.md) explains selection, revisions, saved drafts, editing limits, and task preservation.

Both author connections use exactly `gpt-6-astra`. The default prefers a saved Codex ChatGPT login; each draft preserves its chosen connection. The studio asks Codex to use its own saved login and never reads or copies its token. Codex calls consume subscription usage; API calls use the API project. Uploading a photo stores it locally; generation sends the selected references and brief through the chosen connection.

Task requirements are human-owned versions. **Edit as new version** changes outcome definitions and teacher operations without rewriting previous recordings. The author can arrange the scene and motion but cannot save changes to the task. The judge rubric describes how to inspect those requirements. Demonstration exports retain their task snapshot, and legacy imports remain supported.

The scene language supports the primary box plus up to five additional movable boxes or cylinders, 24 static props, and an approximate conveyor. Supplied teacher operations support up to two sequential transfers or a lift-and-hold example. Photo input allows four images, each at most 8 MiB and 20 megapixels. OBJ input allows 256 KiB, 400 vertices, and 800 triangles, without textures or external files; meshes become static geometry with convex-hull collision. Imported mesh dimensions and photograph scale remain editable assumptions. See [scene-author contracts](docs/scene-author.md).

## Generate machinery with Blender

Choose **Blender geometry + MuJoCo physics** in Scene builder. Astra can build beveled machinery assemblies, inspect a Blender layout render, and test the robot in MuJoCo before saving a demonstration. Visual meshes and separate collision shapes share the same world coordinates. The layout preview is static; the recorded MuJoCo rollout shows the actual motion.

Install the optional worker from the repository root:

```sh
uv sync --project integrations/blender
```

**Try Blender conveyor** runs without an AI call. Sending a request uses the selected photo references and saved Codex login or API connection. Download the editable `.blend` file or a scene package from the preview. The [Blender guide](integrations/blender/README.md) covers setup, geometry limits, collision approximations, artifacts, and licenses.

## Choose the robot and actor

For remote GPU experiments, use the [Runpod sandbox guide](integrations/runpod/README.md). It packages a frozen local ICRT comparison, restores it into a separate cloud data store, and downloads audited results through a private SSH tunnel. The first RTX 4090 run completed all nine episodes with a passing infrastructure audit and the same 0/3 task success in each context condition as the local reference. Cloud execution is separate from local authoring.

The default Cartesian gripper uses three position-controlled slide joints and an idealized proximity grasp. Franka Panda uses an articulated [MuJoCo Menagerie model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), joint dynamics, inverse kinematics, and finger contacts, with no proximity weld. Install its pinned assets before selecting it:

```sh
python3 integrations/icrt/prepare_franka.py
```

The optional [ICRT integration](integrations/icrt/README.md) downloads the released [source](https://github.com/Max-Fu/icrt) and [checkpoints](https://huggingface.co/mlfu7/ICRT) into `.data/models/` and starts a separate inference worker:

```sh
uv sync --project integrations/icrt
uv run --project integrations/icrt python integrations/icrt/prepare.py
uv run --project integrations/icrt python integrations/icrt/worker.py --port 8790
```

The two weight files total about 835 MiB, with additional space needed for dependencies and robot assets. CPU inference has run on an Apple M4 Pro; CUDA needs a compatible PyTorch installation. The worker does not support MPS. The integration guide documents camera order, action units, context limits, source patches, licenses, and setup.

ICRT receives side/wrist images and robot proprioception, not task text or judge decisions. Franka executes its position, rotation, and gripper predictions. Cartesian research mode projects away rotation and records that limitation. Demonstrations must match the selected embodiment. No fine-tuning occurs, and the adapter does not substitute replay when inference fails.

The replay baseline selects the closest demonstrated starting position and replays absolute actions. It does not perform ICL. Use the [HTTP adapter contract](docs/adapter-protocol.md) for another actor. GEN-1.5 and S1 adapters are not included.

## Run Astra experiments

Choose **GPT-6 Astra · motion programs** in Workspace. Set a development proposal budget and final evaluation count, then run with a selected state or VLM judge. Astra receives initial cameras, simulator state, and sampled demonstration state/action pairs. It can revise its motion program between attempts. The first program passing both state and judge checks is frozen for fresh holdout scenes. Proposals, failed recordings, and original verdicts remain available for review.

The [Astra experiment guide](docs/astra-experiments.md) covers controls, observation limits, cancellation, and evidence. This actor currently requires Franka and does not replan from images during each episode. Automatic demonstration/scene revisions are not included in its loop.

## Compare demonstration context

**Compare context** runs conditions with the same robot, task, instructions, judge version, starting seeds, and control horizon. Only the ordered demonstration context changes. Compare one versus several demonstrations, or include a zero-demo condition with an inference actor. Replay needs at least one demonstration. ICRT has no language conditioning, so its zero-demo condition has no task instruction.

Evaluation seeds must differ from selected demonstration seeds. This avoids an exact seed overlap; it does not prove independence from model pretraining. All conditions are validated and their linked experiment records are saved together before scheduling.

Results show pass/fail/unknown/error outcomes, Wilson intervals on resolved judgments, and differences on matched resolved seed pairs. Unknowns and errors are excluded from success-rate denominators and remain visible. Later rejudging preserves the comparison's original evaluation IDs. Export the comparison to retain its configuration and evidence links. Control time depends on the selected robot's frequency: Cartesian runs at 20 Hz and Franka at 15 Hz.

## Configure and calibrate judges

Create a VLM judge with a versioned Markdown rubric and enabled `inspect_state` and/or `replay_frames` tools. Choose **Codex subscription** with `gpt-6-astra`, or configure `STUDIO_JUDGE_ENDPOINT` and optionally `STUDIO_JUDGE_TOKEN` for a chat-completions-compatible VLM. The local state baseline measures task checks directly and does not interpret rubric prose.

State inspection supplies sampled frames, explicit coverage, and min-height/max-speed statistics over every recorded frame in the requested interval. Camera replay supplies actual MuJoCo renders. A verdict without retrieved evidence becomes unknown. Each evaluation retains its judge version; provider failures preserve the recording for review.

The Judges view contains an outcome-labeling queue. It prioritizes uncertainty and reserves a quarter of available slots for passes when they exist, to catch false approvals. Label each outcome and describe corrections. Agreement and false-pass counts use the latest resolved human labels for that exact judge version; human unknowns are excluded. This deliberately sampled agreement is not a population success estimate.

A judge can propose a revised rubric from corrections. The proposal stays a draft until accepted as a new version. Acceptance does not change task requirements, old verdicts, or labels, and the new version needs fresh evaluation. Frozen calibration samples reserve development and holdout cases before labeling, and the review form hides judge verdicts by default. The prepared sample has no human labels yet, so measured human agreement remains unavailable.

Episode acceptance and demonstration promotion remain separate from outcome calibration. Promoted recordings are candidates for a later context experiment. Judge decisions are retrospective, not live motion supervision.

## Fidelity and saved evidence

The [pen-into-pouch workflow](docs/pen-pouch-workflow.md) adds a native Franka contact task with a cylindrical pen and held-open rigid pouch. Astra's state-assisted motion program passed three small position variations; the guide records the evidence, limitations, and reproduction command.

The browser replays recorded MuJoCo geometry; camera tools render recorded joint states. Franka camera renders and compatible browser recordings use the original robot meshes. The browser downloads a cached visual asset once and animates it from recorded link transforms; unavailable or incompatible assets fall back to compact link bounds. Panda camera/controller settings and the conveyor are research approximations, not calibrated factory or DROID replicas. Simulation success does not establish hardware performance.

Metadata is stored in `.data/studio.sqlite`; recordings, references, imported meshes, and downloaded model assets remain under `.data/`, outside Git. Scene, task, judge, and experiment versions preserve their source configuration. Interrupted jobs retain completed evidence. Video-file authoring, automatic demonstration retrieval, gradient updates, and physical robot execution are not included.

The workspace exports [portable scene packages](docs/scene-packages.md) with editable programs, meshes, robot assets and licenses, and optional photos. Package import recompiles the resolved scene using the installed canonical robot runtime. Blender files remain source attachments. The older JSON `/export` endpoint retains path-based asset references and is not portable. Reproducibility assumes the same dependencies and platform.

## Verify and contribute

```sh
cd backend
uv run --frozen pytest -q
```

```sh
cd web
npm ci
npm run build
```

With the local server running, `python3 scripts/smoke.py` records two demonstrations, runs a matched-seed replay comparison, and writes an evidence bundle to `.data/smoke/`. It creates local records and makes no model calls by default. Its optional `--author-photo /path/to/photo.jpg` performs live Astra authoring using the selected provider and consumes its usage.

[CI](.github/workflows/ci.yml) is prepared for backend tests on Python 3.12/3.13 and a Node.js 24 frontend build. It requires no credentials and does not download model weights. Native robot tests skip when separately installed assets are absent. The optional ICRT runtime regression suite has a separate command in its integration guide; it does not load checkpoint weights. GitHub-hosted CI has not been run as part of this release preparation; [validation](docs/validation.md) records the local checks.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [AGENTS.md](AGENTS.md) for repository rules. Studio code uses the [MIT license](LICENSE). Downloaded source, weights, and robot assets retain their own terms; the [ICRT licensing notes](integrations/icrt/README.md#checkpoints-source-changes-and-licenses) document the upstream notices and unresolved model-license metadata.

The latest [recovery, stress, calibration, and ICRT verification](docs/closed-loop-and-stress.md) records a paired gripper-fault recovery, eight passing predeclared stress cases, an unlabelled human-review packet, and the remaining ICRT manipulation failure.


The latest [ICRT camera and action-history investigation](docs/icrt-robotiq-and-feedback.md) fixes wrist-camera framing/occlusion and feeds executed commands back into model history. Contact-based Robotiq teachers pass, but the new ICRT pick/place and pen trials still fail. The report includes preserved traces, reproduction commands, and next diagnostic steps.
