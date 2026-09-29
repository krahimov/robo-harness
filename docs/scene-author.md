# Photo-guided scene author

The author turns reference photos, measurements, and an operation brief into an editable MuJoCo scene and a scripted demonstration. A robotics actor later receives that demonstration as context and runs independently. Keep the resulting authoring and actor evaluations separate.

## Connections

Both connections use exactly `gpt-6-astra` with medium reasoning effort. Each draft saves its provider, selected robot, human task version, references, measurements, and available imported assets. `STUDIO_AUTHOR_PROVIDER=auto` prefers a saved Codex ChatGPT login; `codex` or `openai` chooses an explicit default. Legacy drafts without a provider continue to use the OpenAI API.

For subscription access, install Codex CLI and run `codex login` with ChatGPT on the backend machine. The studio checks `codex login status` without opening the credential store or copying a token. Codex handles sign-in and refresh. Status is cached for ten seconds; **Check connection** forces a fresh probe. Model access is verified by the actual request, and calls count toward Codex usage.

The adapter uses the official `codex exec` runner. The inspected Python SDK's app-server transport did not expose the per-run configuration isolation used here. Each call runs in a temporary directory with `--ephemeral`, `--ignore-user-config`, `--ignore-rules`, and a read-only sandbox. Host tools, personal skills, plugins, hooks, and shell execution are disabled. API keys and studio credentials are excluded from the child environment. The normal saved login remains available to Codex itself. CLI 0.153.0 was tested; `STUDIO_CODEX_PATH` can select another installed binary.

See the official [Codex authentication](https://learn.chatgpt.com/docs/auth) and [non-interactive runner](https://learn.chatgpt.com/docs/non-interactive-mode) documentation.

The API connection calls `https://api.openai.com/v1/responses` with image input and a `simulate_scene` function tool. Credentials come from `STUDIO_AUTHOR_API_KEY`, falling back to `OPENAI_API_KEY`. Requests use `store: false` and carry prior response and encrypted reasoning items through the bounded run. No model or provider fallback occurs on failure. See [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [function calling](https://developers.openai.com/api/docs/guides/function-calling), and [image inputs](https://developers.openai.com/api/docs/guides/images-vision).

## Inputs and human decisions

The user selects a robot and task before generation. The backend enforces the robot choice when compiling model proposals. Human task requirements remain outside the author response schema, so generation cannot save revised success criteria. Changing either selection creates a scene revision; existing recordings keep their snapshots.

Included task versions cover block-to-tray, lift-and-hold, and two-box sorting. The task editor can save a new version with object IDs, required/optional outcomes, numerical regions and thresholds, and up to two teacher operations. Supported checks are object-in-region, released, settled, no off-table drops, and lifted. Teacher operations specify a destination region, the tray, or a held position. The state judge measures these requirements; the VLM judge uses the same task snapshot as its ground truth declaration.

Reference and asset limits are enforced before generation:

| Input | Current limit and interpretation |
| --- | --- |
| Photos | Up to four JPEG, PNG, or WebP files; 8 MiB and 20 megapixels each. The backend orients them, removes source metadata, and stores JPEGs at no more than 1600 pixels on the longest side. |
| Measurements | Up to eight labelled values, with units `m`, `cm`, or `mm`. These guide the author; inferred dimensions and any scale approximation still need review. |
| Imported meshes | Up to four attached geometry-only OBJ assets. Each allows 256 KiB, 400 vertices, and 800 triangulated faces, with nonzero thickness in all axes. Materials, textures, and external references are unsupported. |
| Operation brief | Up to 8000 characters. The user supplies the intended action that a still photograph cannot show. |

OBJ geometry is normalized independently on each axis to fit the scene prop's chosen XYZ half-extents. It becomes a static prop, with a convex hull for collisions. Importing a machinery mesh does not make it articulated or add robot joints. Mesh metadata is available to the author; the upload is not converted into an additional image reference.

Uploading and saving a draft writes local storage. Generation sends the selected photos, operation, task, measurements, and asset metadata through the chosen model connection. Text visible in photos and other reference data is treated as evidence, not as instructions to change the author protocol.

## Simulation and robot selection

The default `cartesian-gripper-v1` uses three slide joints and an idealized proximity weld, holding one object at a time. It can manipulate different boxes sequentially. The scene supports the primary `cube` and up to five additional movable boxes, 24 static box/cylinder/mesh props, a tray, and an approximate conveyor. It does not simulate continuous production arrivals.

`franka-panda-v1` uses the articulated Panda from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda). Download its pinned assets with `python3 integrations/icrt/prepare_franka.py` before selecting it. It moves through bounded inverse kinematics and joint actuators, with contact fingers and full end-effector rotation. Camera placement and controller settings are research assumptions, described in the [Franka integration guide](../integrations/icrt/README.md#franka-simulation).

All scene dimensions are metres. Boxes and imported props use half-extents; cylinders use `size[0]` for radius and `size[2]` for half-height. Conveyor length and width are full dimensions and its height is the top elevation. Limits come from the validated scene schema. Known factory dimensions may require a declared approximation to fit the tabletop workspace.

MuJoCo performs the dynamics. Browser playback draws the recorded transforms, while camera feedback renders recorded MuJoCo joint positions. The Panda camera render uses full robot meshes; the browser uses compact bounds for link geometry. No model-generated Python, shell, or arbitrary MJCF is executed.

## Author loop

1. Send the brief and explicit photo attachments, selected robot, human task, measurements, approved asset metadata, and parent program when revising.
2. Request a validated `SceneProgram` through the API tool or a structured Codex build reply.
3. Compile the program and execute it in MuJoCo. When `waypoints` is null, generate the selected task's teacher path. Model-specified paths allow at most 18 waypoints and 30 seconds of motion.
4. Return numerical task checks and up to three actual camera frames. The author can revise the geometry or path and retry.
5. Save the latest runnable program and recording after the model finishes. A later failed repair preserves the previous runnable candidate and discloses that failure.

Both connections allow five model responses and three simulation attempts. API calls have a 180-second timeout; cancellation takes effect between calls and simulation stages, and late results are discarded. The Codex runner limits each turn to 180 seconds and the job to 660 seconds. It bounds reply/event files and terminates the active process group on cancellation. Neither path substitutes a different model, provider, or robot.

If rendering fails, feedback records the limitation and supplies numerical checks. `camera_available` tracks visual feedback. `ready` means a runnable recording exists, not that every outcome passed. A saved demonstration remains a candidate for an independent actor evaluation.

## Draft and artifact contract

Create a draft after uploading any photos or assets; use the IDs returned by those uploads:

```json
{
  "name": "Packing station",
  "prompt": "Pick the box from the conveyor and place it in the tray.",
  "provider": "codex",
  "embodiment": "franka-panda-v1",
  "task_id": "block-to-tray",
  "image_ids": [],
  "measurements": [{"label": "Conveyor width", "value": 40, "unit": "cm"}],
  "asset_ids": []
}
```

The generated program contains `name`, `description`, `assumptions`, `scene`, and `waypoints`. The resolved recording includes the actual path and task snapshot. Task criteria and provider credentials are not writable fields in the generated program.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/author/status?refresh=true` | Refresh model connection status and read image limits |
| `POST /api/references` | Validate and normalize `{name, data_url}` |
| `POST /api/assets` | Import bounded `{name, obj}` geometry |
| `GET /api/tasks`, `POST /api/tasks` | Read human task versions or save a new version |
| `POST /api/scenes` | Save a draft, optionally linked through `parent_id` |
| `POST /api/scenes/{id}/generate`, `/cancel` | Start or cancel generation |
| `GET /api/scenes/{id}` | Read progress, result, checks, and artifact ID |
| `POST /api/scenes/template?task_id=block-to-tray&embodiment=franka-panda-v1` | Run a supplied task teacher without a model call |
| `POST /api/scenes/{id}/preview` | Validate and record edited scene data as a new revision |
| `POST /api/scenes/{id}/record` | Save the recording into the demonstration library |
| `GET /api/scenes/{id}/export` | Export program, task, MJCF, actions, measurements, and reference/asset metadata |

Four author jobs may be pending, with one running at a time. Completed programs are revised through new records. Recording a scene into the demonstration library is idempotent and preserves the author model/provider, task, source scene, and reference IDs. Existing experiments keep their snapshots.

Metadata and normalized imported geometry are stored in SQLite under `.data/`; JPEGs live in `.data/references/`. Original photo metadata and OBJ source text are not retained. Scene exports include reference metadata rather than photo bytes. Panda XML refers to installed mesh paths, so exports are not self-contained robot-asset bundles.

## Validation boundary

Controlled provider tests exercise validation, bounded retries, cancellation, task preservation, photo/mesh handling, and real MuJoCo execution. Live Codex authoring and an Astra judge using simulator camera evidence have also run locally. The API author remains protocol-tested without a live API request. [Validation results](validation.md) record current factory-reference tests, robot fixtures, outcome checks, and model runs.

The optional [ICRT worker](../integrations/icrt/README.md) runs released weights with demonstration context. The initial Cartesian and native Franka actor checks produced real predictions but no box pickup; the native scripted teacher passed its placement task. Authoring or teacher success does not establish ICL improvement. Matched-seed comparisons and actual human labels are needed to assess that separately.


## Optional Blender builder

Drafts have a `builder` field, either `mujoco` or `blender`, independent of their model connection. Legacy drafts use `mujoco`. Blender drafts use `BlenderProgram`, which extends the scene program with procedural machinery assemblies. A trusted bpy worker generates meshes, and the importer builds separate visual and collision geometry before running the existing MuJoCo teacher. Both Codex and API author loops receive the Blender layout image plus the usual numerical and camera evidence.

Successful scenes include a `blender` record with immutable build ID, design, generated asset IDs, version, geometry counts, and download URLs. `GET /api/blender/status` checks the local runtime. `POST /api/scenes/template?builder=blender&embodiment=franka-panda-v1&task_id=block-to-tray` runs the supplied example. Download paths allow only named completed-build files; partial builds and logs are not exposed. See the [Blender contract](../integrations/blender/README.md).

## Interactive workspace and portable packages

The [agent workspace](agent-workspace.md) adds conversational revisions, selected-object context, direct object editing and reusable components. Its portable ZIP export/import uses `/api/scenes/{id}/package` and `/api/scenes/import`; see the [package contract](scene-packages.md). Earlier descriptions of nonportable scene export refer to the legacy JSON `/export` endpoint.
