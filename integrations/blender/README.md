# Blender scene authoring

The studio uses the official `bpy` 5.2.1 Python package to generate machinery meshes and render an editable Blender scene. MuJoCo owns robot control, contacts, task checks, and the recorded demonstration. The author remains GPT-6 Astra through the selected Codex subscription or OpenAI API connection.

## Setup

From the repository root:

```sh
uv sync --project integrations/blender
```

This installs a separate Python 3.13 environment. A persistent Blender server and the Blender desktop app are not required. The studio starts a short-lived worker for each build and preview. For another compatible environment, set `STUDIO_BLENDER_PYTHON` to its Python executable before starting the backend.

On macOS, run the backend from a normal local terminal with access to native graphics services. A restricted sandbox can crash the native bpy import even though the package is installed. The status endpoint probes the import and reports readiness. CPU Cycles renders were verified on Apple M4 Pro. Other platforms have not been exercised locally.

In Scene builder, choose **Blender geometry + MuJoCo physics**. **Try Blender conveyor** runs a procedural example with the native Franka robot and no AI call. Franka assets must be installed as described in the [ICRT integration](../icrt/README.md). To generate from a photo, attach references, describe the operation, select the task and robot, and choose **Generate with Astra**.

## Authoring contract

Astra returns a bounded declarative program. The repository's trusted Python worker translates that program into `bpy` operations. Model-generated Python, file paths, imports, expressions, and external asset URLs are not executed.

The program supports up to four machinery assemblies with one material each. An assembly combines up to 16 beveled boxes or cylinders with independent positions and rotations. Coordinates are world metres, Z up; sizes are half-extents. Generated meshes have at most 1,200 vertices and 2,400 triangles each. Existing user OBJ upload limits stay unchanged.

Each assembly becomes a visual-only mesh in MuJoCo. Each part marked collidable separately exports a conservative axis-aligned box. This preserves gaps between separate bin walls instead of filling the whole bin with one convex hull. Rotated parts can have oversized collision bounds. There are at most 24 combined visual assemblies, collision parts, and existing scene props. Mass and friction come from the selected MuJoCo fixture and movable-object configuration; photographs do not determine them.

Blender's layout preview includes the actual robot mesh and transforms from the initial MuJoCo state. The preview is static. The browser's recorded MuJoCo rollout supplies the movement, state trace, and demonstration actions. Blender materials affect the layout render; MuJoCo currently retains assembly colors and geometry, not the full Blender shader graph. A generated scene is an approximation, not a calibrated factory reconstruction.

The existing robot, tray, box, and belt remain separate from generated machinery. A `scene.conveyor` still uses the existing approximate drive model. Static decorative rollers do not simulate a roller mechanism. Generated machinery is not automatically articulated.

## Revision and evidence

Each successful candidate saves its design, mesh exports, collision manifest, Blender file, layout render, MuJoCo XML, task, actions, and joint-state trace in `.data/blender/<build_id>/`. The final authored scene records its build ID and generated asset IDs. The agent receives both the Blender layout and MuJoCo camera/state feedback, and can revise a failed candidate within the existing three-build limit.

The UI provides **Blender file** and **Scene package** downloads. Packages include the geometry, `.blend`, preview, design, manifest, MuJoCo XML, and simulation evidence. Reference photos and provider credentials are excluded. Panda geometry embedded in the `.blend` retains the Menagerie license as a Blender text block and a separate package file. MuJoCo XML still references the separately installed Panda assets by absolute path; the ZIP is not a turnkey portable runtime.

Saving a demonstration preserves the Blender build metadata in its source rollout. New Astra revisions preserve the selected builder. Editing the resolved MuJoCo scene specification creates a MuJoCo revision and does not claim to update the previous `.blend` or layout image.

Failed or cancelled compilation removes partial build files and staged mesh records. Candidate downloads become available only after compilation and rendering finish. Existing recordings and previous scene versions remain unchanged.

## Checks

```sh
cd backend
uv run pytest -q tests/test_blender.py
```

These tests exercise import contracts, visual/collision separation, material/mesh conversion, cancellation, cleanup, provider feedback, draft selection, and file access. They use a fake Blender transport with real MuJoCo execution and do not download bpy. The live Blender and photo-guided checks are recorded in [validation](../../docs/validation.md).

Blender and its Python package retain their upstream GPL terms. The studio's MIT license does not replace Blender's license or the licenses of imported robot assets. See the [Blender Python module documentation](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html).
