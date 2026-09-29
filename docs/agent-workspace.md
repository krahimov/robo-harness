# Conversational scene workspace

Scene builder opens a conversation next to the simulator. On smaller windows, the **Agent** and **Scene** tabs switch between the two panes. The prompt composer supports attached photos, camera capture, a selected-object reference, Enter to send, and Shift+Enter for a new line. Its model control opens the author connection settings; GPT-6 Astra remains the configured author.

## Describe and revise

Start with a photo and a description, or open a task example in settings without a model call. Settings contain the selected robot, Blender/MuJoCo builder, task requirements, known measurements, and imported mesh references. Each sent request creates a scene revision. The conversation shows saved requests, actual build activity, completion summaries, errors, and links to previous revisions. Scene history lets you return to an earlier revision and branch from it.

The author receives the previous program, the selected object's concrete description, and a bounded set of ancestor requests and results. This is authoring conversation context. Automatic retrieval of demonstrations into the robotics actor's context is still a separate future feature.

Unsent composer text is saved locally per revision and survives reloading. Direct edits preserve a pending message. Selecting a historical scene restores its own draft. Reference photos remain attached until removed; sending shares those references through the selected author connection.

## Move and inspect

Choose **Edit scene** to inspect the initial simulation state. Select an object in the canvas or the object list. The inspector exposes supported position axes and full dimensions in metres. Drag a colored translation arrow or enter coordinates and choose **Apply & simulate**. Escape cancels a drag before submission. Orbit controls pause during a move.

A move or resize updates the source scene program, creates a new revision, and reruns the teacher. Blender assemblies update all their source parts, then regenerate visual meshes and separate collision shapes. Rotated assemblies support uniform resizing; cylindrical parts keep circular radial dimensions. Task objects and fixed robot parts expose only supported edits. The inspector reports these constraints.

Saved task requirements remain unchanged. Moving a destination tray could silently move its success region, so a saved task locks that tray. Create a new task version explicitly when changing the intended destination. Conversational candidates and raw specification revisions are subject to the same task-preservation checks. A geometrically valid revision can still fail its original task.

Choose **Ask agent about this** to refer to the selected object in the next request. This preserves existing composer text. Component insertion supports a workbench, rear guard, open bin and control cabinet. They are bounded static assemblies, not articulated mechanisms or accurate machinery reconstructions.

## Review the evidence

**Simulator** shows the initial scene and recorded motion. Compatible Franka recordings display the original robot CAD through a cached visual asset, without enlarging every recorded frame or changing collision physics. **Blender render** shows a static rendered layout when that builder was used. **Details** contains the numerical task checks, assumptions, and advanced scene JSON editor. Direct object controls preserve the Blender source; editing the resolved JSON produces a MuJoCo revision.

**Save demonstration** stores the recorded teacher operation and opens Workspace for a separate robot-model experiment. A passing teacher or replay is not evidence of in-context learning.

The download control exports a [portable scene package](scene-packages.md). Import rebuilds the editable MuJoCo scene with fresh local asset identifiers and a fresh recording. Blender design and `.blend` files are preserved as source attachments; arbitrary desktop Blender edits are not automatically converted back to robot physics. Export includes reference photos by default; API callers may set `include_references=false` to omit them.

## Validation

The [scene edit benchmark](scene-edit-benchmark.md) exercises varied structured layouts, all component types, source assembly edits, and an intentionally obstructed task. It makes no AI calls and does not measure general photo-authoring quality. The live conversational and browser checks are recorded in [validation](validation.md).
