# Portable scene packages

A scene package preserves the editable simulation, mesh geometry, human task, known measurements, source photos and optional Blender source files. Import records a fresh MuJoCo rollout and measures the task again. It does not call an AI model or run code from the archive.

## Export and import

`GET /api/scenes/{id}/package` downloads a version 1 ZIP for a ready scene. Photos are included by default. Add `?include_references=false` to omit them. Only normalized photos are exported; original EXIF and GPS metadata were removed at upload.

`POST /api/scenes/import` accepts JSON:

```json
{
  "name": "Revised loading station",
  "data_base64": "UEsDB..."
}
```

`name` is optional. The response is a new ready scene with `source: "imported"` and `builder: "mujoco"`. Its mesh and photo IDs are new local IDs. Known measurements retain their labels, numerical values and units. Older version 1 packages without measurements import with an empty measurement list. The import preserves the task's declared requirements and binds tray-relative regions to the imported layout. Human tasks using schema version 2 receive a separate imported task record with provenance; existing tasks and evaluations remain unchanged.

The importer uses the studio's installed simulator and supplied robot runtime. Franka scenes require the normal Franka assets installation and return HTTP 409 if it is missing. The importer never installs an arbitrary robot model from an archive. Separately, the exported `scene.xml` is portable: it references bundled robot meshes through relative paths and can be loaded directly by MuJoCo after extracting the ZIP.

Invalid packages return HTTP 422 without echoing archive contents or local file paths. Requests above the encoded request limit return HTTP 413.

## Archive contents

| File | Purpose |
| --- | --- |
| `package.json` | Format `robot-harness-scene-package`, version `1`, resource manifest, known measurements and source scene identity. |
| `program.json` | Editable, validated `SceneProgram` with the resolved layout and teacher waypoints. |
| `task.json` | Human task snapshot used when recording and measuring the new rollout. |
| `assets/<id>.obj` | Geometry-only, normalized meshes. Props in `program.json` set their position and half-extents. |
| `references/<id>.jpg` | Optional normalized source photos. |
| `scene.xml` | Snapshot of the compiled MuJoCo scene with relative robot asset paths. |
| `robot/assets/*` | Native robot meshes referenced by the XML, when needed. |
| `robot/panda.xml`, `robot/LICENSE`, `robot/SOURCE.txt` | Supplied robot source, upstream license and pinned source commit. |
| `source/blender-program.json` | Original declarative Blender design when available. |
| `source/scene.blend`, `source/preview.png` | Optional archived editable Blender scene and its original layout preview. |
| `source/ROBOT_ASSETS_LICENSE.txt` | Native robot license accompanying the Blender source, when present. |
| `README.txt` | Editing and import instructions. |

Packages contain no provider credentials, model session logs or uploaded executable scripts. The normalized OBJ meshes contain geometry only, without material or texture references. Original upload text is not retained by the existing asset importer; the package exports the stored geometry that the simulator actually uses.

## Editing a package

1. Extract the ZIP into a directory.
2. Edit `program.json`, or edit one of its normalized OBJ meshes. Keep every manifest resource and its relative path. Blender-generated visual meshes and collision boxes are separate props; adjust their collision boxes as well when changing physical dimensions.
3. Set `waypoints` to `null` if a changed layout needs the runtime to construct a new teacher path. Existing explicit waypoints otherwise remain part of the program.
4. ZIP the directory's contents with `package.json` at the archive root, then import it.
5. Inspect the new rollout and measured task checks before saving a demonstration.

Changing `program.json` does not update the archived XML or Blender file in the ZIP. Import rebuilds the current simulation from the program and mesh assets. The imported scene's next package export contains its newly compiled XML.

Blender files and the original Blender design remain source attachments. They are not executed or parsed to derive physics. Editing a `.blend` file alone therefore does not update the imported simulation. The imported scene does not display the archived Blender preview as current output. `package_source.blender_program` preserves the source design with remapped asset IDs, and `package_source.attachments` contains download URLs for the archived source files. Direct arbitrary `.blend` round-trip editing is not supported.

## Bounds and validation

The importer accepts at most 64 MiB of compressed ZIP data, 96 MiB expanded, 128 archive entries and 40 MiB per entry. Base64 data is limited to 89,478,488 characters. Known measurements allow up to eight entries, each with a 1–120 character label, a positive value no greater than 100,000 and a unit of `m`, `cm` or `mm`. Individual JSON files are limited to 2 MiB; photos to 8 MiB before normalization; OBJ meshes to 512 KiB, 1,200 vertices and 2,400 triangles. Existing scene limits still apply, including four mesh props and 24 total props.

The manifest must exactly describe the archive's files. Normal directory entries are allowed only for declared resources. Duplicate names, path traversal, absolute paths, symlinks, encryption, unsupported compression, unexpected files and executable scene-language fields are rejected. Files are decoded in memory within those bounds, not extracted to user-provided paths. Assets remain hidden while the simulation compiles; a failed import removes its new assets, photos, artifact and source attachments. Successful scene, reference and task records are published together in one database transaction.

## Verification

`backend/tests/test_scene_packages.py` covers import into an empty data directory, mesh and reference ID remapping, edited layout replay, task rebinding, native XML loading from an unrelated directory, source-only Blender preservation, rollback, request errors and hostile archive shapes. Native XML verification uses the installed pinned MuJoCo Menagerie model and skips only when those optional assets are absent.

The live factory-photo scene also passed a package round trip on September 6, 2026. Its 10,537,309-byte package contained 82 files, two mesh assets, one source photo and the Blender source. Import into a separate empty data directory recreated the native Franka rollout and passed all four task requirements. The evidence record is `.data/validation/portable-package-live.json`; the source package is `.data/validation/blender-photo-portable.zip`.
