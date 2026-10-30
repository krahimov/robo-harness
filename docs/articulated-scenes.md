# Articulated scenes, skills, and robots

The scene language now includes an articulated scene graph, so tasks can involve parts that open, close, slide, or turn. MuJoCo simulates the dynamics. The studio compiles a validated description into MJCF itself; model-written XML, file references, plugins, and actuators are not accepted.

## Scene graph

`scene.bodies` lists root bodies (up to 12, with 24 bodies, 120 geoms, and five nesting levels in total). Each body has:

| Field | Meaning |
| --- | --- |
| `name` | Unique across the scene; also the object ID used by task checks. |
| `position`, `euler_xyz` | Pose in the parent frame (world for roots), metres and radians. |
| `joints` | Up to three `hinge`, `slide`, `ball`, or `free` joints. Hinge values are radians, slide values metres. `range`, `initial`, `damping`, `stiffness`, `spring_ref`, `friction_loss`, and `armature` are uncalibrated simulation assumptions. Only a root may use `free`. |
| `geoms` | `box`, `cylinder`, `sphere`, `capsule`, or `mesh` (an imported asset). Sizes are half-extents; cylinders and capsules use `[radius, radius, half-length]`. |
| `sites` | Named points such as handles or push points, used by skills. |
| `children` | Nested bodies that move relative to this one. |

A root body without joints is fixed to the table. MuJoCo does not filter contacts between a static root and its children, so a lid rests on its box walls.

Geoms can also be `mesh`, referencing an imported mesh asset (`asset_id`); like props, the geometry is scaled to the geom's half-extents. A scene supports 16 mesh geoms in its graph, each bounded by the mesh store's 400 vertices and 800 triangles.

`scene.equalities` adds constraints between scene-graph parts: `connect` pins an anchor (in body1's frame) to body2 or the world, `weld` fixes a relative pose, and `joint` couples two hinge or slide joints with a polynomial. `scene.contact_excludes` lists body pairs that must not collide, which closed linkages usually need. The Cartesian gripper toggles only its own grasp welds, and the Franka and arm builders carry scene constraints through their own equality and contact sections.

Recorded frames gain `bodies` (bounding-box center, half-extents, origin, quaternion, speed, and contact grasp), `joints` (type, value, velocity), and `sites` (world positions). Scenes without bodies keep the previous frame format.

`scene.include_cube` and `scene.include_tray` (both default true) control the legacy primary block and destination tray. Without them, frames omit the `cube`, `cube_speed`, and `cube_extent` keys, the replay baseline falls back to context order, and the default block-to-tray teacher is unavailable. A scene with neither and no human task reports a "No task selected" verification with status unknown and review required; an empty requirement list never counts as a pass. The articulated presets leave both out.

Root bodies appear in the Scene Builder object list and can be dragged along the table. When the saved task requires a body to stay in a region, that body is locked like the tray; create a new task version to move it.

### Cloth

`scene.cloths` (up to two) adds rectangular grid cloths simulated as MuJoCo flex: `count` vertices along X and Y (3–15 each), `spacing`, `mass`, `thickness`, edge `damping`, `friction`, optional `self_collide`, `yaw`, and up to 16 `pinned` vertex indices held in place. Vertex *i* sits at row *i* // ny, column *i* % ny. Edge lengths are constrained; bending and stretch stiffness are not calibrated. Contact uses a margin equal to the thickness so the very light vertices do not sink into surfaces.

Robots hold cloth with an **idealized pinch**: when the gripper closes with its pinch point within 2 cm of an unpinned vertex, a point constraint holds that vertex until the gripper opens. This is not finger friction; a contact pinch of cloth lying flat was not reliable with these grippers. Frames record each cloth's vertices, bounding box, speed, and `pinched_vertex`, and the browser draws the cloth as a mesh. Cloths appear in the object list and can be dragged unless the task fixes their region.

### Tendons and scene actuators

`scene.tendons` adds fixed tendons (linear couplings of hinge/slide joints) and spatial tendons (straight segments between sites), with stiffness, damping, rest length, and range. Frames record each tendon's length.

`scene.actuators` adds powered scene parts: `position` (target tracked with `kp`/`kv`), `velocity` (target speed, `kv`), or `motor` (`ctrl` × `gear` force), driving one joint or tendon. Each follows a piecewise-constant `schedule` of `[seconds, value]` pairs starting at 0 s, within `ctrl_range`. Robot controllers own their own actuators first; scene actuators come after them and are never driven by robot actions. Frames record each actuator's control and force.

## Task checks

Requirements can use `check: "joint_range"` with `joint`, `min_value`, and `max_value`. It passes when the joint's final value is inside the range. `check: "cloth_points_near"` with a cloth as `object_id`, `vertex_a`, `vertex_b`, and `max_distance` passes when those two vertices end within that distance, for example a folded corner. Existing checks (`inside_region`, `settled`, `no_drops`, `lifted`, and others) accept a scene-graph body name as `object_id`. Tasks can carry human-owned `teacher_skills` instead of the pick-and-place `operations`.

## Teacher skills

Skills name scene targets instead of coordinates:

- `move_to` moves the fingertip center to a `site`, `body`, or `point`, plus an `offset`. `via_height` first travels above the target.
- `grasp` and `release` close or open the fingers in place.
- `follow_joint` drives a hinge or slide joint to `to` by tracking a site on the moving part along that joint's arc or slide.

Targets can also be a cloth vertex: `{"cloth": "towel", "vertex": 0}`. `move_to` and `follow_joint` take a `tool`: `grasp` (default for `move_to`), `pinch`, or `push` (default for `follow_joint`). Each robot declares tool points as offsets from its grasp site; the compiler commands the grasp site so that the chosen tool point lands on the target. On two-finger grippers all three coincide. On the Allegro hand the grasp point sits in the palm for power grasps, the pinch point at the closed fingertips, and the push point on the outside of the curled fingers.

The compiler runs each skill in a private simulator before resolving the next target, then `record()` replays the resulting waypoints. Scene programs store the skill list rather than the expanded path. A program provides either `waypoints` or `skills`. A skill program allows 24 skills and 60 seconds of motion.

## Robots

`studio/robots.py` registers each embodiment with its simulator and capabilities. Scene and draft schemas accept exactly the registered IDs, and `GET /api/robots` lists them for the UI.

| Robot | Capabilities | Articulated parts |
| --- | --- | --- |
| `cartesian-gripper-v1` | `weld_grasp` | No. Its geometry has no contacts, so `follow_joint` is refused. |
| `franka-panda-v1` | `contact_fingers`, `wrist_rotation` | Yes, by fingertip contact. |
| `ur5e-2f85-v1` | `contact_fingers`, `wrist_rotation` | Yes. Universal Robots UR5e with a Robotiq 2F-85 gripper. |
| `ur5e-allegro-v1` | `contact_fingers`, `wrist_rotation`, `dexterous_hand` | Yes, with the push tool point. UR5e with a 16-joint Wonik Allegro right hand. |
| `tidybot-v1` | `contact_fingers`, `wrist_rotation`, `mobile_base` | Yes, on the near half of the table. Stanford TidyBot: holonomic base, Kinova Gen3, Robotiq 2F-85. |

Every robot also has `idealized_cloth_pinch`.

**Dexterous hand.** The action interface still carries one gripper value. For the Allegro hand it drives a whole-hand power-grasp synergy: all 16 position actuators interpolate between an open pose and a closed pose with the thumb in opposition. A grasp registers when a thumb segment and the palm or a finger segment both touch the object. The hand's servo stiffness (`kp = 6`, Menagerie uses 1) is a fixture assumption needed to lift a 60 g block; individual finger control is not exposed to actions.

**Mobile base.** TidyBot stands on a floor 0.5 m below the tabletop (a low table; the studio adds the floor and table legs) beside the table's long −Y edge, facing it. Its base slides and heading are part of the inverse kinematics, so reaching a target drives the base along the table while the arm moves. IK limits keep the base outside the table footprint. The Kinova reaches about 0.7 m at table height, so only the near half of the table (about y < −0.05 m) is reachable; beyond that the IK error stays large and is recorded rather than hidden.

The browser shows registry arms with their CAD surfaces: frames record `robot_bodies` (position and quaternion of each link that carries a visual mesh), and `GET /api/robot-visuals/arm/{embodiment}.glb` serves those meshes in link coordinates. The viewport places each mesh on its recorded link pose and hides the collision shapes it covers.

Skills and tasks check capabilities rather than embodiment IDs. Tasks can declare `required_capabilities`; the articulated presets require `contact_fingers` and run on either arm with the same skill program.

`studio/adapters/arms.py` assembles Menagerie arms generically: an `ArmConfig` lists the model parts and their attachment site, the driven joints and actuators, the fingertip site, the gripper actuator, and the left and right finger bodies used for contact grasp detection. MuJoCo's model editor (`MjSpec`) attaches the parts with name prefixes. The browser draws the arm's collision primitives; robot collision meshes are shown as their bounding boxes. Install the UR5e and 2F-85 assets with `python3 integrations/icrt/prepare_arms.py`, which uses the same pinned Menagerie commit as the Franka. Scene packages copy only the referenced meshes, part XML files, and licenses from those folders.

Adding another Menagerie arm, hand, or mobile manipulator needs a new `ArmConfig` (parts, driven joints, hand actuators with open and closed poses, grasp sides, optional grasp site, floor, base heading and IK limits) and a `RobotSpec` entry with capabilities and tool points. Legged robots and robots with other control schemes still need their own simulator class.

## MJCF import

`studio/mjcf_import.py` turns a MuJoCo MJCF document into scene-graph bodies. The document is never simulated as uploaded:

1. A whitelist accepts `mujoco`, `compiler`, `default`, `worldbody`, `body`, `frame`, `joint`, `freejoint`, `geom`, `site`, `inertial`, `asset`, `mesh`, `equality` (`connect`, `weld`, `joint`), and `contact` (`exclude`). Cameras, lights, options, and visual settings are dropped with a warning. Includes, textures, materials, height fields, actuators, sensors, tendons, contact pairs, keyframes, plugins, and DOCTYPE or entity declarations are rejected. A `file` attribute is accepted only on `<mesh>`, and only when the named OBJ or STL was uploaded alongside the XML (at most eight companion files); nothing is read from disk. Site-based equalities and equalities that start inactive are not imported. Uploads are limited to 256 KiB and 4000 elements.
2. MuJoCo compiles the remaining document in memory, which resolves defaults, classes, `childclass`, degree or radian angles, `fromto`, and orientation shorthands.
3. Compiled bodies, joints (type, axis, range, damping, stiffness, friction loss, armature, initial value), geoms (size, pose, color, friction, collision flags, and body mass split by volume), sites, equalities, and contact exclusions become studio scene data. Each mesh is moved into its body's frame and stored as a normalized mesh asset, with the geom at the mesh's bounding-box center; a preview stores nothing. A rejected import deletes any meshes it stored. Names receive the import name as a prefix. Geoms attached to the world become a fixed `<name>_static` body; frame-only bodies get a tiny contact-free geom.
4. The roots are placed at the requested table position and yaw and validated with the rest of the scene.

`POST /api/mjcf/preview` converts without saving. `POST /api/scenes/{id}/mjcf` adds the object as a new scene revision and reruns the teacher; companion meshes are sent base64-encoded in `files`. In the Scene Builder, **Add component → Import MuJoCo XML…** uses the same endpoint; select the XML together with any OBJ or STL files it names. A body named like the import keeps one copy of the name.

Tendons (fixed, or spatial through sites) and `motor`, `position`, and `velocity` actuators on joints or tendons are imported too. Each imported actuator starts with a schedule that holds its current setting. Geometry-wrapping tendons, pulleys, `general` actuators, actuators with activation dynamics, site or crank transmissions, and position/velocity actuators with a gear other than 1 are rejected. Site-based `connect` and `weld` equalities become body constraints anchored at the first site.

`examples/mjcf/cabinet.xml` is a two-compartment cabinet with a sliding drawer and a hinged door, written with default classes and degree angles.

## Angle units

MuJoCo reads joint ranges and Euler angles in degrees unless the model sets `compiler angle="radian"`. The base scene template now declares radians, so scene-graph hinge values mean the same thing on every robot. Before this, hinge ranges were correct only on the Franka, whose builder added the compiler setting.

The template's joint damping of 15 N·s/m used to be a global default intended for the gantry slides. It is now set on those three joints only. Other joints already set their own damping; a global default would, for example, freeze the per-vertex joints of a MuJoCo flex.

## Close the box

The `close-box-flaps` preset is a cardboard box with two hinged flaps that start folded open at about 2.1 rad. It requires contact fingers and runs on the Franka or the UR5e. The teacher closes its fingers, approaches each flap from outside, and pushes it shut along its hinge arc. The requirements are both hinges within 0.2 rad of closed, and the box body staying within 3 cm of its footprint. Flaps are rigid panels on damped hinges; cardboard bending and creases are not simulated. An unrelated motion fails both flap checks.

## Open the drawer

The `open-drawer` preset imports `examples/mjcf/cabinet.xml`. The teacher lowers closed fingers into the gap behind the drawer handle and pulls along the slide to 0.12 m. Requirements: the drawer is at least 0.09 m open, the upper door stays within 0.15 rad of closed, and the cabinet stays in place. It passes on the Franka and the UR5e; a motion that does not pull fails the drawer check.

## Fold the towel

The `fold-towel` preset lays a 9 × 9 vertex, 16 cm towel on a 2 cm folding board, which lets every robot's fingertips reach the corners above the shared z = 0.025 m action limit. The teacher pinches corner 0, lifts it, carries it over corner 8 on the same edge, and lets go 3 cm above the board. Lowering further pressed the cloth between the Allegro fingers and the board and destabilized it. Requirements: the two corners end within 5 cm, the towel is released, and it stays in the work area. It passes on the Cartesian gripper, Franka, UR5e + 2F-85, and UR5e + Allegro; a motion that does not fold fails the corner check. The towel straddles the table's center line, beyond TidyBot's reach.

## Not yet supported

- Deformable solids other than cloth (sponges, dough, cables as flex), liquids, cutting, and tearing.
- Contact-based cloth grasping; the cloth pinch is idealized.
- Per-finger control of the hand from robot actions; the hand follows one grasp synergy.
- Legged robots, and mobile robots that drive around the table; TidyBot drives along one side.
- Authors still decline requests that need these.
