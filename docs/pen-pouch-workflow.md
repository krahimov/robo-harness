# Pen into an open pouch

The studio now includes `pen-into-open-pouch`: a Franka Panda grasps a horizontal pen, lifts it, turns it 90 degrees, lowers it into a held-open pouch, releases it, and withdraws. Open **Pen into an open pouch** in scene history or **Astra / pen into an open pouch** in demonstrations. Selecting the task preset also selects MuJoCo and the Franka embodiment.

## Recorded result

GPT-6 Astra authored a bounded motion program using initial camera images and explicit simulator state. The first proposal on the final fixture passed the development scene (seed 2000). The frozen program then passed all six task checks on seeds 3000, 3001, and 3002. These vary the pen's initial X/Y position by at most 4 mm per axis; they do not establish broad generalization. Each recording contains 384 frames over 25.53 seconds.

A separate Astra judge inspected state and rendered camera evidence for holdout 3000 and passed all six checks. Human review remains pending. No gradient updates were performed. This is a state-assisted LLM tool controller: it chooses relative XYZ/yaw motions, and the executor resolves their targets from live state at each motion start. It is not vision-only control, a dedicated robotics ICL policy, or a real-robot result. ICRT was not evaluated on this new task.

The deterministic teacher is separate. Its final recipe passed seeds 42 and 3000–3002. The saved Astra demonstration contains Astra's program, not that teacher trajectory.

## Fixture and fixed criteria

The pen is a 14 cm long, 16 mm diameter, 18 g cylinder on raised anti-roll supports. It initially lies along Y. The pouch interior is X `[0.005, 0.195]`, Y `[0.04, 0.16]`, Z `[0.012, 0.11]` metres. The pen cannot fit across the 12 cm Y opening, so rotation is necessary.

| Required outcome | State measurement |
| --- | --- |
| Lifted | Pen held above 0.14 m at least once |
| Inserted while held | Entire pen bounds inside the pouch while the grasp is active |
| Inside at completion | Entire final pen bounds inside the pouch |
| Released | No active grasp at completion |
| Settled | Speed below 0.03 m/s throughout the final 0.5 seconds |
| No off-table drop | Pen center never below -0.02 m |

These checks do not prove every possible sequencing or damage constraint. In particular, “no off-table drop” does not prohibit every slip or intentional release. The saved trajectory supplies the evidence for the observed lift, rotation, insertion, release, and withdrawal order.

The pouch has rigid collision walls and decorative handles. It is a proxy for a bag held open, with no fabric deformation, zipper, or bag-opening operation. Contacts are uncalibrated simulation parameters; the pen is relatively thick and the supports simplify pickup.

## Changes needed to make the task work

- Added cylindrical movable objects, initial orientation, and per-object position variation. Extent checks account for a cylinder's world orientation.
- Fixed waypoint orientation interpolation: equivalent Euler angles near negative/positive pi previously caused an unintended full wrist turn. Quaternion interpolation now follows the shortest rotation.
- Added an opt-in precision contact profile: elliptic friction cone, `impratio=100`, Newton solver, tolerance `1e-10`, five no-slip iterations, and stiffer contact parameters. The pen fixture also uses six-dimensional fingertip contacts. Existing scene defaults remain unchanged.
- Raised the pouch floor surface to 14 mm, 2 mm inside the declared containment region, so resting contact penetration does not straddle its lower boundary. Task bounds and thresholds were kept fixed throughout.

These contact choices follow the mechanisms described in [MuJoCo's preventing-slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip). They make this fixture numerically usable; they have not been calibrated against hardware. Grasping uses finger/object contact, with no weld attachment.

Earlier development recordings include failed grasps, slips, and floor penetration. They remain in `.data/validation/pen-pouch/`; `astra-1` contains four failed model proposals on an earlier fixture. `astra-2` contains the successful final-fixture proposal and frozen holdouts. The reported 3/3 refers only to the final holdout set, not all development attempts.

## Evidence and reproduction

Local evidence is under `.data/validation/pen-pouch/`:

- `astra-2/proposal-1.json`, `frozen-program.json`: actual model output and program.
- `astra-2/rollout-1.json`, `holdout-3000.json` through `holdout-3002.json`: complete simulation recordings.
- `astra-2/report.json`, `independent-judge.json`: numerical results and independent judge evidence trace.
- `teacher-repeatability.json`: separate scripted teacher results.
- `pen-pouch-astra.gif`: replay of holdout 3000 at twice recorded speed.
- `evidence-manifest.json`: hashes of saved JSON evidence.

To generate a new Astra trial, with the project's dependencies, Franka assets, and Codex login installed:

```sh
cd backend
uv run python ../integrations/workflows/pen_actor.py --output ../.data/validation/pen-pouch/new-trial
```

The output directory must not exist. This invokes the model and consumes subscription usage. It allows up to four development proposals, then tests a frozen passing program on the three fixed holdout seeds. Reusing those seeds for further tuning makes them development cases; use fresh holdouts for a new generalization claim.

Validation on September 8, 2026: **226 backend tests passed**, including the native contact rollout; frontend production build passed. Existing dependency deprecation and bundle-size warnings remain.

## Next task stages

First expand pen position, angle, and diameter variation with fresh holdouts. Then add obstructing objects and a recovery attempt after a failed grasp. Opening a flexible bag, keeping it open with a second arm, and closing its zipper require additional deformable-object and manipulation work; they are not implemented by this fixture.
