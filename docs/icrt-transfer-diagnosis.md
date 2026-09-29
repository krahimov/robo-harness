# ICRT simulation transfer diagnosis

September 8, 2026. ICRT still does not complete the simulated pen task. The new experiments locate the failure in action prediction on the synthetic observations, before grasp contact. They do not identify one isolated visual or task feature as the sole cause.

## What fails first

In the existing 415-step, one-demonstration run, the minimum fingertip-to-pen-center distance was 24.16 cm. The zero-context condition reached 23.22 cm. Neither grasped the pen. The arm does not reach a plausible grasp pose, so adjusting pen friction or the final insertion criteria cannot address this observed failure.

A passing simulator demonstration can be replayed with native contacts, and the same loaded ICRT runtime predicts actions well on the released real recordings. Those controls separate a functioning simulator and checkpoint from useful transfer between them.

## Recorded observations, without new physics

The complete saved pen demonstration contains 383 actions. The first probe sampled 120 observation/action pairs across its entire duration, including its ending. The query uses 120 pairs from a separate saved Astra program. During prediction the query observations stay fixed; no predicted action changes the next observation. This removes execution dynamics as the source of these prediction errors.

| Query and history | Mean position error | Gripper agreement |
| --- | ---: | ---: |
| Same demonstration as context and query, predicted history | 2.36 cm | 54.2% |
| Same demonstration, correct past actions | 1.56 cm | 63.3% |
| Separate simulator episode, predicted history | 2.30 cm | 25.0% |
| Separate simulator episode, correct past actions | 1.61 cm | 62.5% |

The same-episode conditions deliberately include the query in context. They are diagnostic controls, not generalization tests. Recorded-history conditions supply only an already elapsed action, never the current target before predicting it.

Holding the current pose gives 0.60 cm error on the separate simulator query, compared with the model's 2.30 cm. Its 99.2% gripper agreement reflects long intervals with no gripper change and does not imply task competence. Simulator demonstrations use small command increments, averaging roughly 0.6–0.7 cm, while the selected real reference has about 3.8 cm command displacement. The policy must reproduce the appropriate action distribution, not merely generate plausible motion.

## Coordinate frame intervention

A single rigid transform was chosen from the context's initial pose to align it with the real reference's initial pose. Both context and query observations/actions receive that same transform. Images remain fixed. The transform preserves local action deltas and uses no query target to choose it.

| Condition | Mean position error | Gripper agreement |
| --- | ---: | ---: |
| Original simulator query | 2.3020 cm | 25.0% |
| Shifted simulator coordinates | 2.3020 cm | 25.0% |
| Shifted and rotated simulator coordinates | 2.3020 cm | 25.0% |
| Real reference query 6 | 0.8960 cm | 98.9% |
| Real reference in the simulator's coordinate range | 0.8960 cm | 98.9% |

Changing the overall pose frame did not explain the performance gap in these tests. This does not invalidate the previously fixed flange/fingertip transform, which is a different conversion between physical control sites.

## Wrist image orientation

Only the wrist image is rotated, in both context and query. The normal preprocessing remains unchanged. This is an image-orientation diagnostic, not a calibration of the camera's physical extrinsics.

| Wrist rotation | Mean position error | Gripper agreement |
| --- | ---: | ---: |
| Original | 2.30 cm | 25.0% |
| 90° | 1.86 cm | 53.3% |
| 180° | 2.02 cm | 41.7% |
| 270° | 2.03 cm | 34.2% |

The 90° condition was selected for a real simulator execution after its offline result. Using the same scene, seed 62000, task requirements, 415-action horizon, and 95-frame context budget as the earlier comparison, it still failed. It produced zero grasp frames and a minimum fingertip distance of 23.65 cm. There were 334 clipped position predictions. The run completed without an execution error.

The recording is in studio history as **ICRT pen / wrist rotated 90 degrees**, experiment `877e996a016740469c4c095997b3bb12`. Model weights and action limits were unchanged. The candidate camera rotation is not enabled in the default adapter.

## Complete demonstration without temporal compression

A further offline probe retained every one of the 383 demonstration frames, with the same 120-frame separate query. The experimental runtime allowed more than its usual 256 context frames, but the combined 505-frame budget stayed below the transformer's existing 512-frame limit. No model capacity or weights changed.

Error was 2.26 cm and gripper agreement was 25.0%, compared with 2.30 cm and 25.0% for sampled context. Removing demonstration subsampling did not fix the failure in this test. This is distinct from the earlier reference experiment where cutting off a demonstration's ending caused a large regression.

## What the evidence supports

The current checkpoint is not producing a useful pen approach or grasp policy from these synthetic observations. Predicted-action history amplifies the error, but correct past actions alone do not make the recorded simulator predictions accurate. A rigid coordinate change, full-resolution context, and a wrist image rotation do not close the gap.

The best-supported remaining explanation is a mismatch between the synthetic sensorimotor examples and the model's learned visual/task/action distribution. Our images show a distant table and Panda fingers; the reference uses substantially different camera views and gripper geometry. The pen/pouch workflow and slow motion commands also differ. These factors are not individually isolated, so it would be premature to claim that camera appearance alone is the cause.

The [authors' project](https://icrt.dev/) describes sensorimotor demonstrations formatted like the training data. In-context task conditioning does not establish arbitrary simulator or embodiment transfer. A credible next experiment needs a simulator observation and control setup matched to the reference robot, with a simple pick-and-place task and a successful teacher. Model adaptation would be a separate approach with changed weights and must be reported separately from fixed-weight ICL.

## Reproduction and checks

New tools live under `integrations/icrt/`:

- `prepare_transfer_probe.py` exports simulator context/query arrays from saved artifact IDs.
- `transfer_probe.py` compares predicted and recorded-past history.
- `coordinate_probe.py` applies rigid-coordinate and wrist-image interventions.
- `density_probe.py` evaluates every demonstration frame against a bounded query.
- `camera_rollout.py` executes the actual ICRT actor with the isolated wrist-image intervention.

The CLI help documents inputs. Use the backend environment for MuJoCo export/execution and the ICRT environment for model inference. Every probe refuses to overwrite its output directory. Raw evidence and source hashes are in `.data/validation/icrt-transfer-2/`; old runs remain unchanged. The density input `context-full.npz` was generated by the saved `prepare_full.py`, retaining all 383 actions from the original demonstration.

Ten integration tests passed, covering action references, temporal inference, reference metrics, rigid-transform round trips, preserved local deltas, unchanged gripper values, and preservation of images/source arrays. All five new scripts compiled. No production adapter, model weight, task requirement, or judge rubric was changed in this investigation. No paid GPU was started.
