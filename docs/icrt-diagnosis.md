# ICRT failure investigation

September 8, 2026. The Astra tool controller remains the working actor. ICRT still does not grasp the factory cube after the corrections below. We found an interface defect, but have not isolated a complete explanation for the failed task.

## Confirmed pose-frame defect

DROID's [published robot configuration](https://github.com/droid-dataset/droid/blob/ba46d4af805bce44e6a40cff10ed094ee5090ab8/config/panda/franka_panda.yaml) specifies `panda_link8` as the end-effector frame. Its [robot implementation](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py) reports forward kinematics in that frame. Studio previously supplied the native simulator's fingertip control site directly. In our pinned Menagerie robot these frames differ by 10.34 cm along the hand's local Z and 45 degrees around that axis.

Adapter `icrt-studio-v3` now converts native demonstration observations and commanded targets into the flange frame, then converts predicted flange commands back to the simulator's fingertip frame before applying motion limits. Cartesian projection is unchanged. The conversion lives in `backend/studio/adapters/icrt_frames.py` and is shared by the research rollout script. Native commands without an explicit orientation now use the simulator's actual default, a downward-facing hand.

For column-vector rotations, with `d = [0, 0, 0.1034]`:

```
R_flange = R_tip @ Rz(+pi/4)
p_flange = p_tip - R_tip @ d
R_tip = R_flange @ Rz(-pi/4)
p_tip = p_flange + R_tip @ d
```

Tests check a known downward pose and independently compare the converted site against MuJoCo link7 forward kinematics plus its 0.107 m flange offset. This establishes consistency with the documented DROID frame; it does not establish equivalence to every recording in the released ICRT data.

## Controlled local retests

Each condition used factory development seed 2000, a passing native teacher, fixed checkpoint weights, and 180 control steps at 15 Hz. Both demonstration poses and rollout poses changed together for the corrected-frame conditions. Task requirements remained fixed.

| Pose frame | Inference | Grasp frames | Closest fingertip-to-cube distance | Result |
| --- | --- | ---: | ---: | --- |
| Previous fingertip input | One-step execution | 0 | 7.64 cm | Fail |
| Previous fingertip input | Ten-step temporal ensemble | 0 | 19.30 cm | Fail |
| Corrected flange input | One-step execution | 0 | 8.70 cm | Fail |
| Corrected flange input | Ten-step temporal ensemble | 0 | 16.44 cm | Fail |

These are four diagnostic episodes, not a success-rate benchmark. The correction did not produce an observed task-success gain. Earlier 0/3 comparisons remain historical artifacts using v2; they have not been overwritten or relabeled as v3 results.

The [paper's Appendix 8.4](https://arxiv.org/html/2408.15980v1) describes temporal ensembling with a ten-step action horizon. Production uses one-step execution. `integrations/icrt/temporal_runtime.py` tests an ensemble that anchors every predicted chunk to the observation that generated it before averaging overlapping targets. Simply changing the upstream ensemble flag would mix pose references under our existing correction. Synthetic tests verify that older chunks retain their original reference after the robot moves. The experimental ensemble remains separate from production because these tests did not improve task success; it is not a claimed reproduction of the paper's full deployment stack.

## Checks that passed

The loaded model exactly matches all 149 released policy tensors and all 148 visual-encoder tensors checked, with no unmatched active visual tensors. Streaming a 24-frame context in eight-frame chunks produced exactly the same tested action as loading it at once. The image pipeline uses the expected bicubic resize, center crop, and ImageNet normalization. Source dataset preprocessing converts BGR to RGB; our RGB input is consistent with it.

These checks narrow the investigation. They do not rule out every model-wrapper defect or prove that our camera calibration, action timing, or simulated embodiment matches the training setup.

## Real-data probe and remaining uncertainty

We fetched only 96 consecutive frames from the authors' `real_episode_2024-05-31-tiger-to-black-bowl_5` in [ICRT-MT](https://huggingface.co/datasets/Ravenh97/ICRT-MT), using bounded range reads totaling 35,127,296 bytes. The sample shows a different gripper and substantially different wrist/side camera views from our simulation. Visual and embodiment transfer are plausible remaining failure sources, not experimentally isolated causes.

A limited offline probe used these 96 observations, retaining predicted action history. Mean translation error was 3.87 cm without context and 4.62 cm with the same 96-frame segment as context; binary gripper agreement was 83.3% in both cases. Holding the current position gives a 3.80 cm translation-error baseline. This truncated, same-episode diagnostic establishes neither useful real-data prediction nor closed-loop success. It is not held-out evaluation.

Checkpoint provenance also needs care. The filename contains `droid_pretrained`, but the supplied YAML references an expanded real-task dataset and another pretrained checkpoint; the official inference notebook uses this same release. The checkpoint contains no dataset or epoch metadata. Therefore we cannot identify it as the paper's DROID-only ablation from its filename. The current release's policy, encoder, and YAML hashes match our pinned revision.

The [author reports](https://github.com/Max-Fu/icrt/issues/8#issuecomment-2730369207) that the paper did not evaluate simulation and highlights simulation transfer and action chunking as concerns. That supports investigating the deployment match, not concluding that ICRT lacks in-context learning.

The next decisive milestone is a reference-data reproduction with complete, separate context/query episodes and confirmed checkpoint provenance. After that, match the simulator's gripper, cameras, workspace, and timing to that reference before measuring simulation transfer. Changing demonstration content alone cannot resolve an unverified observation/action contract.

## Evidence and verification

Local evidence is preserved in `.data/validation/icrt-diagnosis/`: four rollout directories with configurations, teachers, predictions, state, and verdicts; `runtime-audit.json`; `real-probe.json`; `sample-manifest.json`; and `author-comments.json`. Scripts `audit_runtime.py` and `probe_dataset.py` retain the diagnostics without changing model weights. Generated samples and recordings stay outside Git.

- Backend: 221 tests passed, including the independent pose-frame checks.
- Runtime action-reference and temporal-ensemble tests: 5 passed.
- Experimental boundary tests: 3 passed.
- Frontend TypeScript/Vite build passed with its existing bundle-size advisory.

All new diagnostics ran locally on CPU. No paid GPU resource was started for this investigation.

## Follow-up: complete reference episodes

The [complete reference-data evaluation](icrt-reference-evaluation.md) has now finished. Eighteen planned conditions across three separate query episodes show that the released checkpoint and current runtime benefit from a complete demonstration. Production mean position error fell from 3.53 cm to 0.95 cm; gripper agreement rose from 72.0% to 98.7%. Cutting that demonstration back to 96 frames increased mean error to 4.46 cm in a secondary controlled test. The negative truncated probe above should not be used to infer absence of context conditioning. Simulation transfer remains unresolved; offline reference predictions do not establish robot task completion.

## Follow-up: camera interventions and long rollouts

The [recovery and stress verification report](closed-loop-and-stress.md#icrt-fixed-capacity-unresolved-manipulation) adds a controlled reference-image intervention and a query-capacity fix. Blanking or swapping cameras degraded held-out reference predictions. Reserving context space allowed both 415-step simulated pen trials to complete, but neither grasped the pen. These results narrow the diagnosis without establishing a sole cause or successful simulation transfer.

## Follow-up: locating the simulated approach failure

The [simulation transfer diagnosis](icrt-transfer-diagnosis.md) adds recorded-observation probes, rigid-coordinate interventions, three wrist rotations, a full-resolution context test, and another 415-step execution. ICRT stays more than 23 cm from the pen in the tested rollouts. Errors persist even without new physics and with correct past actions. Neither coordinate alignment nor retaining every demonstration frame fixes the recorded-action predictions. A wrist rotation improves offline predictions but still fails in execution. The remaining problem is transfer to the synthetic sensorimotor distribution; the exact visual/task/action factors are not yet individually isolated.
