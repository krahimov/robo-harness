# Recovery, stress tests, calibration, and ICRT

Verified September 8, 2026 (local time). Camera feedback and a fixed stress suite are implemented. Human calibration is waiting for labels. ICRT completes the longer rollout after a context-capacity fix, but still fails the pen task.

## Camera feedback during an episode

Enable **Inspect cameras and replan during execution** in the Astra workspace. The executor pauses simulated time at grasp/lift/release checkpoints, supplies current overview and wrist images plus simulator state, and asks Astra to keep or replace the remaining motions. It preserves the running simulator, task requirements, physics, and executed history. Each episode receives a fresh feedback instance, with one to three model calls. The total motion budget is 44 seconds plus one second of settling. This is state-assisted, checkpoint-based control, not vision-only control or a real-time hardware controller.

The saved trace includes checkpoint time, reason, policy hash, camera/state artifact, raw reply artifact, and replacement motions. Invalid feedback stops execution and retains partial evidence. Final episodes share a frozen starting program and feedback rules; no final case feeds another case or development.

A paired fault test used the previously frozen pen program and seed 2000. The gripper was forced to stay open until the controller issued an explicit open-then-close retry. Actual executed actions retain the intervention.

| Condition | Result | Evidence |
| --- | --- | --- |
| No camera feedback | Failed; no completed placement | Experiment `84f9ef9fb53e4baf8dd2debac04056f1` |
| Camera feedback v2, two calls | Passed all six numerical requirements | Experiment `7e65e52b204b4b318257992f9bd3a797` |

At 7.47 seconds Astra observed the failed grasp and replaced the remaining motions. At 11.60 seconds it observed a successful grasp and kept the revised program. The resulting recording contains 462 frames over 30.73 seconds. This establishes recovery for this injected fault, not a broad recovery success rate. The trial used numerical task evaluation; its success is not a human-calibrated VLM verdict.

Two earlier probes are retained. The first camera launch failed because the diagnostic script lacked the backend module path. A weaker single-close fault passed both with and without feedback; it does not establish a recovery benefit. The stronger paired test above was introduced to make that distinction measurable.

## Eight fixed stress cases

Choose **Pen stress v1 · 8 fixed cases**. Cases are declared before evaluation: baseline, pen ±20 mm in X, pen ±15° on the table, 12 mm pen diameter, pouch +40 mm in X, and an obstacle near the transfer path. Pouch relocation translates the target bounds with the fixture; it does not relax the success predicates or change target dimensions. Supports move with the pen.

The frozen program passed **8/8** cases on fresh seeds 52000–52007, with feedback disabled. All six numerical outcomes passed in each case. Experiment: `8d00ef9ad2b84f19b0cb43c7c7b43c0d`, **Pen stress suite v1 / aligned supports**.

The earlier suite scored 7/8, but inspection found a sign error that rotated the pen and its supports in opposite directions. That original suite remains recorded; the corrected suite uses new seeds, and an independent geometry test checks support alignment. Neither eight cases nor new seeds establish general manipulation robustness. The pouch is rigid and held open; cloth, zippers, and physical hardware remain untested.

## Human calibration

**Judges → Frozen calibration samples** reserves separate development and holdout evaluations before labeling. The UI hides the selected judge's verdicts by default, and every human label stays tied to an exact evaluation and rubric version. Automatic rubric proposals exclude reserved holdout labels. Unknown human labels remain unresolved.

Sample `b92d1505cf754cd3bc90ec1a1f7398e5` contains two model passes and two model failures, divided evenly between development and holdout. It has **zero human labels** as of this verification, so both agreement values are null. The user must review the four recordings and enter the six outcomes for each. No automated judgment has been counted as a human label.

This small stratified sample checks a calibration workflow; it cannot estimate population agreement. Even a fully labeled four-case packet would need a larger, independently sampled follow-up before supporting broad calibration claims.

## ICRT: fixed capacity, unresolved manipulation

The adapter now reserves query capacity before selecting demonstration frames. A 415-step rollout has a 95-frame demonstration budget under the runtime's 512-frame limit, including a two-frame margin. Uniform sampling keeps the demonstration ending. Invalid horizon/context combinations fail before execution. The weights are unchanged.

The matched pen comparison `f363836e8e494c0ea803c3aa57055f6d` used seed 62000 and 415 actions:

| Context | Completed actions | Held frames | Clipped predictions | Task |
| --- | ---: | ---: | ---: | --- |
| None | 415 | 0 | 211 | Fail |
| One complete-ending demonstration, sampled to 95 frames | 415 | 0 | 300 | Fail |

The capacity fix prevents a long-run failure; it does not explain or resolve the earlier short-run 0/3 manipulation results.

A separate offline camera intervention kept the complete reference demonstration, robot state, actions, weights, and runtime fixed. It queried separate recorded episodes 6–8 using the model's own predicted history. Values below are unweighted means across the three query episodes:

| Images | Mean translation error | Gripper agreement |
| --- | ---: | ---: |
| Original reference images | 0.95 cm | 98.7% |
| Blank query images | 3.43 cm | 57.6% |
| Blank context and query images | 3.19 cm | 57.6% |
| Swapped camera order in context and query | 3.29 cm | 51.8% |

This shows sensitivity to visual input and camera order. It supports investigating visual deployment mismatch, but it does not isolate simulator appearance as the sole cause. Offline prediction is not closed-loop task completion. ICRT's simulated pen grasp remains unresolved.

## Verification and evidence

- Backend: **244 tests passed**, including checkpoint replacement on the same simulator, fresh feedback per episode, frozen stress rules, capacity reservation, aligned fixtures, and calibration holdout exclusion.
- ICRT runtime: **8 tests passed** for action references, temporal predictions, and reference evaluation.
- Frontend: TypeScript and production build passed. Existing dependency deprecation warnings and the bundle-size advisory remain.
- Browser: feedback controls, eight-case selection, recorded checkpoint trace, frozen review sample, hidden verdicts, and review playback verified. Saved runs and the unlabelled calibration sample survived the backend restart.

Raw recordings, plans, and model replies remain under `.data/validation/closed-loop-sticky-control/`, `closed-loop-sticky-recovery/`, `pen-stress-2/`, and `icrt-vision-ablation-1/`. `.data/validation/remaining-work/` contains the combined verification summary, final ICRT snapshots, and a source/evidence hash manifest. No paid cloud GPU was started; inference used local CPU and the existing Codex connection.
