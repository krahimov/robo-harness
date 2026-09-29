# Feedback-driven robotics experiments

September 7, 2026, New York time.

The moving-conveyor task now has a successful **Astra-authored tool controller**. It passed its development episode and **3/3 held-out seeds** with a frozen motion program. The released compact **ICRT policy still failed every new test**. These are different actors and observation contracts.

## What succeeded

GPT-6 Astra, through the existing Codex subscription connection, received initial side/wrist RGB, explicit simulator object state, the factory scene description, and the original task. It generated a bounded seven-motion program. The executor resolves relative targets from live object/gripper positions at the start of each motion, interpolates the requested pose, and applies top-down IK and gripper commands. Astra is not called at every physics step. No prerecorded teacher trajectory was supplied to this actor.

The program led the moving cube by 3 cm during approach, closed, lifted, carried it over the tray, lowered, released, and retracted. Native Franka finger contacts handled the grasp; there was no proximity weld. Simulation ran locally in MuJoCo.

The development seed was 2000. After the first program passed, it was frozen before testing seeds 3000, 3001, and 3002 with the existing 2 cm initial-position variation. All three passed placement, release, settling, and no off-table drops. Recorded grasp contact lasted 67, 82, and 75 frames, respectively. This is a small holdout test of a state-assisted motion program, not evidence of vision-only robotics or physical in-context learning.

A separate Astra judge session evaluated seed 3000 using three evidence requests: complete-interval state statistics, four sampled replay images, and the final half-second of state. It returned pass for all four original outcomes. Human review remains pending; agreement between these two judges is not human calibration.

The candidate demonstration is in the local studio library as **Astra / moving-conveyor pick and place**, ID `4830484130024ed4b5e59bab7b3b9bae`. Its metadata identifies the actor and explicit simulator-state assistance. It was not silently promoted as an approved ICRT skill.

Evidence: `.data/runpod/astra-tool-actor-1/`, including the proposal, frozen program, complete rollouts, holdout report, separate judge result, and `successful-rollout.gif`.

## What did not succeed

ICRT ran on an RTX 4090 with the same fixed compact DROID-pretrained weights as the earlier trial. No gradients or checkpoint changes were used. Each new candidate ran one 240-step episode at 15 Hz, with complete actions and state saved. Every supplied teacher passed before being offered as context.

| Candidate | Change | Result |
| --- | --- | --- |
| Stationary baseline | No belt or machinery; one native teacher | Fail, no grasp |
| Closer camera | Closer overview and lighter diagnostic tabletop | Fail, no grasp |
| Binary gripper | Closer view; threshold gripper commands | Fail, no grasp |
| Constrained orientation | Front view, binary gripper, top-down IK | Fail, no grasp |
| Astra revision 1 | Realized next EE poses as demo actions | Fail, no grasp |
| Astra revision 2 | Half-duration demonstration motions | Fail, no grasp |
| Astra revision 3 | Two shorter demonstrations | Fail, no grasp |
| Forward cube | Move the diagnostic cube farther from the base | Fail, no grasp |
| Forward-right cube | Also move the diagnostic cube laterally | Fail, no grasp |
| Astra context | Original factory task with the successful Astra demonstration | Fail, no grasp |

These ten tests vary several experimental conditions and are not a matched benchmark or a statistical success-rate estimate. They use development seed 2000. The final factory context demonstration also uses seed 2000; even this favorable same-scene transfer failed. Earlier comparisons remain unchanged.

A separate probe fed ICRT the observed frames of a successful stationary demonstration. Mean translation error against commanded targets was 2.84 cm; binary gripper agreement was 51.8%. The probe retains the model's preceding action history, so it is neither fully teacher-forced inference nor an independent rollout. It suggests a model/interface/distribution problem beyond downstream contact dynamics; it does not establish a single root cause.

Evidence: `.data/runpod/iteration-evidence/complete/iterations/`. The downloaded archive's SHA-256 matched the remote file, and a local manifest covers 67 archived files. The three optimizer proposals and reports are in `.data/runpod/astra-icrt-improvement-2/`.

## Reusing the experimental loop

The experiments below used a CLI. Astra motion-program development is now also available through the [in-app experiment loop](astra-experiments.md). The ICRT optimizer remains a CLI workflow.

- `integrations/icrt/iterate.py` records and validates demonstrations, executes a candidate on the private ICRT worker, and saves task snapshots, model provenance, actions, state, and verdicts. Failed teachers become explicit rejected candidates without running the actor. Output directories cannot overwrite prior attempts.
- `integrations/icrt/improve.py` uses Astra through Codex to propose bounded camera, demonstration, and controller settings from prior judge reports. Proposals cannot redefine the task, change weights, choose evaluation seeds, or replace ICRT with replay. A passing development candidate is frozen for disjoint holdout seeds. The three live revisions above exercised this loop.
- `integrations/icrt/tool_actor.py` tests the separate Astra motion-program approach. It can revise failed programs from recorded feedback, and freezes the first passing one for holdout evaluation. This actor succeeded on its first proposal, so its failure-revision branch has not been exercised live.
- `integrations/icrt/teacher_probe.py` measures predictions on demonstration observations without calling the result rollout success.

On a prepared Linux GPU host with the private ICRT worker running:

```sh
MUJOCO_GL=egl PYTHONPATH=backend backend/.venv/bin/python integrations/icrt/iterate.py \
  --output .data/iterations/first --task stationary --view close --seeds 2000
```

Locally, using a verified SSH config and existing Codex sign-in:

```sh
PYTHONPATH=backend:. backend/.venv/bin/python integrations/icrt/improve.py \
  --ssh-config /absolute/path/to/ssh_config --host iteration \
  --output .data/iterations/optimizer --rounds 3 --history /absolute/path/to/report.json

PYTHONPATH=backend:. backend/.venv/bin/python integrations/icrt/tool_actor.py \
  --output .data/iterations/astra-controller --task factory
```

The factory CLI currently reads the saved first-trial fixture under `.data/runpod/gpu-trial-1/trial.json`; this is a research workflow, not a portable public example. The optimizer expects the remote checkout at `/workspace/studio`. It uses the backend venv for simulation and the separate worker for inference. The old trial SSH config points to a terminated pod and must be replaced for a new run. Credentials remain local to Codex/SSH and are not sent to model prompts or the GPU workload.

## Verification and resources

- Backend regression suite: 219 passed.
- New experiment-boundary tests: 3 passed. They cover immutable task/actor fields, failed-teacher rejection without inference, output preservation, and bounded motion schemas.
- Frontend TypeScript/Vite build passed with its existing bundle-size advisory.
- Paid pod `gr9mr2wy55cges` was created at 03:01:01 UTC and terminated at approximately 03:14:06 UTC on September 8. About 13 minutes at the reported $0.74/hour compute rate is approximately $0.16 compute plus storage, not a billing invoice. The complete evidence was downloaded before termination.

The practical result is a working state-assisted LLM motion controller and an exercised experiment-improvement loop. Improving the released ICRT checkpoint's actual task performance remains unresolved; no camera or demonstration candidate from this batch earned promotion.
