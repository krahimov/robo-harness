# Astra experiments in the studio

Astra is selectable as a robot actor in **Workspace → Robotics model → GPT-6 Astra · motion programs**. Its runner uses the existing experiment, episode, evaluation, and human-review records. It does not launch Runpod or require an API key. The saved Codex login supplies model access, and model and VLM judge calls consume subscription usage.

## Run a task

1. Save a scene's demonstration into Workspace, or choose a task preset such as **Pen into an open pouch**. Astra currently requires Franka Panda.
2. Select Astra as the robotics model. Keep up to four compatible demonstrations, or remove them for a run without demonstration samples.
3. Choose a judge. State checks run locally; an Astra or external VLM judge uses the configured connection and rubric version.
4. Set the development proposal budget, from one to six, and final evaluation count, up to eight. The development seed selects the scene used during revisions. For the pen task, its separate object-variation field controls the pen's jitter; the primary-box slider does not.
5. Click **Run experiment**. Inspect proposals and recordings as they appear. The first program that passes both fixed state checks and the chosen judge is frozen and tested on fresh seeds.
6. Open each development or final episode to inspect its recording, camera frames, and outcome report. Rejudging creates a new evaluation. Human acceptance is still required before saving an episode as a candidate demonstration.

## What Astra controls

The actor receives the initial overview and wrist images, explicit simulator state, task instructions, goal positions, and up to 16 evenly spaced state/action pairs per demonstration. These are sampled numerical examples, not full visual demonstration clips. The UI and saved run contract disclose this distinction.

It returns four to eighteen motions. Each motion names a moving object, gripper, or declared goal; adds a world-space XYZ offset; chooses a wrist yaw; and opens or closes the gripper. The executor resolves the relative target from live simulator state at the start of each motion. It interpolates position and shortest-path orientation, using the Franka contact simulator. Requested motion duration is at most 30 seconds, followed by one second of settling.

Astra can revise programs between development attempts and, optionally, at camera checkpoints within an episode. Checkpoints inspect overview/wrist images and current simulator state after grasp, lift, or release motions. They can replace the remaining motions without resetting the simulator. A fresh feedback session is used for each episode, with one to three calls and at most 44 seconds of motion plus settling. Simulated time pauses during model calls; this is not a real-time hardware controller. See [recovery and stress verification](closed-loop-and-stress.md).

The only automatically revised artifact is the motion program. The scene, physics, task thresholds, selected demonstrations, judge version, and model weights stay fixed for the run. Automatic scene or demonstration editing remains separate work.

## Evidence boundaries

- Every run snapshots the task, scene, judge, and selected context. SHA-256 hashes identify the task, scene, judge, and frozen program.
- The server allocates final seeds before scheduling, avoiding the development seed, stored demonstration seeds, and known prior experiment seeds. They are never included in the actor's development prompts.
- Invalid proposals consume the proposal budget and retain their raw model response. Valid proposals retain programs, recordings, verdicts, and episode links. Simulation interruption preserves aligned partial frames/actions with an execution-error verdict.
- A VLM pass cannot override a failing numerical task check. A judge error stops the run with evidence retained.
- After a passing development proposal, the planner closes. Holdouts share the same frozen starting program and, when enabled, feedback rules. Each episode may react to its own observations; final cases never supply feedback to other episodes or development.
- Final results count only holdout episodes that passed both checks. Development successes are displayed separately. Rejudging an episode cannot rewrite the original frozen-run summary.
- Budget exhaustion, cancellation, provider errors, and judge errors are distinguishable from task success. Restarting the backend marks unfinished runs interrupted and preserves stored evidence.

Fresh seeds establish separation from this run's development episodes, not physical independence or generalization beyond the configured variation. With zero variation, different seeds can produce identical scenes. If someone manually tunes on final results, those cases have become development data.

## API

`POST /api/experiments` accepts `adapter: "astra"`, `max_attempts`, `feedback_mode` (`none` or `checkpoints`), `max_checkpoints` (1–3), `stress_suite` (`none` or `pen-v1`), and the existing scene, task ID, demonstration IDs, judge ID, instructions, and `episodes` fields. For Astra, `episodes` is the final holdout count. The server rejects caller-supplied holdout seeds, horizon overrides, incompatible embodiments, and more than four context demonstrations.

`GET /api/experiments/{id}` includes `phase`, `message`, `attempts`, `frozen_program`, `holdout_seeds`, `holdout_completed`, `holdout_passes`, and `outcome`. Episode records distinguish `development` and `holdout`. Existing cancellation, rejudging, camera replay, review, and promotion endpoints apply.

Astra is excluded from **Compare context** for now. That feature compares ordinary episode actors under matched conditions; optimizing each condition would require a separately specified comparison design.

The earlier CLI controllers remain available as historical research tools. The in-app actor lives in `backend/studio/adapters/astra.py`; orchestration lives in `backend/studio/astra_runs.py`.

## Live validation

On September 8, 2026, the run **Astra studio / pen insertion** was submitted through the browser UI with a three-proposal budget, the saved pen demonstration as sampled context, and **Astra evidence judge v2**. The first proposal passed development seed 2000. Its frozen program then passed all six numerical requirements and the separate VLM judge on fresh seeds 125081894, 1957549927, and 312561033. All four recordings contain 415 frames over 27.6 seconds. Human review is pending.

This is a small test with the existing held-open rigid pouch and ±4 mm pen position variation. It does not establish broad manipulation robustness, vision-only control, or real hardware success. Because the live program passed its first development attempt, the new runner's fail→revise branch was verified with controlled regression tests rather than a failure in this particular live run.

Evidence is saved in `.data/validation/astra-studio/`: the full experiment snapshot, model response, four recordings, and a SHA-256 manifest. The experiment remains in the studio history, with ID `de8c6487b6f44edeb759ecd80aea468c`. Tests cover failed development then frozen holdouts, exhausted budgets, invalid proposals, partial cancellation evidence, judge errors, state-check vetoes, seed separation, prompt/context sampling, and actual native contact execution.

Final verification: **238 backend tests passed** and the frontend production build passed. Browser checks confirmed the controls, completed run history, replay, judge report, and human-review actions. The completed run survived a backend restart. Existing dependency deprecation and bundle-size warnings remain.
