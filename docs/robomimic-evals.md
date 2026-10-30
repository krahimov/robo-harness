# robomimic / RoboTurk-derived evals

robomimic's benchmark tasks run on robosuite. Its multi-human (`mh`) datasets were collected through **RoboTurk**. The studio ports two of these tasks as human-owned task presets, so every actor is scored by the same fixed numerical checks: GPT-6 Astra, Claude Opus 5.5, ICRT, or replay.

| Preset | robosuite env | Requirements (all required) |
|---|---|---|
| `robomimic-lift` | `Lift` | Cube held with its underside ≥ 2 cm above the table at least once; no off-table drops |
| `robomimic-can` | `PickPlaceCan` | Complete can inside the target bin compartment at episode end; released; at rest; no drops |

Both presets require the Franka Panda with finger-contact grasping. robomimic also uses a Panda.

## These are ports, not the benchmark

Scores are **not comparable** to published robomimic success rates. Each preset lists its deviations in `task.benchmark.deviations`. The main ones:

- **Lift:** the cube half-size is 2.5 cm instead of 2 cm. The threshold keeps robosuite's 2 cm underside clearance, and an active grasp is required while lifted.
- **Can:**
  - Only the target bin quadrant is modelled (19.5 × 24.5 cm, 4 cm walls).
  - The can radius is 2.5 cm because of the scene language's cylinder limit.
  - The whole can must be in the bin, which is stricter than robosuite's center-only check.
  - There is no clutter.
- The task horizon follows the motion-program limits, not robosuite's 20 Hz step horizon.

A scripted motion program solves both tasks through the Franka contact executor, and an idle program fails them (`tests/test_benchmarks.py`). So the checks can be achieved, but they are not passed trivially.

## Run them

In the UI, pick the task in Workspace. Choose **LLM motion programs**, select GPT-6 Astra or Claude Opus 5.5, choose a judge, and run.

To compare both models across the suite on a running studio:

```bash
python scripts/run_model_evals.py                     # lift + can × both models, 4 holdouts each
python scripts/run_model_evals.py --holdouts 8 --attempts 3 --feedback checkpoints
```

Each (task, model) pair is an ordinary experiment. It runs development revisions, then freezes a program and tests it on fresh holdout seeds. The script prints a table and saves `.data/evals/<timestamp>.json`. Holdout passes require both the state checks and the judge. Human review is separate.

## Next steps (not implemented)

- **Square (NutAssembly):** needs a movable nut with a hole, which the scene language does not have.
- **A robosuite worker:** robosuite behind the [adapter protocol](adapter-protocol.md), running in its own environment because of robosuite's `mujoco<3.10` pin. This would give exact benchmark geometry and contacts, and let robomimic `ph`/`mh` HDF5 demos load as ICRT context. The studio would still compute success from per-frame object states rather than trusting robosuite's reward. It needs an adapter from motion programs to 7-D delta OSC actions.
- The original RoboTurk Pilot dataset is Sawyer-only, uses a legacy format, has no stated license, and is 8.8 GB. We skip it in favour of robomimic's MIT-licensed Panda datasets (`robomimic/robomimic_datasets` on Hugging Face).
