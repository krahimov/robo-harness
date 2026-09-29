# ICRT complete reference-episode evaluation

This experiment tests action prediction on recorded robot observations. It does not execute a robot or score task completion. No model weights change.

## Frozen protocol

The dataset is ICRT-MT revision `7eee568a981d2f58b5425beb2e0ea87eb9d30e1d`. The complete May 31 tiger-to-black-bowl episode 5 supplies context, with 117 frames. Episodes 6, 7, and 8 are separate queries with 95, 97, and 101 frames. The selection was fixed before prediction. These episodes may have appeared during model training; separate query episodes are not a claim of unseen-task generalization.

Each query is tested with zero and one demonstration under three procedures:

- Production own-action history: current runtime, one-step execution, current-pose correction, feedback of predicted actions.
- Corrected recorded-action history: same runtime and execution, but only past recorded actions replace predicted history. Each past action is encoded relative to its own observation. Current or future query targets are never supplied before prediction.
- Upstream notebook procedure: the notebook's wrapper calls, full prompt, previous recorded action, teacher forcing, default 16-step execution horizon, and temporal ensembling disabled. This preserves upstream pose-reference behavior so it can be compared explicitly.

All conditions receive the same recorded query observations. Every frame is retained at its original index; no resampling or truncation is allowed. Query outputs cannot alter later observations, so this is offline prediction rather than closed-loop imitation. Zero-context runs receive no task instruction because the released policy has no language input.

Metrics include mean Euclidean translation error, geodesic rotation error, gripper error and balanced accuracy, plus translation error on frames whose requested movement exceeds 1 cm. Holding the recorded current pose and gripper is the baseline. Aggregate metrics weight each episode equally.

## Reproduction

From the repository root, with the pinned ICRT source and weights installed:

```sh
uv run --no-project --with h5py --with numpy python integrations/icrt/fetch_reference.py
integrations/icrt/.venv/bin/python integrations/icrt/reference_eval.py
uv run --project integrations/icrt pytest -q integrations/icrt/test_reference_eval.py
```

The fetcher reads bounded ranges instead of downloading the full 44.5 GB HDF5 file. It refuses to replace downloaded episodes. The evaluator verifies hashes, checks distinct context/query IDs and complete frame indices, and refuses to overwrite a result directory. Local evidence is in `.data/validation/icrt-reference-1/`; the evaluation plan is written before inference.

The upstream reference is [tools/inference.ipynb](https://github.com/Max-Fu/icrt/blob/e0c95887d577c95ef9d0eef5ee23960043d6cb5e/tools/inference.ipynb), using data from [ICRT-MT](https://huggingface.co/datasets/Ravenh97/ICRT-MT). Our query episode IDs differ from the notebook's example; its call procedure is preserved in the upstream condition.

## Results

All 18 planned conditions completed on local CPU. The production controller supplied no recorded query actions. Its mean position error fell by 73.0% with a separate, complete demonstration, and improved on all three query episodes.

| Query episode | No context position error | One demonstration position error | No context gripper agreement | One demonstration gripper agreement |
| --- | ---: | ---: | ---: | ---: |
| 6 | 3.61 cm | 0.90 cm | 70.5% | 98.9% |
| 7 | 3.48 cm | 1.10 cm | 75.3% | 100.0% |
| 8 | 3.50 cm | 0.86 cm | 70.3% | 97.0% |

Aggregated with equal weight per episode:

| Procedure | Context | Mean position error | Gripper agreement |
| --- | ---: | ---: | ---: |
| Production, predicted history | 0 | 3.53 cm | 72.0% |
| Production, predicted history | 1 | 0.95 cm | 98.7% |
| Corrected, recorded past history | 0 | 1.13 cm | 98.6% |
| Corrected, recorded past history | 1 | 0.77 cm | 99.3% |
| Upstream notebook procedure | 0 | 3.47 cm | 90.4% |
| Upstream notebook procedure | 1 | 1.78 cm | 98.3% |
| Hold current recorded pose/gripper | None | 3.37 cm | 92.1% |

The notebook procedure also improved with context. Corrected recorded-history inference was more accurate than production without context, showing that predicted-action history affects this offline test. With context, the gap shrank. Because notebook mode differs in both pose handling and execution horizon, its comparison is not an isolated test of either variable.

This is evidence that the released model and our inference runtime can use demonstration context on these reference episodes. It does not prove unseen-task learning, collision avoidance, grasp success, or closed-loop stability. The simulation failures remain failures. Camera appearance, embodiment geometry, action distributions, and controller dynamics remain plausible transfer problems that this offline test cannot isolate.

The prior 96-frame, same-episode probe remains preserved. Its negative result should not be used to conclude that the released checkpoint cannot benefit from context.

## Verification

Eight integration tests passed, including three new checks for split/capacity integrity, metrics, and past-action pose encoding. The new scripts do not alter the production controller, checkpoint, UI, or previous simulation verdicts. No GPU pod was started. `summary.json`, the frozen `evaluation/plan.json`, all 18 prediction traces, and source episode hashes are retained beside the complete data.

## Secondary context-completion diagnostic

After seeing the first positive result, we repeated the three complete query episodes with only the first 96 of the same 117 demonstration frames. This exploratory ablation changes context completeness only; query frames, runtime, and weights remain fixed.

| Query | 96-frame context position error | Complete context position error |
| --- | ---: | ---: |
| 6 | 4.47 cm | 0.90 cm |
| 7 | 4.84 cm | 1.10 cm |
| 8 | 4.06 cm | 0.86 cm |

Mean error was 4.46 cm with the truncated demonstration versus 0.95 cm with the complete demonstration. At frame 95 the gripper is still over the bowl and opening. Later frames show release and retreat, with the gripper open and withdrawn at frame 116. The ablation does not distinguish semantic completion from the effect of additional frames, but it establishes sensitivity to this specific truncation. Preserving demonstration endings is a concrete requirement to test in future context preparation.

The negative earlier short probe must be interpreted in light of this sensitivity. It was not sufficient evidence against this checkpoint's context-conditioning ability. These results do not by themselves explain failure on complete synthetic demonstrations. The secondary plan, predictions, and metrics are preserved in `prefix-ablation/`.

A final diagnostic replayed the exact earlier 96 query frames with the full 117-frame context. The old and new camera, proprioception, and target arrays matched byte for byte over that prefix. Mean error fell from the previously recorded 4.62 cm to 0.82 cm; gripper agreement rose from 83.3% to 99.0%. This uses the same episode for context and query and is solely an explanation check for the earlier probe, not generalization evidence. Its predictions and comparison checks are in `same-episode-completion-check.json`. The three separate query episodes above remain the primary result.
