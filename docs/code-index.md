# Source index

## Current scenario module

| File | Role |
|---|---|
| `scripts/scenario_selector.py` | Fit one of six components, calibrate a loss budget, and route original answers without evaluation labels |
| `scripts/pack_scenario_cache.py` | Assemble extracted blocks and explicit A/B record lists into portable NPZ caches |
| `scripts/extract_scenario_qwen.py` | Frozen Qwen prefill: final state, question/image pools and source score |
| `scripts/extract_scenario_internvl.py` | Frozen InternVL prefill with its native conversation template and CLIP-UP image loader |
| `scripts/verify_scenario_records.py` | Recompute component tables, frozen Circular results, unified-reference losses/repairs and paired intervals from released records |
| `splits/mmupd_adaptation.json` | Exact CM partition, adaptation IDs and image-cluster folds |
| `results/two_block/` | Two source heads, all six scenario heads/policies, row scores, labels and original answers |

The [scenario guide](scenario-adaptation.md) describes the complete extraction, fitting, recalibration and application commands. `scripts/verify_vizwiz_full.py` recomputes the [open-ended complete-module confirmation](vizwiz-full.md), including its paired intervals, from compact row records.

## InternVL training route

| File | Role |
|---|---|
| `internvl/ivl_recipe.py` | Both stage recipes, training rows from the Qwen scripts' row functions, data order and the CPU data check |
| `internvl/ivl_common.py` | Model loading, prompt, training example, LoRA layer, forward pass, loss, schedulers and float32 merge |
| `internvl/ivl_train_sft.py` | Answer SFT (B1) or ordinary refusal SFT |
| `internvl/ivl_merge.py` | Merge an adapter into bfloat16 weights with one rounding, then reload and check |
| `internvl/ivl_check_masking.py` | Optional read-only check of inputs, loss masking, zero-initialized LoRA and gradients |
| `internvl/recorded_run.json` | Data reports, environment, training and merge records of the runs behind the released results |

The [InternVL training guide](internvl-reproduction.md) gives the environment, recipe and commands.

## Source-head and baseline pipeline

| File | Role |
|---|---|
| `train_answer_sft.py` | Answer-only SFT; saves the shared merged B1 |
| `train_refusal_sft.py` | Matched ordinary refusal-SFT branch |
| `extract_qwen_baselines.py` | Native greedy answers, first-token logits, last-prompt hidden features |
| `fit_qwen_probes.py` | Fixed-C and cluster-cross-validated logistic probes; configurable `--c-grid` |
| `selector.py` | Correct-answer-loss threshold and answer-preserving NOTA routing |
| `calibrate_selector.py` | T/CM-only frozen policy selection; accepts a selected `--probe` archive |
| `apply_selector.py` | Label-free application to a new logits or hidden-state cache |
| `compare_branches.py` | Shared-B1 comparison, loss/repair decomposition, gain retention |
| `evaluation.py` | Native cache reader, circular metrics, paired image-cluster intervals |
| `x2_score.py` | Deterministic answer/abstain/invalid scoring |
| `mmupd_adapter.py` | MM-UPD TSV access and official parser loading |
| `prepare_images.py` | Map official images to released manifest paths |
| `score_refusal_sft.py` | Ordinary SFT metrics and paired contrasts |
| `score_cached_controls.py` | Fixed-C, maximum probability, and entropy controls |
| `compare_decision_rules.py` | Default versus calibrated decision rules |
| `extract_bicr_black.py`, `fit_bicr.py` | BICR black-image features, fitting, and evaluation |
| `fit_acr_mcq.py` | Raw and standardized ACR-MCQ adaptations |
| `paper_diagnostics.py` | Budget curves and task breakdowns |
| `contrast_working_points.py` | ReFrain versus EmbLoRA paired contrasts |
| `score_confirmation.py` | Frozen independent confirmation and gain retention |
| `verify_results.py` | Recompute published main/confirmation counts from correctness records |

Scenario and source-head scripts are in `scripts/`; InternVL training scripts are in `internvl/`. `results/` contains experiment evidence; `manifests/` specifies the benchmark inputs; `tests/` exercises threshold boundaries, denominator choice, label-free routing, and sample alignment.
