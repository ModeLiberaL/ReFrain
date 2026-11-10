# Source index

| File | Role |
|---|---|
| `train_answer_sft.py` | Answer-only SFT; saves the shared merged B1 |
| `train_refusal_sft.py` | Matched ordinary refusal-SFT branch |
| `extract_qwen_baselines.py` | Native greedy answers, first-token logits, last-prompt hidden features |
| `fit_qwen_probes.py` | Fixed-C and cluster-cross-validated logistic probes |
| `selector.py` | Correct-answer-loss threshold and answer-preserving NOTA routing |
| `calibrate_selector.py` | T/CM-only frozen policy selection |
| `apply_selector.py` | Label-free application to a new cache |
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

All Python files are in `scripts/`. `results/` contains experiment evidence; `manifests/` specifies the benchmark inputs; `tests/` exercises threshold boundaries, denominator choice, label-free routing, and sample alignment.
