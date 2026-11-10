# Reproduction

Run GPU commands on the training server. Paths below are examples supplied by the operator. Commands assume the repository is the current directory and the images described in [data.md](data.md) have been prepared.

```bash
pip install -r requirements-gpu.txt
git clone https://github.com/AtsuMiyai/UPD third_party/UPD
```

The scorer imports only the official pure option-parsing functions; no remote judge is used.

## Shared answering checkpoint

```bash
python scripts/train_answer_sft.py \
  --manifest manifests/T.jsonl --heldout manifests/CM.jsonl manifests/E.jsonl \
  --image-root data/images --model models/Qwen2.5-VL-3B-Instruct \
  --output runs/b1/sft
```

The script removes the NOTA option from Standard training examples and trains for three epochs with seed 42, LoRA rank 16, alpha 32, dropout 0.05, learning rate 2e-5, effective batch size 4, and a linear schedule. It saves one merged model at `runs/b1/sft/B1`. Every abstention branch loads this same checkpoint.

## Cache native responses and probe inputs

```bash
for split in T CM E; do
  python scripts/extract_qwen_baselines.py \
    --input manifests/${split}.jsonl --image-root data/images \
    --model runs/b1/sft/B1 --output runs/b1/qwen_base/${split} \
    --batch-size 1 --max-new-tokens 1024
done
python scripts/extract_qwen_baselines.py \
  --input manifests/E.jsonl --image-root data/images \
  --model models/Qwen2.5-VL-3B-Instruct --output runs/b0/qwen_base/E \
  --batch-size 1 --max-new-tokens 1024
```

The first generation forward pass supplies raw vocabulary logits and the final prompt hidden state. Greedy generation supplies the unchanged native response. These hidden states are **last-prompt** states; they are not TAPS's last-response states.

## Fit, calibrate, and apply

```bash
python scripts/fit_qwen_probes.py \
  --cache runs/b1/qwen_base/T runs/b1/qwen_base/CM runs/b1/qwen_base/E \
  --feature logits --output runs/b1/probe
python scripts/calibrate_selector.py --root runs/b1 \
  --upd-code-root third_party/UPD --budget 0.005 --output runs/b1/policy.json
python scripts/apply_selector.py --cache runs/b1/qwen_base/E \
  --probe runs/b1/probe/train_cv.npz --policy runs/b1/policy.json \
  --output runs/b1/refrain_E.jsonl
python scripts/compare_branches.py --root runs/b1 --reference runs/b0 \
  --upd-code-root third_party/UPD
```

Only T labels enter probe fitting and cross-validation. CM correctness enters calibration. Evaluation labels enter reporting only. Ties are retained (`score >= threshold`); the candidate grid includes negative and positive infinity. The 0.5% budget divides lost correct answers by **all Standard calibration examples**, not by the subset B1 answers correctly.

`compare_branches.py` reports the two prescribed budgets, their realized losses, independent losses/repairs, SFT-gain retention, and 2,000 image-cluster bootstrap intervals. To include EmbLoRA, pass `--clipup path/to/EmbLoRA_E_answers.jsonl`. Each line must contain `sid` and `response`. In the result files, R0 means B0, R1 means shared B1, R2 means ReFrain, and R3 means EmbLoRA. Legacy `passes_original_rule` fields additionally require the lower 95% interval bound for Standard change to exceed -1 percentage point and the UPD-change lower bound to exceed zero. This is separate from the 0.5% calibration budget.

## Ordinary refusal SFT and controls

```bash
python scripts/train_refusal_sft.py \
  --manifest manifests/T.jsonl --heldout manifests/CM.jsonl manifests/E.jsonl \
  --image-root data/images --model runs/b1/sft/B1 --output runs/b1/refusal_sft
python scripts/extract_qwen_baselines.py \
  --input manifests/E.jsonl --image-root data/images \
  --model runs/b1/refusal_sft/model --output runs/b1/refusal_sft/E --batch-size 1
python scripts/score_refusal_sft.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/score_cached_controls.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/compare_decision_rules.py --root runs/b1 --upd-code-root third_party/UPD
```

For the full paired comparisons in `score_refusal_sft.py`, first supply EmbLoRA predictions to `compare_branches.py`. Ordinary refusal SFT uses the same B1, T data, three epochs, rank 8, alpha 16, dropout 0.05, learning rate 4e-5, weight decay 1e-4, and cosine schedule as the matched EmbLoRA LoRA recipe. It uses option-format token supervision without CLIP injection.

The EmbLoRA baseline uses [BenVr/CLIP-UP](https://github.com/BenVr/CLIP-UP): Structure-CLIP correlations, embedding injection, and injected LoRA. The paper's Qwen integration loads B1, fits on the same 677 T pairs, and evaluates all displayed rotations with deterministic parsing. That integration and its checkpoints are not part of this initial source release; the published comparison records are included.

BICR and ACR-MCQ adaptations:

```bash
python scripts/extract_bicr_black.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/fit_bicr.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/fit_acr_mcq.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/fit_acr_mcq.py --root runs/b1 --upd-code-root third_party/UPD --standardize
python scripts/paper_diagnostics.py --root runs/b1 --upd-code-root third_party/UPD
python scripts/contrast_working_points.py --root runs/b1 --upd-code-root third_party/UPD
```

BICR implements a correctness head and black-image ranking regularization, with a T-only image-cluster validation split, 50 Optuna trials, and full-T refit. Its source is [Ledengary/BICR](https://github.com/Ledengary/BICR). ACR-MCQ uses hidden features and option logits with risk/gating heads and is an adaptation to the common multiple-choice executor. Details and selected settings are in the corresponding result JSON files.

## Frozen confirmation

Use `manifests/I.jsonl` after freezing B1, the probe, both thresholds, and comparison checkpoints. `apply_selector.py` accepts the new cache without refitting or reading its gold labels. `score_confirmation.py --root runs/confirmation --upd-code-root third_party/UPD` expects `confirm.jsonl`, `B1/`, `EmbLoRA/`, `ordinary_SFT/` caches, `B0_standard/answers.jsonl`, `frozen_probe.npz`, and `frozen_models_thresholds.json`. The latter contains the `calibration` object created by the main comparison. Freeze the **new run's** thresholds; the numerical thresholds in `results/frozen_thresholds.json` belong to the original run's probe.

## Resource use and scope

Raw full-vocabulary caches use approximately `4 * examples * vocabulary_size` bytes plus hidden states. Train and extract on a GPU host; probe fitting and result analysis can run on CPU. The release preserves fixed experiment row counts in the two SFT scripts. To study another dataset, adapt those protocol assertions and create image-disjoint manifests deliberately.
