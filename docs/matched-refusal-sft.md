# Matched-data ordinary refusal SFT (T + A)

**Why this control exists.** The full ReFrain module uses three sets of labels:
- T trains the source head;
- the adaptation split A trains the residual;
- B selects the threshold.

The original ordinary refusal-SFT baseline trains on T. This control also gives it the A labels: it trains on T + A, from the same Qwen B1, with the recipe of `scripts/train_refusal_sft.py` unchanged. B selects the gate threshold and is not used for model training.

## Data

**`manifests/TA.jsonl`** (2,708 rows)
- The 1,354 rows of `manifests/T.jsonl`, unchanged and in manifest order.
- Then the 1,354 adaptation rows. These follow the `train_sid` order of `splits/mmupd_adaptation.json` and are taken from `manifests/CM.jsonl` with one added field, `pair`:
  - 395 natural Standard/UPD pairs share task and index (`cm_a/<task>/<index>`);
  - the remaining 282 Standard and 282 UPD rows were sorted by sid, permuted separately per type with `numpy.random.default_rng(20261117)`, and zipped (`cm_a/seeded/<k>`).
- The file fixes the row order. The row order fixes the pair indices, and the pair indices fix the training order.

**`manifests/B.jsonl`** (2,338 rows)
- The calibration rows, in the `cm_b_ids` order of the split file, with no added fields.
- Here it serves only for the held-out image check and for scoring.

**Images**
- Image paths are relative to `--image-root`, as in the other manifests.
- No image of T + A appears in B, E or I.

## Recipe

The recipe is that of `scripts/train_refusal_sft.py`:

| Setting | Value |
|---|---|
| Starting point | `runs/b1/sft/B1` |
| LoRA | rank 8, alpha 16, dropout 0.05, on the language model's q/k/v/o and gate/up/down projections |
| Optimizer | AdamW, learning rate 4e-5, weight decay 1e-4 |
| Schedule | cosine, `int(0.03 × total steps)` warmup steps |
| Epochs and batching | 3 epochs; one row per forward pass; four rows per optimizer step |
| Seed | 42 |
| Precision | gradient checkpointing; bfloat16 autocast |
| Target | the correct option letter (NOTA for UPD rows) plus `<|im_end|>` |
| Pair order | numpy's global generator, seeded with 42, permutes the 1,354 pairs once per epoch; each pair contributes its UPD row, then its Standard row |
| Merge | peft `merge_and_unload`, as in the published script |

**Budget**
- 2,031 optimizer steps (677 per epoch), with 60 warmup steps.
- The T-only run takes 1,017 steps.

## Command

```bash
python scripts/train_refusal_sft_ta.py --manifest manifests/TA.jsonl \
  --heldout manifests/B.jsonl manifests/E.jsonl manifests/I.jsonl --image-root data/images \
  --model runs/b1/sft/B1 --output runs/b1/refusal_sft_TA
```

**The entry point.** `scripts/train_refusal_sft_ta.py` is the script that trained the recorded run, with only its docstring changed. `scripts/train_refusal_sft.py` stays as published for the T-only run. The two differ only in:
- the row check: 2,708 rows and 1,354 pairs read from `pair`, instead of T's 1,354 rows and 677 pairs;
- three data-description fields of `config.json` (`sampling`, `data`, `a_pairing`); `a_pairing` names the internal preparation script, whose rule is given under Data above.

**Outputs.** The same as the published script: `config.json`, `train_refusal.jsonl`, `train.jsonl`, `status.json`, `adapter/`, `merge_check.json`, and the merged `model/`.

## Consistency with the recorded run

**Rows and schedule**
- `manifests/TA.jsonl` equals the recorded run's training rows field by field, except that image paths are release-relative.
- The entry point gives the same pair indices as the recorded run, and therefore the same schedule:
  - 2,031 steps, 60 of them warmup;
  - four rows in every step;
  - the same order in each epoch.
- The first rows of each epoch are:

  | Epoch | First pair (UPD, then Standard) | Second pair (UPD, then Standard) |
  |---|---|---|
  | 1 | `T/clip_up/train/aad/upd/18077`, `T/clip_up/train/aad/standard/18077` | `CM/opt/mmupd/iasd/option/upd/3001442`, `CM/opt/mmupd/iasd/option/standard/3001442` |
  | 2 | `CM/opt/mmupd/aad/option/upd/2000687`, `CM/opt/mmupd/aad/option/standard/2000687` | `T/clip_up/train/ivqd/upd/588519`, `T/clip_up/train/ivqd/standard/588519` |
  | 3 | `CM/opt/mmupd/iasd/option/upd/3001718`, `CM/opt/mmupd/iasd/option/standard/1000935` | `CM/opt/mmupd/aad/option/upd/1452`, `CM/opt/mmupd/aad/option/standard/1452` |

  The first pair of epoch 3 is a seeded pair.

**Script**: `scripts/train_refusal_sft_ta.py` equals the recorded run's training script except for the docstring.

**Environment**: the recorded run uses one RTX 6000D and the Qwen environment (PyTorch 2.13.0 with CUDA 13.0, Transformers 5.17.0, peft 0.21.2, NumPy 2.3.5); its `config.json` records the exact versions.

## Evaluation

**Native outputs.** `scripts/extract_sft_scores.py` is the extractor used by the recorded run and the joint-SFT control. It generates native greedy answers (batch 1, at most 1,024 new tokens) and raw first-token A–F logits from the merged model. Run it for E, I, A and B:

```bash
for split in E I A B; do
  python scripts/extract_sft_scores.py --input manifests/$split.jsonl \
    --image-root data/images --model runs/b1/refusal_sft_TA/model \
    --output runs/b1/refusal_sft_TA_scores/$split
done
```

**Gate score**
- The score keeps an answer by −(NOTA logit − largest logit among the other displayed letters).
- It is applied to B1's answers, not to the SFT model's answers.
- Its threshold follows the scenario module's rule: candidates are the A scores plus ±∞, and the threshold is calibrated on B at budgets 0 and 0.5%.
- The gate is compared with the source head and the two-block heads:
  - at the same realized loss;
  - at frozen thresholds;
  - with 2,000 paired image-cluster bootstrap draws (seed 20261116).

## Recompute the released statistics

```bash
tar -xzf results/two_block_records.tar.gz
tar -xzf results/matched_sft_records.tar.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_matched_sft.py
```

This recomputes the gate score from raw letter logits, its A/B calibration, native loss/repair ledgers, Circular metrics, exact/common-loss comparisons, frozen points and paired intervals. The second archive also includes the native E/I answer texts. The reference archive provides shared B1 outcomes and option-rotation groups.

To independently parse the released native text, use the official UPD checkout prepared for the main pipeline and install pandas:

```bash
python scripts/parse_circular_answers.py --manifest manifests/E.jsonl \
  --answers results/matched_sft/E_answers.jsonl --upd-root external/UPD \
  --output runs/parsed_TA_E.npz
```
