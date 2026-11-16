# Two-block scenario adaptation

ReFrain keeps the answering model fixed. A source answerability score is adapted with a regularized residual over the final prompt state and pooled question states. A separate calibration set chooses the veto threshold against an original-correct-loss budget.

The CPU implementation uses Python 3.12, NumPy 2.3.5 and SciPy 1.17.1:

```bash
pip install -r requirements-scenario.txt
tar -xzf results/two_block_records.tar.gz
```

## Cache format

Each `.npz` file has one row per image–question input. All arrays share the same row order.

| Key | Shape | Meaning |
|---|---|---|
| `sid` | N | Unique string sample IDs |
| `source_score` | N | Frozen source-head logit; higher means more answerable |
| `final_state` | N × d | Final normalized state at the last prompt token |
| `question_state` | N × d | Question-token mean at relative layer depth 0.75, after per-token final RMSNorm |
| `image_state` | N × d | Same-layer image-token mean, used only by R1v |
| `groups` | N | Image-group IDs; needed for fitting |
| `std` | N | Answerability labels; needed in A and B |
| `folds` | N | Fixed image-group fold IDs 0–4; needed only in A |
| `correct` | N | Reference-answer correctness; needed only in B |

Use float32 feature blocks and float64 source scores. A and B must contain disjoint image groups. Every training fold must contain both answerability classes. Standardization is fitted within each training fold; validation uses that fold's statistics. The final head uses statistics from all of A.

Qwen2.5-VL-3B uses final layer 36 and question layer 27, each of dimension 2,048. InternVL3-8B uses layers 28 and 21, each of dimension 3,584. Question pooling starts after the image-end marker and ends before the user-turn separator, so it includes the question's instruction. Apply the backbone's final RMSNorm to each selected token before taking a float32 mean. The fixed layer fraction and pooling rule are shared by both backbones.

## Fit and freeze

For Qwen2.5-VL-3B, extract one split from a JSONL manifest containing `sid`, `image`, and `prompt`:

```bash
python scripts/extract_scenario_qwen.py --manifest CM.jsonl --image-root data/images --model checkpoints/B1 \
  --head results/two_block/qwen/source_head.npz --output features/CM
python scripts/pack_scenario_cache.py --features features/CM \
  --records splits/qwen_A_records.json --image V27.npy --output A.npz
python scripts/pack_scenario_cache.py --features features/CM \
  --records splits/qwen_B_records.json --image V27.npy --output B.npz
```

The A/B record lists specify sample order, image group, answerability, reference correctness, and A's fold assignment. For inference, omit `--records` to pack an unlabeled feature cache. The Qwen extraction script uses the repository's `extract_qwen_baselines.py` image preprocessing. The source head acts on raw first-generation-position logits. Its training script accepts the explicit candidate grid `--c-grid 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2`.

InternVL extraction uses CLIP-UP's image loader from the upstream repository:

```bash
python scripts/extract_scenario_internvl.py --manifest CM.jsonl --model checkpoints/InternVL_B1 \
  --head results/two_block/internvl/source_head.npz --clip-up-root external/CLIP-UP \
  --image-root images --output features/internvl_CM
python scripts/pack_scenario_cache.py --features features/internvl_CM \
  --final L28.npy --question Q21.npy --image V21.npy \
  --records splits/internvl_A_records.json --output internvl_A.npz
```

InternVL manifests use image paths relative to `--image-root`, or absolute paths. The recorded extractor uses six image tiles at most and the backbone's native conversation template. It obtains the first prefill through a native one-token generation. The recorded Qwen GPU stack is PyTorch 2.13.0 with CUDA 13.0 and Transformers 5.17.0; InternVL uses the same PyTorch build with Transformers 4.49.0 and the CLIP-UP image loader. CPU fitting uses the separate requirements above.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/scenario_selector.py fit \
  --arm R1b --adaptation A.npz --calibration B.npz --output fitted/R1b
```

The objective is summed binary cross-entropy plus `lambda * ||v||² / 2`; the intercept is unpenalized. `lambda=1/C`. Five image-group folds choose C by mean fold BCE from nine finite candidates, with ties preferring stronger shrinkage. R1b also includes infinite shrinkage, which fits an intercept while preserving the source ranking.

The output contains `head.npz`, fold scores in `selection.json`, `policy.json`, and `calibration_frozen.json`. Defaults create policies at budgets 0 and 0.005. Calibration chooses the largest A-score candidate, including both infinities, whose B loss is at most `budget * number_of_answerable_B_rows`. A veto is strictly `score < threshold`; tied scores are kept.

## Change the budget without fitting

```bash
python scripts/scenario_selector.py calibrate \
  --head fitted/R1b/head.npz --adaptation A.npz --calibration B.npz \
  --budgets 0 0.0025 0.005 0.01 0.02 --output policies.json
```

This command keeps the fitted coefficients fixed and reads only A and B. The budget limits empirical calibration loss; evaluation loss is measured separately.

## Apply without evaluation labels

`answers.jsonl` contains `sid` and the original `response`. For multiple-choice tasks it also contains each question's displayed `nota` letter. Open-ended evaluation uses `unanswerable` by default, configurable with `--abstention`.

```bash
python scripts/scenario_selector.py apply \
  --head fitted/R1b/head.npz --policy fitted/R1b/policy.json --budget 0.005 \
  --features E_features.npz --answers E_answers.jsonl --output E_routed.jsonl
```

The application cache needs only IDs, source scores and feature blocks. It contains no target labels. Each output records the score, decision and final response; kept responses preserve the original text.

## Component controls

| Arm | Representation | Source offset | Infinite-shrinkage candidate |
|---|---|---|---|
| R0 | Source score | Yes | Only candidate |
| R1 | Final state | Yes | Yes |
| R1b | Final + question | Yes | Yes |
| R2 | Final state | No | No |
| R2b | Final + question | No | No |
| R1v | Final + image | Yes | Yes |

All controls use the same A/B records, folds and threshold rule. R2b retains an unpenalized intercept. Frontier comparisons use evaluation labels to describe attainable trade-offs; deployed policies use the independently frozen B thresholds.
