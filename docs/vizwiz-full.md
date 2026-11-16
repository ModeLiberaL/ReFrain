# Complete-module confirmation on VizWiz

This experiment evaluates the complete two-block method on open-ended questions. The answer model is Qwen2.5-VL-3B after answer SFT on 4,000 answerable VizWiz examples. Its source head was trained on 6,000 separate questions. The existing 3,000-question calibration pool is split into A/B, 1,500 each, with seed 20261117. The architecture, regularization selection rule and calibrated thresholds were fixed before extracting H features.

H contains 4,000 images/questions: 3,043 answerable questions, 1,630 originally correct answers, and a gain pool of 602 unanswerable inputs not already rejected by B1. It was previously used to confirm source-head regularization and was not used to choose the two-block architecture. This is an in-domain confirmation with one backbone and one answer-training seed.

## Results

Entries are additional correct abstentions (original-correct losses). The frontier uses the common 0.5% realized-loss ceiling, allowing at most 15 losses. Frozen columns apply the independently calibrated B threshold.

| Score | Frontier | Frozen budget 0 | Frozen budget 0.5% |
|---|---:|---:|---:|
| Source head | 167 (15) | 59 (0) | 197 (23) |
| Final-state residual | 163 (14) | 68 (0) | 204 (22) |
| Same-feature target LR | 170 (15) | 42 (1) | 162 (13) |
| ReFrain | 195 (14) | 73 (0) | 216 (18) |
| MSP | 112 (15) | — | — |
| Negative entropy | 116 (15) | — | — |
| Logit probe, C=1 | 106 (14) | — | — |

The pre-specified primary contrast is ReFrain minus its source head: +4.65 conversion points, paired 95% interval [0.17, 9.09]. At exactly 15 common losses, the gains are also 195 versus 167. Frozen zero-budget gains differ by +2.33 points [0.83, 4.03], with zero loss in both arms. The frozen 0.5% point gives +3.16 points [1.29, 5.00] with five fewer losses. The full head selects C=1e-4 (lambda=10,000).

## Recompute from row records

```bash
pip install -r requirements-scenario.txt
tar -xzf results/vizwiz_full_records.tar.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_vizwiz_full.py
```

The archive contains H scores and outcome masks for all seven methods, four frozen residual heads, calibrated policies, full frontier counts, and expected comparisons. The command reproduces all four loss ceilings, exact common-loss counts, frozen points and the 2,000-draw paired intervals. It needs no images or model weights. One image contributes one question; bootstrap seed is 20261116.

The source-only study in [vizwiz-reproduction.md](vizwiz-reproduction.md) uses the full CM for calibration. This complete-module study uses only B for calibration and A scores as candidate thresholds, so their source frozen points differ: 194 gains/21 losses in the former versus 197/23 here. Their source frontier at 15 losses is identical. The C=1 probe here also differs from the earlier cross-validated probe.

## Scope

Only the pre-specified 0.5% full-minus-source frontier contrast is statistically separated; other ceilings and the question-block/source-offset contrasts are reported in the records. The full module's frozen 0.5% point has 0.592% realized loss on H. The calibration budget is an empirical constraint on B, not a guarantee on new questions.
