# ReFrain

**Multimodal abstention with explicit preservation of acquired correct answers**

ReFrain adds a budget-controlled abstention stage to a frozen vision-language answerer. The current method adapts a source answerability score with a regularized residual over two representations: the final prompt state and an intermediate question-token mean. An independent calibration set chooses the threshold. At inference, the selector preserves the original response or emits the task's abstention symbol.

The budget counts previously correct answers removed by abstention, divided by all answerable inputs. Repairs do not cancel losses. It constrains empirical calibration loss; the realized loss on new data is measured separately.

## Main results

Both backbones start from their answer-SFT checkpoint B1. MM-UPD E contains 5,626 answerable and 5,052 unanswerable inputs. I is the AI2D-derived transfer set with 912 answerable and 816 unanswerable inputs. The architecture, heads and thresholds transfer to I without fitting there.

At the common 0.5% **realized-loss ceiling**, entries are additional correct abstentions (original-correct losses):

| Backbone / set | Source head | Final-state target LR | Same-feature target LR | ReFrain |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B / E | 1,204 (27) | 1,612 (26) | 1,811 (27) | 1,754 (28) |
| Qwen2.5-VL-3B / I | 501 (4) | 500 (4) | 579 (3) | 584 (4) |
| InternVL3-8B / E | 352 (28) | 760 (28) | 958 (28) | 897 (28) |
| InternVL3-8B / I | 126 (4) | 193 (4) | 153 (2) | 193 (4) |

At the independently frozen **0.5% calibration budget**, ReFrain gains 1,762/449 abstentions with 32/0 losses on Qwen E/I, and 1,002/193 with 70/5 losses on InternVL E/I. Circular Dual is 69.02/60.84 and 78.97/66.16, respectively. Circular requires all prescribed option rotations to be correct; these scores are distinct from row-level accuracy.

The question block improves both E frontiers over the final-only residual. Source anchoring has a domain-dependent trade-off: removing it improves E ranking, while keeping it gives a stronger transferred Qwen I frozen point (449 versus 306 added abstentions, both at zero losses). All six component controls share the same adaptation/calibration records and current B1 answers.

Native joint SFT reaches Circular Dual 67.53/54.03 on Qwen E/I; sequential SFT reaches 66.41/52.82. ReFrain and joint SFT are not statistically separated on Qwen E. Sequential SFT has the higher InternVL I point estimate (70.57 versus 66.16). Native SFT learns refusal on T; ReFrain additionally uses adaptation labels from A. These complete-system comparisons use a common B1 accounting reference and their respective training data. The controlled representation comparisons use identical A/B data.

## Method

1. Train the answerer and retain B1. Fit a source answerability head on first-token vocabulary logits from T.
2. Freeze B1 and the source head. Add a residual over final and question states, retaining the source score as a fixed offset. Five image-group folds on A select one shrinkage parameter, including the source-ranking limit at infinite shrinkage.
3. On disjoint B, choose the largest A-score threshold whose original-correct loss meets the chosen calibration budget.
4. Preserve each original answer above the threshold; otherwise return its displayed NOTA letter or an open-answer refusal. Deployment reads features and original answers, without target labels.

Changing the budget recalibrates the threshold without retraining the answering model or residual head. Qwen uses layers 36/27; InternVL uses 28/21. Apply the final RMSNorm to individual question tokens before pooling. The two-block residual contains 4,097/7,169 coefficients including its intercept.

## Reproduce the released results on CPU

```bash
pip install -r requirements-scenario.txt
tar -xzf results/two_block_records.tar.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_scenario_records.py
```

This command recomputes component frontiers, frozen points, per-type results, Circular metrics, unified loss/repair counts and paired 95% intervals from compact row records. It uses 2,000 image-cluster bootstrap draws with seed 20261116 and requires no model weights or images.

The portable fitting entry point defaults to the full two-block method:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/scenario_selector.py fit \
  --adaptation A.npz --calibration B.npz --output fitted/R1b
python scripts/scenario_selector.py apply \
  --head fitted/R1b/head.npz --policy fitted/R1b/policy.json --budget 0.005 \
  --features E_features.npz --answers E_answers.jsonl --output E_routed.jsonl
```

See [scenario adaptation](docs/scenario-adaptation.md) for cache formats, extraction, all six controls and budget changes; [reproduction](docs/reproduction.md) for the original answer-training and source-head pipeline; [datasets](docs/data.md) for image preparation and upstream terms; and the [code index](docs/code-index.md).

The [compact record archive](results/two_block_records.tar.gz) unpacks `results/two_block/` and `splits/`: explicit split/fold IDs, both source heads, all six scenario heads and frozen policies, row-level scores and labels, and the B1 answer texts used for scoring. Original source-only results remain in their existing result directories. The VizWiz source-head confirmation is documented separately in [the open-answer reproduction guide](docs/vizwiz-reproduction.md).

## Related work

The implementation builds on [MM-UPD](https://github.com/AtsuMiyai/UPD), [CLIP-UP](https://github.com/BenVr/CLIP-UP), and linear probing for VLM abstention. [TAPS](https://arxiv.org/abs/2610.16318) studies answer training followed by selection. Baseline adaptations in this repository use the common B1 and task-specific abstention format; their results describe our reproduced configurations.

## Limitations

E and I informed architecture development. The study covers two backbones and one answer-training seed. Calibration loss can exceed its nominal budget after transfer: the frozen 0.5% policies incur 0.569%/1.244% loss on Qwen/InternVL E. Source anchoring does not improve every domain. Question pooling depends on the backbone's input template, and the full two-block online latency has not been measured. Source-only TextVQA transfer can falsely refuse answerable questions. Complete-system SFT comparisons differ in training-label use.

## License and data attribution

Code uses the [MIT License](LICENSE). MM-UPD, CLIP-UP data, AI2D, VizWiz and model weights retain their upstream terms. Images and model weights are obtained separately from the official sources; they are not bundled with the row records.
