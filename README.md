# ReFrain

**Multimodal Abstention That Preserves Acquired Correct Answers**

ReFrain (Regression-Free Abstention) adds a calibrated abstention selector to a frozen vision-language answerer. It fits a linear probe on first-token vocabulary logits, constrains the loss of originally correct answers on an independent calibration split, and either preserves the original response or emits the displayed abstention option.

The main experiment starts from one answering-SFT checkpoint, **B1**, shared by every subsequent abstention branch. This makes it possible to measure both additional rejection ability and retention of the correct answers acquired during answering SFT.

## Results

Shared B1, Qwen2.5-VL-3B, MM-UPD evaluation: 5,626 Standard and 5,052 unsolvable examples. Percentages are example-level accuracy. Loss and repair counts are relative to B1 on Standard examples.

| Method | Standard ↑ | UPD ↑ | Correct answers lost ↓ | Answers repaired ↑ | SFT gains retained ↑ |
|---|---:|---:|---:|---:|---:|
| B1 | 87.22 | 54.63 | 0 | 0 | 57/57 |
| ReFrain, 0% calibration budget | 87.22 | 61.94 | 0 | 0 | 57/57 |
| ReFrain, 0.5% calibration budget | 87.08 | 74.05 | 8 | 0 | 55/57 |
| CLIP-UP EmbLoRA | 86.85 | 64.65 | 62 | 41 | 40/57 |
| Ordinary refusal SFT | 85.32 | 87.47 | 140 | 33 | 23/57 |

At the 0.5% calibration operating point, ReFrain improves UPD accuracy over EmbLoRA by **9.40 percentage points**, with **8 vs. 62** lost correct answers. Ordinary refusal SFT attains higher UPD accuracy while losing more correct answers.

On the independent, image-disjoint AI2D-derived confirmation set, the frozen 0.5% policy reaches **66.05% UPD accuracy**, versus **33.33%** for EmbLoRA, with **0 vs. 7** lost correct answers. It retains all **8/8** answering-SFT gains. Models, probe, and thresholds are frozen before confirmation inference.

Full precision results, paired image-cluster intervals, component controls, and budget curves are in [results/](results/). The result tables can be recomputed from the released per-example correctness records with `python scripts/verify_results.py`.

## Method

1. Train the answerer once and save B1. Clone that checkpoint for the comparison branches.
2. Freeze B1. Fit an L2-regularized logistic probe on raw first-token vocabulary logits, using Standard/UPD labels on T. Choose regularization by five-fold image-cluster validation.
3. On CM, choose the largest threshold from the T-score grid whose lost-correct count divided by the number of Standard examples is at most the prescribed budget.
4. Keep the original response when its score is at least the threshold; otherwise return that example's NOTA letter. Deployment does not use target labels.

The loss budget is **not** cross-entropy loss, the error rate among answered examples, or net accuracy loss. It counts previously correct Standard answers that the added abstention stage removes; newly repaired answers do not offset that count.

## Installation and reproduction

Python 3.10+ is recommended. CPU analysis:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/verify_results.py
```

GPU training and extraction use the recorded PyTorch, Transformers, and PEFT versions in `requirements-gpu.txt`. Install the appropriate CUDA build of PyTorch for the target server. The original run used a 72 GB GPU; CPU analysis does not load model weights.

See [reproduction commands](docs/reproduction.md), [datasets and manifests](docs/data.md), and [source index](docs/code-index.md). The release includes the main Qwen pipeline, ordinary refusal SFT, cached confidence controls, BICR and ACR-MCQ adaptations, frozen confirmation scoring, and analysis scripts. CLIP-UP model training uses the upstream implementation and the matched recipe described in the reproduction guide.

## Related work

[TAPS](https://arxiv.org/abs/2610.16318) also separates answer training from post-hoc selection. ReFrain studies multimodal unsolvability under a correct-answer-loss budget, with a shared adapted checkpoint and per-example tracking of acquired-answer retention. [The comparison](docs/taps-comparison.md) describes the substantial common structure and the differences in features, supervision, operating-point selection, and evaluation.

The work builds on [MM-UPD](https://github.com/AtsuMiyai/UPD), [CLIP-UP](https://github.com/BenVr/CLIP-UP), and linear probing for LVLM abstention. BICR and ACR controls are adaptations to the common B1 and NOTA evaluation setting; their results are not the original papers' benchmark scores.

## Limitations

The calibration budget is an empirical constraint, not a population guarantee. A zero-loss observation does not guarantee zero loss on future inputs. The post-SFT study uses one answering-SFT seed; retained-gain sets contain 57 examples on E and 8 on confirmation. E was used during method development; the separate AI2D-derived set provides frozen confirmation. TAPS and ReFrain have not been compared on a common benchmark, so their headline scores do not establish a performance ranking. The first release covers the Qwen pipeline; it does not include every exploratory backbone integration or model checkpoint.

## License and data attribution

Project code is released under the [MIT License](LICENSE). Annotation manifests derive from MM-UPD, the CLIP-UP training data, and AI2D; those datasets and model weights remain subject to their upstream terms. See [data attribution](docs/data.md). Images and model weights are obtained separately from the official sources.
