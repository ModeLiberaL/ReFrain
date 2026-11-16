# Data and manifest format

## Two-block adaptation

Unpack the compact record archive from the repository root with `tar -xzf results/two_block_records.tar.gz`. It creates the result and split directories described here.

The current scenario module uses the same fixed record IDs on both backbones. `splits/mmupd_adaptation.json` lists CM-A (2,229 rows), adaptation A (1,354), calibration B (2,338), and the image-cluster fold assignments. With seed 20261116, a two-fold `StratifiedGroupKFold` splits CM by image and UPD type; the first training half is CM-A. A is quota-sampled from CM-A using the InternVL B1 outcomes: 673 initially correct answerable, 4 other answerable, 127 non-abstaining unanswerable, and 550 already-abstained unanswerable rows. The released IDs preserve that exact selection when a reader's retrained B1 has different answers. A has 90 image clusters. All of CM-B is B, including 1,233 answerable rows; a 0.5% budget allows six lost correct answers on B.

`splits/{qwen,internvl}_{A,B}_records.json` provides per-backbone reference correctness and the five A folds. The two backbones share the selected records and folds; their B1 correctness labels differ. A's answerability labels train the residual; B's original-correct labels calibrate the threshold. Neither E nor I enters fitting or calibration.

`results/two_block/{qwen,internvl}/` contains source and scenario heads, frozen thresholds, A/B/E/I score records, official option metadata, and B1 answers. Records distinguish answerability (`std`), initially correct answerable answers (`correct`), remaining unanswerable gain opportunities (`gain`), and whole-task correctness (`base_correct`). Each `prediction::` array is a route's whole-task correctness. Native SFT abstention arrays use `abstain::` keys. Question metadata plus answer text supports rescoring with the official MM-UPD parser; image files are obtained separately.

The full module uses T for source fitting and A for residual fitting. Earlier native sequential and joint SFT controls use T for refusal learning; joint SFT additionally uses the answer-training records. The released component controls share A/B data, while complete-system training routes retain their documented label and exposure budgets.

## Original source-head protocol

| Split | Examples | Purpose |
|---|---:|---|
| T | 1,354 | Probe fitting; 677 Standard examples for answer SFT; 677 Standard/UPD pairs for refusal SFT |
| CM | 4,567 | Independent threshold calibration; 2,406 Standard examples |
| E | 10,678 | Main evaluation; 5,626 Standard and 5,052 UPD examples |
| I | 1,728 | Frozen AI2D-derived confirmation; 912 Standard and 816 UPD examples |

The files in `manifests/` preserve the experiment's prompts, options, labels, ordering, and image-cluster assignments. Image paths are relative to the `--image-root` directory. T has 432 image/pair-connected clusters; CM and E contain 183 and 416 image clusters. The confirmation set has 228 evaluation clusters and uses 271 distinct source images, including mismatched-image construction.

Each JSONL row contains:

| Field | Meaning |
|---|---|
| `sid` | Unique example ID, including split and rotation |
| `split` | T, CM, E, or I |
| `task` | AAD, IASD, or IVQD task identifier |
| `type` | `standard` or `upd` |
| `cluster` | Unit for data partitioning and paired bootstrap |
| `family`, `index` | Original question family and circular-rotation index |
| `pair` | Standard/UPD pairing identifier, when used in training |
| `image` | Image path relative to the selected image root |
| `prompt` | Exact model input text |
| `display` | Ordered pairs of option letter and option text |
| `gold`, `nota` | Correct option and displayed abstention option |

Obtain the source data from [MM-UPD](https://huggingface.co/datasets/MM-UPD/MM-UPD), [CLIP-UP's data instructions](https://github.com/BenVr/CLIP-UP), and [AI2D](https://allenai.org/data/diagrams). The benchmark annotations retain the provenance and terms of those sources; this project's MIT license applies to its code. Images and pretrained weights are not redistributed.

After downloading those sources:

```bash
python scripts/prepare_images.py --output data/images \
  --clipup-images /datasets/clipup/multiple_choice_train_and_val/images \
  --mmupd-root /datasets/MM-UPD \
  --ai2d-images /datasets/ai2d/images
```

The MM-UPD root must contain `data/mmaad_20240303_option.tsv`, `data/mmiasd_20240303_option.tsv`, and `data/mmivqd_20240303_option.tsv`. The CLIP-UP image directory must have the same `coco/`, `visual_genome/`, and other relative paths referenced by T. The utility extracts MM-UPD image bytes and checks that the source prompt agrees with the frozen manifest.

Download [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) to a local model directory. The training/extraction scripts use local files and never download weights implicitly.
