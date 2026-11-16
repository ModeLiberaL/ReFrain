# InternVL3-8B answer training and ordinary refusal SFT

This guide rebuilds the InternVL3-8B checkpoints used in the paper:

- **B1**: answer SFT from the pinned InternVL3-8B, merged into the weights. The ReFrain source head, the six scenario heads, ordinary refusal SFT and native scoring all start from this one checkpoint.
- **Ordinary refusal SFT**: a second LoRA trained from the merged B1, then merged.

Both stages reuse the Qwen recipes of `scripts/train_answer_sft.py` and `scripts/train_refusal_sft.py`. The training rows come from those scripts' own `answer_rows` and `refusal_rows` functions. The code is in `internvl/`. It runs on one GPU; the data check runs on a CPU.

## Setup

```bash
pip install -r requirements-internvl.txt
git clone https://github.com/BenVr/CLIP-UP external/CLIP-UP
huggingface-cli download OpenGVLab/InternVL3-8B --revision 24dc81a234a6e1901f3314eeadaa2813f2b78038 \
  --local-dir models/InternVL3-8B
```

- **Model revision.** The revision is the one CLIP-UP pins in `clip_up/vlms/internvl/clip_up_internvl_utils.py`. The scripts load only local files.
- **Images.** Prepare them as described in [data.md](data.md). Image paths in the manifests are relative to `--image-root`; absolute paths are used as they are.
- **CLIP-UP.** The code uses only CLIP-UP's InternVL image loader (at most six tiles plus a thumbnail) and its image-token constants. Importing that module also imports CLIP-UP's package dependencies, which is why `requirements-internvl.txt` lists them. The CLIP-UP checkout of the recorded runs carries no Git metadata, so its commit is unknown; the loader is identified by its function and the model revision it pins.
- **Import order.** `internvl/ivl_common.py` imports `transformers.generation` and `transformers.modeling_utils` before CLIP-UP and its scikit-learn dependency, preserving the model-module import order of the recorded environment.

**Recorded environment**
- One NVIDIA GeForce RTX 3090 (24 GB).
- Package versions, read from the training environment:

  | Package | Version |
  |---|---|
  | torch | 2.13.0+cu130 |
  | torchvision | 0.28.0 |
  | transformers | 4.49.0 |
  | numpy | 2.3.5 |
  | scipy | 1.17.1 |
  | peft | 0.10.0 (imported by CLIP-UP only) |
  | accelerate | 0.34.2 |
  | safetensors | 0.8.0 |
  | Pillow | 12.3.0 |
  | timm | 0.9.12 |
  | einops | 0.6.1 |
  | openai-clip | 1.0.1 |

  `requirements-internvl.txt` pins these versions. pandas, scikit-learn and tqdm, which CLIP-UP's package imports, were not queried and are left unpinned.
- bfloat16 weights and eager attention; flash-attn was not installed and is not used.
- Environment variables: `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false`.

## Recipe

| | Answer SFT (B1) | Ordinary refusal SFT |
|---|---|---|
| Starting point | Pinned InternVL3-8B | Merged B1 |
| Rows | `answer_rows`: the 677 Standard rows of T, NOTA option removed, letters remapped | `refusal_rows`: all 1,354 rows of T with the original prompt; a UPD row's target is its NOTA letter |
| LoRA | rank 16, alpha 32, dropout 0.05; float32 adapter | rank 8, alpha 16, dropout 0.05; bfloat16 adapter |
| Optimizer | AdamW (torch default betas/eps), lr 2e-5, weight decay 0.01 | AdamW, lr 4e-5, weight decay 1e-4 |
| Schedule | linear, 16 warmup steps | cosine, 30 warmup steps |
| Steps | 510 (170 per epoch; the last step of an epoch holds 1 row) | 1,017 (339 per epoch; the last step holds one pair) |
| Row order | epoch e uses `numpy.random.default_rng(42 + e).permutation(677)`, e = 0, 1, 2 | `numpy.random.RandomState(42)` permutes the 677 pairs once per epoch; each pair is UPD, then Standard |
| Trainable parameters | 40,370,176 | 20,185,088 |

**Shared by both stages**
- Seed 42 for Python, NumPy and torch. It is set after the model is loaded, so the LoRA initialization and dropout masks do not depend on the loading code.
- Three epochs; one row per forward pass; four consecutive rows per optimizer step; no gradient clipping; gradient checkpointing.
- The adapter after the last epoch is the result: there is no evaluation or checkpoint selection.

**Model and loss**
- LoRA wraps q/k/v/o and gate/up/down in all 28 language-model layers (196 modules). The vision tower and projector are frozen and stay in eval mode.
- Input: the native InternVL conversation template with `<image>\n` and the manifest prompt. This is the same prompt `scripts/extract_scenario_internvl.py` builds.
- Target: the gold letter plus `<|im_end|>`. Only these two tokens are supervised, and the loss is their mean cross-entropy in float32.

**Merge**
- `internvl/ivl_merge.py` computes float32(W) + (alpha/rank)·B·A and rounds once to bfloat16.
- The merge is part of the recipe. In the recorded runs, a large part of the float32 update fell below bfloat16 resolution (relative rounding error 0.79 for B1 and 0.65 for the refusal model). Merging the same adapter another way can therefore give a different checkpoint, so use this script.

**Differences from the Qwen scripts**
- The LoRA layer is implemented in `ivl_common.py`, not with peft. It has peft's layout and initialization (A Kaiming-uniform, B zero) and stores the adapter in each stage's dtype.
- The forward pass runs without autocast: the model is bfloat16, and the LoRA branch is computed in the input's dtype.
- The target has no trailing newline. The two supervised tokens are the same as in the Qwen scripts.
- Only the language model is switched to train mode.

## Commands

Run these from the repository root. Each GPU stage refuses an output directory that holds an unfinished run; use a new directory instead of resuming.

**1. Data and order check (CPU, no model)**

```bash
python internvl/ivl_recipe.py --mode answer --manifest manifests/T.jsonl --output runs/internvl/answer_data.json
python internvl/ivl_recipe.py --mode refusal --manifest manifests/T.jsonl --output runs/internvl/refusal_data.json
```

- The report must equal the recorded run's report in `internvl/recorded_run.json`: rows, gold letters, option counts, and the first rows and indices of every epoch.
- If you trained the Qwen stages first, `--qwen-run runs/b1/sft` (answer) or `--qwen-run runs/b1/refusal_sft` (refusal) also compares rows, recipe values, records per step and learning rates with that run.

**2. Answer SFT and B1**

```bash
python internvl/ivl_check_masking.py --mode answer --manifest manifests/T.jsonl \
  --heldout manifests/CM.jsonl manifests/E.jsonl --image-root data/images \
  --model models/InternVL3-8B --clip-up-root external/CLIP-UP \
  --output runs/internvl/answer_masking.json --gpu-rows 8
python internvl/ivl_train_sft.py --mode answer --manifest manifests/T.jsonl \
  --heldout manifests/CM.jsonl manifests/E.jsonl --image-root data/images \
  --model models/InternVL3-8B --clip-up-root external/CLIP-UP --output runs/internvl/answer_sft
python internvl/ivl_merge.py --role b1 --base models/InternVL3-8B \
  --adapter runs/internvl/answer_sft/adapter --output runs/internvl/B1 \
  --manifest manifests/T.jsonl --image-root data/images --clip-up-root external/CLIP-UP --check-rows 32
```

**3. Ordinary refusal SFT**

```bash
python internvl/ivl_check_masking.py --mode refusal --manifest manifests/T.jsonl \
  --heldout manifests/CM.jsonl manifests/E.jsonl --image-root data/images \
  --model runs/internvl/B1 --clip-up-root external/CLIP-UP \
  --output runs/internvl/refusal_masking.json --gpu-rows 8
python internvl/ivl_train_sft.py --mode refusal --manifest manifests/T.jsonl \
  --heldout manifests/CM.jsonl manifests/E.jsonl --image-root data/images \
  --model runs/internvl/B1 --clip-up-root external/CLIP-UP --output runs/internvl/refusal_sft
python internvl/ivl_merge.py --role refusal --base runs/internvl/B1 \
  --adapter runs/internvl/refusal_sft/adapter --output runs/internvl/refusal_model \
  --manifest manifests/T.jsonl --image-root data/images --clip-up-root external/CLIP-UP --check-rows 32
```

- **Masking check.** `ivl_check_masking.py` is an optional read-only check on T; it takes no optimizer step. It verifies that:
  - training ids start with the inference-prompt ids;
  - only the letter and end token are supervised;
  - the image-token count is right (256 per tile);
  - a zero-initialized LoRA leaves the first-token logits unchanged;
  - the loss agrees with two independent computations;
  - after one backward pass, only the LoRA matrices have gradients.
- **Smoke test.** `ivl_train_sft.py --smoke-steps 2` reports seconds per step and peak memory without writing an adapter.
- **Start checkpoints.** The answer stage refuses a merged checkpoint as `--model`. The refusal stage requires the merged B1.

After step 2, `runs/internvl/B1` is the `--model` of `scripts/extract_scenario_internvl.py` (see [scenario-adaptation.md](scenario-adaptation.md)).

## Outputs

| Path | Content |
|---|---|
| `answer_sft/`, `refusal_sft/` | `config.json` (recipe, data report, comparison with the recorded run, environment), `train_rows.jsonl`, `train_log.jsonl` (one line per step: loss, lr, gradient norm, longest sequence, seconds, peak memory), `adapter/` (`adapter.safetensors` in float32 plus `adapter.json`), `complete.json` |
| `B1/`, `refusal_model/` | Merged bfloat16 weights, an unchanged copy of the base directory's configuration, remote code and tokenizer files, and `ivl_merge.json` (rounding statistics, reload check, and a comparison of base, adapter-attached and merged first-token logits on 32 T rows) |

Each merged checkpoint takes about 16 GB.

## Recorded runs

`internvl/recorded_run.json` documents the runs behind the released InternVL results; machine paths are removed. It contains:
- the data reports;
- the masking-check and smoke-test summaries;
- the training summary: steps, seconds, peak memory and per-epoch mean loss;
- the merge statistics.

On the RTX 3090:
- Answer SFT: 510 steps in 4,238 s, peak 18.2 GiB.
- Refusal SFT: 1,017 steps in 8,816 s, peak 17.7 GiB.
- Reloading each merged checkpoint reproduced the in-memory logits exactly.
- On the 32 check rows, the merged model and the adapter-attached model had the same first-token argmax.

## Scope

- **Not bitwise reproducible.** GPU kernels and library builds differ, so a rerun is not guaranteed to reproduce these weights bit for bit. The data order, recipe, learning rates and merge procedure are fixed and checked.
- **Native answers.** The B1 answers in the released records came from greedy generation (batch 1, at most 1,024 new tokens), with the loading, template and tiling used here. The released records contain these answer texts, and `scripts/extract_scenario_internvl.py` extracts the features from a rebuilt B1; a full-generation script is not part of this release.
- **Fixed counts.** The scripts keep the experiment's fixed row counts (1,354 / 677). For another dataset, adapt those assertions deliberately.
