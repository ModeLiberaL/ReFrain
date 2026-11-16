# Complete-module online cost

On Qwen MM-UPD E, full ReFrain adds **1.414 ms [1.282, 1.556]**, or **1.755% [1.582%, 1.939%]**, to the existing answer pass. Plain B1 and the full module average 80.534 and 81.948 ms. This is a paired measurement on one RTX6000D with bfloat16, SDPA and batch one.

The 500 inputs cover 242 image clusters. Each input is measured three times, with randomized arm order and five warmup inputs per repetition. Image files are read before timing to warm the page cache. Timing includes image loading, processing, full generation, decoding, feature capture, float64 scoring and the final veto. Both arms complete generation. All 1,500 paired token sequences and veto decisions match the research references; maximum score difference is 1.6e-14. Intervals use 2,000 paired image-cluster resamples after averaging the three repetitions within each input.

## Verify the reported measurement on CPU

```bash
pip install -r requirements-scenario.txt
tar -xzf results/full_module_cost_records.tar.gz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_full_module_cost.py
```

The archive includes raw paired observations, reported summaries, frozen heads and the fixed 500-input fixture. Verification reproduces all means, medians, P95 values, paired increments and intervals without a GPU.

## Measure on a GPU

Prepare B1 and images through the main reproduction guide, unpack the fixture, and run from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
python -u scripts/measure_full_module.py --model runs/b1/sft/B1 --image-root data/images \
  --inputs results/full_module_cost/inputs --output runs/full_module_cost
```

`measure_full_module.py` implements the measured online integration. It captures the source logits, final prompt state and normalized question-token mean during the first generation prefill, computes the full score, and routes the response with the frozen 0.5% policy. It writes paired observations, protocol and status. The measurement completes 1,500 pairs and stops if the two arms generate different tokens. Runtime values depend on the hardware and serving environment.

The source-only 1.17% MM-UPD and 0.55% VizWiz measurements used separate sessions. They cannot be subtracted from the complete-module measurement to isolate residual-only latency.
