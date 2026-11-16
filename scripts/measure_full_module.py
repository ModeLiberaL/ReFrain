"""Measure paired end-to-end latency of B1 and full two-block ReFrain on Qwen MM-UPD E.

Use the released 500-input fixture and frozen heads, batch one, greedy bfloat16 SDPA generation and three randomized
paired repetitions. Timing includes image loading, processing, generation, decoding, prefill feature capture,
float64 scoring and the veto. Both arms finish generation before the veto. Model loading is excluded.

    python scripts/measure_full_module.py --model runs/b1/sft/B1 --image-root data/images \\
      --inputs results/full_module_cost/inputs --output runs/full_module_cost
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261116


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    temporary.replace(path)


def load_head(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def score(s0, l36, q27, head):
    """Full-module score: s0 + intercept + standardized [final state, question state] @ coef, in float64. The block
    order and arithmetic are those of the research scorer (adaptive_residual/fit.py score_head)."""
    x = np.concatenate([np.asarray(l36, dtype=np.float64), np.asarray(q27, dtype=np.float64)])
    return float(s0 + float(head['intercept']) + ((x - head['mean']) / head['scale']) @ head['coef'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True, help='merged Qwen MM-UPD B1 directory')
    ap.add_argument('--image-root', type=Path, required=True, help='root of the relative image paths in inputs/sample.json')
    ap.add_argument('--inputs', type=Path, default=HERE / 'inputs')
    ap.add_argument('--code', type=Path, default=HERE, help='directory holding extract_qwen_baselines.py')
    ap.add_argument('--output', type=Path, default=HERE / 'run')
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code.resolve()))
    from extract_qwen_baselines import image_for_processor
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output / 'measurements.jsonl').exists(), 'measurements.jsonl exists; use a new --output'
    sample = read(a.inputs / 'sample.json')
    rows, budget = sample['rows'], sample['budget']
    reference = {r['sid']: r for r in read(a.inputs / 'reference.json')['rows']}
    assert len(rows) == 500 and [r['sid'] for r in rows] == sample['sids'] and set(reference) == set(sample['sids'])
    source = load_head(a.inputs / 'source_head.npz')
    head = load_head(a.inputs / 'R1b_head.npz')
    assert head['blocks'].tolist() == ['final_state', 'question_state'] and bool(head['use_source'])
    threshold = float(read(a.inputs / 'policy.json')[budget]['threshold'])
    paths = {r['sid']: (a.image_root / r['image'] if not Path(r['image']).is_absolute() else Path(r['image'])) for r in rows}
    for path in paths.values():
        path.read_bytes()  # warm the page cache; also fails early on a missing image

    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    language = model.model.language_model
    assert len(language.layers) == 36 and language.config.hidden_size == 2048
    assert source['coef'].shape == (model.get_output_embeddings().weight.shape[0],)
    vision_end = model.config.vision_end_token_id
    turn_end = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert processor.tokenizer.convert_ids_to_tokens(turn_end) == '<|im_end|>'
    source_coef, source_intercept = source['coef'].astype(np.float64), float(source['intercept'])
    gpu_state = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,power.limit,clocks.sm,clocks.mem',
                                '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    write(a.output / 'protocol.json', dict(
        dataset='MM_UPD', sample_seed=SEED, sample_source=sample['source'], inputs=len(rows),
        repetitions=a.repetitions, warmup_rows=a.warmup, batch_size=1, dtype='bfloat16', attention='sdpa',
        decoding='greedy', max_new_tokens=a.max_new_tokens, arm_order='numpy default_rng(20261116).permutation per input',
        arms={'B1': 'plain generation, no hooks', 'Full': 'generation with prefill hooks, source score, two-block residual, veto'},
        budget=budget, threshold=threshold, head_coefficients=int(head['coef'].size) + 1,
        source_head_parameters=int(source_coef.size) + 1,
        timing='image load, processor, full generation, decoding and response; Full adds span lookup, hook captures, '
               'float64 scoring and veto', host=socket.gethostname(), gpu=torch.cuda.get_device_name(0), gpu_state=gpu_state,
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, cpu_threads=2,
        model=str(a.model.resolve())))
    write(a.output / 'status.json', dict(state='running', started=time.time()))

    cached, span = {}, {}

    def capture27(module, inputs, output):
        if 'Q27' in cached:
            return
        h = output[0] if isinstance(output, tuple) else output
        cached['Q27'] = language.norm(h[:, span['start']:span['end']]).float().mean(1).detach().cpu().numpy()[0]

    def capture36(module, inputs, output):
        if 'L36' not in cached:
            cached['L36'] = output.last_hidden_state[0, -1].detach().float().cpu().numpy()

    def capture_logits(module, inputs, output):
        if 'raw' not in cached:
            cached['raw'] = output.logits[0, -1].detach().float().cpu().numpy().astype(np.float64)

    def prepare(row):
        image = image_for_processor(paths[row['sid']], processor.image_processor)
        prompt = processor.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=row['prompt'])])],
                                               tokenize=False, add_generation_prompt=True)
        return processor(text=[prompt], images=[image], return_tensors='pt').to('cuda')

    def plain(row):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        inputs = prepare(row)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens)
        tokens = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        response = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return dict(seconds=seconds, generated_tokens=len(tokens), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    response=response), tokens

    def full(row):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        inputs = prepare(row)
        ids = inputs['input_ids'][0]
        begin = int((ids == vision_end).nonzero(as_tuple=True)[0][0]) + 1
        end = next(j for j in (ids == turn_end).nonzero(as_tuple=True)[0].tolist() if j >= begin)
        span.update(start=begin, end=end)
        cached.clear()
        hooks = [language.layers[26].register_forward_hook(capture27), language.register_forward_hook(capture36),
                 model.register_forward_hook(capture_logits)]
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens)
        finally:
            for hook in hooks:
                hook.remove()
        s0 = float(cached['raw'] @ source_coef + source_intercept)
        z = score(s0, cached['L36'], cached['Q27'], head)
        veto = z < threshold
        tokens = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        response = row['nota'] if veto else processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        return dict(seconds=seconds, generated_tokens=len(tokens), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    s0=s0, score=z, veto=bool(veto), response=response, span=[begin, end]), tokens

    arms = {'B1': plain, 'Full': full}
    rng = np.random.default_rng(SEED)
    started, pairs = time.monotonic(), 0
    with (a.output / 'measurements.jsonl').open('w') as log:
        for repetition in range(a.repetitions):
            for row in rows[:a.warmup]:
                for arm in arms.values():
                    arm(row)
            for row in rows:
                order = rng.permutation(['B1', 'Full']).tolist()
                record = dict(dataset='MM_UPD', repetition=repetition, sid=row['sid'], group=str(row['cluster']), order=order)
                tokens = {}
                for name in order:
                    record[name], tokens[name] = arms[name](row)
                if tokens['B1'] != tokens['Full']:
                    write(a.output / 'status.json', dict(state='failed', reason='generated tokens differ', sid=row['sid'],
                                                         repetition=repetition, pairs=pairs))
                    raise SystemExit(f"generated tokens differ: {row['sid']}")
                ref = reference[row['sid']]
                record['check'] = dict(tokens_equal_research_cache=tokens['B1'] == ref['generated_token_ids'],
                                       s0_difference=record['Full']['s0'] - ref['s0'],
                                       score_difference=record['Full']['score'] - ref['score'],
                                       veto_equal_offline=record['Full']['veto'] == ref['veto'])
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                pairs += 1
                if pairs % 100 == 0:
                    log.flush()
                    print(json.dumps(dict(pairs=pairs, seconds=time.monotonic() - started)), flush=True)
    write(a.output / 'status.json', dict(state='complete', paired_records=pairs, seconds=time.monotonic() - started,
                                         host=socket.gethostname()))
    print('complete', pairs, flush=True)


if __name__ == '__main__':
    main()
