"""Extract final and question-segment states from frozen Qwen2.5-VL-3B.
One prefill stores L36, Q27, V27, source scores, token spans, and first-token diagnostics.
Images and prompts are supplied by a JSONL manifest; answers are generated separately.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    temp.replace(path)


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='data', help='Name recorded for this feature split.')
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--head', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, help='Directory prepended to relative image paths in the manifest.')
    ap.add_argument('--code', type=Path, default=Path(__file__).resolve().parent,
                    help='Directory holding extract_qwen_baselines.py.')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--archived-answers', type=Path, help='archived answers.jsonl for the live first-token check')
    ap.add_argument('--archived-scores', type=Path, help='archived {CM,H}_scores.npz for the live source-score check')
    a = ap.parse_args()
    sys.path.insert(0, str(a.code.resolve()))
    from extract_qwen_baselines import image_for_processor  # noqa: E402
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(20261117)

    rows = jsonl(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    assert len({r['sid'] for r in rows}) == len(rows)

    def image_path(row):
        path = Path(row['image'])
        return a.image_root / path if a.image_root and not path.is_absolute() else path

    for r in rows:
        assert image_path(r).is_file(), (r['sid'], str(image_path(r)))
    a.output.mkdir(parents=True, exist_ok=True)
    model_dir = a.model.resolve()
    weights = sorted(model_dir.glob('*.safetensors'))
    assert weights, 'no safetensors in --model'
    binding = dict(split=a.split, rows=[dict(sid=r['sid'], image=str(image_path(r)), prompt=r['prompt']) for r in rows],
                   model=str(model_dir), model_files={p.name: p.stat().st_size for p in sorted(model_dir.iterdir()) if p.is_file()},
                   model_config=json.loads((model_dir / 'config.json').read_text()),
                   head=file_record(a.head), manifest=file_record(a.manifest), limit=a.limit, layer_index=26,
                   feature_blocks=['L36', 'Q27', 'V27'],
                   pooling='per-token final RMSNorm of layer index 26 output, float32 mean over the question segment',
                   template='user content [image, prompt]; add_generation_prompt=True; batch_size=1; prefill only')
    if (a.output / 'inputs.json').exists():
        assert json.loads((a.output / 'inputs.json').read_text()) == binding, 'inputs differ from the recorded binding'
    else:
        write_json(a.output / 'inputs.json', binding)
    state = (json.loads((a.output / 'status.json').read_text()) if (a.output / 'status.json').exists()
             else dict(state='running', completed=0, total=len(rows), seconds=0.))
    if state['completed'] == len(rows):
        print('complete', a.output)
        return

    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    language = model.model.language_model
    assert len(language.layers) == 36 and language.config.hidden_size == 2048
    vision_end = model.config.vision_end_token_id
    turn_end = proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert proc.tokenizer.convert_ids_to_tokens(turn_end) == '<|im_end|>'
    penalty = float(getattr(model.generation_config, 'repetition_penalty', None) or 1.0)
    with np.load(a.head) as h:
        coef, intercept = h['coef'].astype(np.float64), float(h['intercept'])
    assert coef.shape == (model.get_output_embeddings().weight.shape[0],)
    environment = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), dtype='bfloat16', attention='sdpa',
                       repetition_penalty=penalty, generation_config=model.generation_config.to_dict(),
                       image_processor=proc.image_processor.to_dict())
    if (a.output / 'environment.json').exists():
        assert json.loads((a.output / 'environment.json').read_text()) == json.loads(json.dumps(environment, default=str)), \
            'resume in the recorded environment'
    else:
        write_json(a.output / 'environment.json', json.loads(json.dumps(environment, default=str)))

    mode = 'r+' if state['completed'] else 'w+'
    n = len(rows)
    arrays = dict(L36=np.lib.format.open_memmap(a.output / 'L36.npy', mode=mode, dtype=np.float32, shape=(n, 2048)),
                  Q27=np.lib.format.open_memmap(a.output / 'Q27.npy', mode=mode, dtype=np.float32, shape=(n, 2048)),
                  V27=np.lib.format.open_memmap(a.output / 'V27.npy', mode=mode, dtype=np.float32, shape=(n, 2048)),
                  s0=np.lib.format.open_memmap(a.output / 's0.npy', mode=mode, dtype=np.float64, shape=(n,)),
                  first_raw=np.lib.format.open_memmap(a.output / 'first_raw.npy', mode=mode, dtype=np.int64, shape=(n,)),
                  first_processed=np.lib.format.open_memmap(a.output / 'first_processed.npy', mode=mode, dtype=np.int64, shape=(n,)),
                  span=np.lib.format.open_memmap(a.output / 'span.npy', mode=mode, dtype=np.int32, shape=(n, 2)))
    archived = ({r['sid']: r['generated_token_ids'][0] for r in jsonl(a.archived_answers)}
                if a.archived_answers else {})
    archived_s0 = {}
    if a.archived_scores:
        with np.load(a.archived_scores) as f:
            archived_s0 = dict(zip(f['sid'].tolist(), f['ReFrain_regularized'].astype(np.float64).tolist()))
    cached, span = {}, {}
    max_s0_diff = 0.

    def capture27(module, inputs, output):
        if 'Q27' not in cached:
            h = output[0] if isinstance(output, tuple) else output
            cached['Q27'] = language.norm(h[:, span['start']:span['end']]).float().mean(1)[0].detach().cpu().numpy()
            cached['V27'] = language.norm(h[:, span['visual']]).float().mean(1)[0].detach().cpu().numpy()

    def capture36(module, inputs, output):
        if 'L36' not in cached:
            cached['L36'] = output.last_hidden_state[0, -1].detach().float().cpu().numpy()

    hooks = [language.layers[26].register_forward_hook(capture27), model.model.register_forward_hook(capture36)]
    started, agree, seen = time.perf_counter(), 0, 0
    try:
        for i in range(state['completed'], n):
            row = rows[i]
            image = image_for_processor(image_path(row), proc.image_processor)
            prompt = proc.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=row['prompt'])])],
                                              tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[prompt], images=[image], return_tensors='pt').to('cuda')
            ids = inputs['input_ids'][0]
            ends = (ids == vision_end).nonzero(as_tuple=True)[0].tolist()
            assert len(ends) == 1, (row['sid'], 'vision_end', ends)
            closing = [j for j in (ids == turn_end).nonzero(as_tuple=True)[0].tolist() if j > ends[0]]
            assert closing, (row['sid'], 'no user-turn end')
            span.update(start=ends[0] + 1, end=closing[0],
                        visual=(ids == model.config.image_token_id).nonzero(as_tuple=True)[0].tolist())
            assert span['end'] > span['start'], row['sid']
            cached.clear()
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            raw = logits.detach().cpu().numpy().astype(np.float64)
            processed = raw.copy()
            if penalty != 1.0:   # transformers RepetitionPenaltyLogitsProcessor over the prompt ids
                seen_ids = np.unique(ids.cpu().numpy())
                values = processed[seen_ids]
                processed[seen_ids] = np.where(values < 0, values * penalty, values / penalty)
            assert np.isfinite(raw).all() and 'Q27' in cached and 'L36' in cached
            arrays['L36'][i], arrays['Q27'][i], arrays['V27'][i] = cached['L36'], cached['Q27'], cached['V27']
            arrays['s0'][i] = float(raw @ coef + intercept)
            arrays['first_raw'][i], arrays['first_processed'][i] = int(raw.argmax()), int(processed.argmax())
            arrays['span'][i] = (span['start'], span['end'])
            if row['sid'] in archived:
                seen += 1
                agree += int(arrays['first_processed'][i] == archived[row['sid']])
            if row['sid'] in archived_s0:
                max_s0_diff = max(max_s0_diff, abs(float(arrays['s0'][i]) - archived_s0[row['sid']]))
            if (i + 1) % 50 == 0 or i + 1 == n:
                for x in arrays.values():
                    x.flush()
                state = dict(state='complete' if i + 1 == n else 'running', completed=i + 1, total=n,
                             seconds=state.get('seconds', 0.) + time.perf_counter() - started,
                             live_first_token_agreement=[agree, seen],
                             live_max_abs_s0_difference=max_s0_diff if archived_s0 else None)
                started = time.perf_counter()
                write_json(a.output / 'status.json', state)
                print(json.dumps(state), flush=True)
            del inputs, logits
    finally:
        for hook in hooks:
            hook.remove()
        for x in arrays.values():
            x.flush()


if __name__ == '__main__':
    main()
