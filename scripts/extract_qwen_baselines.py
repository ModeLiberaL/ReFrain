"""Cache native answers, final prompt hidden states and first-token raw logits."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    temp.replace(path)


def image_for_processor(path, image_processor):
    from PIL import Image
    image = Image.open(path).convert('RGB')
    factor = image_processor.patch_size * image_processor.merge_size
    width, height = image.size
    if min(width, height) < factor:
        min_pixels = image_processor.size['shortest_edge']
        max_pixels = image_processor.size['longest_edge']
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        image = image.resize((w_bar, h_bar))
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--witness', action='store_true')
    args = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = read_rows(args.input)
    assert rows and len({r['sid'] for r in rows}) == len(rows)
    for row in rows:
        assert (args.image_root / row['image']).is_file(), row['image']
        assert row['prompt']
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    binding = {'rows': rows, 'model': str(args.model.resolve()),
               'image_root': str(args.image_root.resolve()), 'batch_size': args.batch_size,
               'max_new_tokens': args.max_new_tokens, 'witness': args.witness}
    if (out / 'inputs.json').exists():
        assert json.loads((out / 'inputs.json').read_text(encoding='utf-8')) == binding, 'input/settings differ'
    else:
        write_json(out / 'inputs.json', binding)
    state = json.loads((out / 'status.json').read_text(encoding='utf-8')) if (out / 'status.json').exists() else {'completed': 0, 'seconds': 0}
    if state['completed'] == len(rows):
        print('complete', flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    vocab, hidden_dim = model.get_output_embeddings().weight.shape
    mode = 'r+' if state['completed'] else 'w+'
    logit_file = np.lib.format.open_memmap(out / 'logits.npy', mode=mode, dtype=np.float32, shape=(len(rows), vocab))
    hidden_file = np.lib.format.open_memmap(out / 'hidden.npy', mode=mode, dtype=np.float32, shape=(len(rows), hidden_dim))
    responses = read_rows(out / 'answers.jsonl') if (out / 'answers.jsonl').exists() else []
    responses = responses[:state['completed']]
    assert [r['sid'] for r in responses] == [r['sid'] for r in rows[:state['completed']]]
    with (out / 'answers.jsonl').open('w', encoding='utf-8') as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'environment.json', {
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(), 'dtype': 'bfloat16', 'attention': 'sdpa',
        'generation_config': model.generation_config.to_dict(),
        'processor': processor.image_processor.to_dict(), 'chat_template': processor.chat_template,
        'hidden_dimension': hidden_dim, 'vocabulary_size': vocab,
        'logits': 'unprocessed model logits at the first generated token, float32 storage',
        'hidden': 'final normalized hidden state at the last prompt token, float32 storage',
        'allocated_output_bytes': len(rows) * (vocab + hidden_dim) * 4})
    elapsed_before = state['seconds']
    for start in range(state['completed'], len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        images = [image_for_processor(args.image_root / r['image'], processor.image_processor) for r in batch]
        prompts = [processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': r['prompt']}]}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        inputs = processor(text=prompts, images=images, padding=True, truncation=False, return_tensors='pt').to('cuda')
        cached = {}

        def capture_hidden(module, positional, result):
            if 'hidden' not in cached:
                cached['hidden'] = result.last_hidden_state[:, -1].detach().float().cpu().numpy()

        def capture_logits(module, positional, result):
            if 'logits' not in cached:
                cached['logits'] = result.logits[:, -1].detach().float().cpu().numpy()

        hh = model.model.register_forward_hook(capture_hidden)
        hl = model.register_forward_hook(capture_logits)
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        finally:
            hh.remove()
            hl.remove()
        assert np.isfinite(cached['hidden']).all() and np.isfinite(cached['logits']).all()
        if args.witness and start == 0:
            with torch.inference_mode():
                direct = model(**inputs, use_cache=True, logits_to_keep=1).logits[:, -1].float().cpu().numpy()
            max_error = float(np.max(np.abs(direct - cached['logits'])))
            assert np.allclose(direct, cached['logits'], rtol=0, atol=1e-5), max_error
            write_json(out / 'witness.json', {'rows': [r['sid'] for r in batch],
                'direct_forward_vs_generation_raw_logits_max_error': max_error,
                'hidden_shape': list(cached['hidden'].shape), 'logits_shape': list(cached['logits'].shape)})
        generated = generated[:, inputs['input_ids'].shape[1]:]
        answers = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        stop = start + len(batch)
        logit_file[start:stop] = cached['logits']
        hidden_file[start:stop] = cached['hidden']
        logit_file.flush()
        hidden_file.flush()
        with (out / 'answers.jsonl').open('a', encoding='utf-8') as f:
            for row, answer, tokens in zip(batch, answers, generated):
                f.write(json.dumps({'sid': row['sid'], 'response': answer,
                    'generated_token_ids': tokens.cpu().tolist()}, ensure_ascii=False) + '\n')
        state.update(completed=stop, total=len(rows), seconds=elapsed_before + time.perf_counter() - t0,
                     state='complete' if stop == len(rows) else 'running')
        write_json(out / 'status.json', state)
        print(json.dumps(state), flush=True)
        del inputs, generated, cached


if __name__ == '__main__':
    main()
