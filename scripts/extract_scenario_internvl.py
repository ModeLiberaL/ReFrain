"""Extract InternVL3-8B scenario features from the first native prefill."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers.generation
import transformers.modeling_utils
from transformers import AutoModel, AutoTokenizer


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSONL rows with sid, image and prompt.')
    parser.add_argument('--model', type=Path, required=True, help='Merged InternVL3-8B answering checkpoint.')
    parser.add_argument('--head', type=Path, required=True, help='Frozen raw-logit source head NPZ.')
    parser.add_argument('--clip-up-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from PIL import Image
    sys.path.insert(0, str(args.clip_up_root.resolve()))
    from clip_up.vlms.internvl import clip_up_internvl_utils as images

    torch.manual_seed(20261117)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    binding = dict(rows=rows, model=str(args.model.resolve()), max_tiles=6,
                   layer=21, pooling='per-token final RMSNorm, then float32 token mean')
    if (args.output / 'inputs.json').exists():
        if json.loads((args.output / 'inputs.json').read_text()) != binding:
            raise ValueError('The output directory belongs to different inputs.')
    else:
        write_json(args.output / 'inputs.json', binding)
    state = (json.loads((args.output / 'status.json').read_text()) if (args.output / 'status.json').exists()
             else dict(completed=0, seconds=0.))
    if state['completed'] == len(rows):
        return
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=False, trust_remote_code=True, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False, local_files_only=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(images.IMG_CONTEXT_TOKEN)
    image_end = tokenizer.convert_tokens_to_ids(images.IMG_END_TOKEN)
    turn_end = tokenizer.convert_tokens_to_ids(model.conv_template.sep.strip())
    language = model.language_model
    assert len(language.model.layers) == 28 and language.config.hidden_size == 3584
    with np.load(args.head) as source:
        coef, bias = source['coef'].astype(float), float(source['intercept'])
    write_json(args.output / 'environment.json', dict(torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'), gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda, language_attention=language.config._attn_implementation,
        max_tiles=6, native_generation_tokens=1, model=str(args.model.resolve())))
    mode, n = ('r+' if state['completed'] else 'w+'), len(rows)
    arrays = {key: np.lib.format.open_memmap(args.output / (key + '.npy'), mode=mode,
              dtype=np.float32, shape=(n, 3584)) for key in ['L28', 'Q21', 'V21']}
    arrays['s0'] = np.lib.format.open_memmap(args.output / 's0.npy', mode=mode, dtype=np.float64, shape=(n,))
    arrays['first_token'] = np.lib.format.open_memmap(args.output / 'first_token.npy', mode=mode, dtype=np.int64, shape=(n,))
    arrays['span'] = np.lib.format.open_memmap(args.output / 'span.npy', mode=mode, dtype=np.int32, shape=(n, 2))
    captured, span = {}, {}

    def middle_hook(module, inputs, output):
        if 'Q21' in captured:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        captured['Q21'] = language.model.norm(hidden[:, span['start']:span['end']]).float().mean(1)[0].detach().cpu().numpy()
        captured['V21'] = language.model.norm(hidden[:, span['visual']]).float().mean(1)[0].detach().cpu().numpy()

    def final_hook(module, inputs, output):
        if 'L28' not in captured:
            captured['L28'] = output.last_hidden_state[0, -1].detach().float().cpu().numpy()

    def logits_hook(module, inputs, output):
        if 's0' not in captured:
            raw = output.logits[0, -1].detach().float().cpu().numpy().astype(float)
            assert raw.shape == coef.shape
            captured['s0'] = float(raw @ coef + bias)

    hooks = [language.model.layers[20].register_forward_hook(middle_hook),
             language.model.register_forward_hook(final_hook), language.register_forward_hook(logits_hook)]
    started, prior = time.monotonic(), state['seconds']
    try:
        for i in range(state['completed'], n):
            row = rows[i]
            image = Image.open(args.image_root / row['image']).convert('RGB')
            pixels = images.load_image_internvl(image, max_num=6).to(dtype=torch.bfloat16, device='cuda')
            template = model.conv_template.copy()
            template.append_message(template.roles[0], '<image>\n' + row['prompt'])
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt().replace('<image>', images.IMG_START_TOKEN +
                images.IMG_CONTEXT_TOKEN * model.num_image_token * pixels.shape[0] + images.IMG_END_TOKEN, 1)
            encoded = tokenizer(prompt, return_tensors='pt')
            ids = encoded['input_ids'][0]
            ends = (ids == image_end).nonzero(as_tuple=True)[0].tolist()
            assert len(ends) == 1
            start = ends[0] + 1
            end = next(j for j in (ids == turn_end).nonzero(as_tuple=True)[0].tolist() if j >= start)
            visual = (ids == model.img_context_token_id).nonzero(as_tuple=True)[0].tolist()
            assert tokenizer.decode(ids[start:end], skip_special_tokens=False).strip() == row['prompt'].strip()
            assert len(visual) == model.num_image_token * pixels.shape[0]
            span.update(start=start, end=end, visual=visual)
            captured.clear()
            with torch.inference_mode():
                generated = model.generate(input_ids=encoded['input_ids'].cuda(), pixel_values=pixels,
                    attention_mask=encoded['attention_mask'].cuda(), do_sample=False, max_new_tokens=1,
                    eos_token_id=turn_end, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else turn_end)
            for key in ['L28', 'Q21', 'V21', 's0']:
                assert np.isfinite(captured[key]).all()
                arrays[key][i] = captured[key]
            arrays['first_token'][i] = int(generated[0, 0])
            arrays['span'][i] = (start, end)
            if (i + 1) % 50 == 0 or i + 1 == n:
                for array in arrays.values():
                    array.flush()
                state = dict(state='complete' if i + 1 == n else 'running', completed=i + 1,
                             total=n, seconds=prior + time.monotonic() - started)
                write_json(args.output / 'status.json', state)
                print(json.dumps(state), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        for array in arrays.values():
            array.flush()


if __name__ == '__main__':
    main()
