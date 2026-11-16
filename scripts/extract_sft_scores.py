"""Cache native SFT answers and the scores needed for selective evaluation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from extract_qwen_baselines import image_for_processor, read_rows, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = read_rows(a.input)
    assert rows and len({r['sid'] for r in rows}) == len(rows)
    a.output.mkdir(parents=True, exist_ok=True)
    binding = dict(rows=rows, model=str(a.model), image_root=str(a.image_root),
                   batch_size=1, max_new_tokens=1024, do_sample=False)
    if (a.output/'inputs.json').exists():
        assert json.loads((a.output/'inputs.json').read_text()) == binding
    else:
        write_json(a.output/'inputs.json', binding)
    state = json.loads((a.output/'status.json').read_text()) if (a.output/'status.json').exists() else dict(completed=0, seconds=0)
    if state['completed'] == len(rows):
        return
    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    proc.tokenizer.padding_side = 'left'
    letter_ids = [proc.tokenizer.encode(x, add_special_tokens=False) for x in 'ABCDEF']
    assert letter_ids == [[i] for i in range(32, 38)], letter_ids
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    mode = 'r+' if state['completed'] else 'w+'
    letter = np.lib.format.open_memmap(a.output/'letter_logits.npy', mode=mode, dtype=np.float32, shape=(len(rows), 6))
    confidence = np.lib.format.open_memmap(a.output/'confidence.npy', mode=mode, dtype=np.float64, shape=(len(rows), 2))
    answers = read_rows(a.output/'answers.jsonl')[:state['completed']] if (a.output/'answers.jsonl').exists() else []
    assert [x['sid'] for x in answers] == [x['sid'] for x in rows[:state['completed']]]
    with (a.output/'answers.jsonl').open('w') as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')
    write_json(a.output/'environment.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(), generation_config=model.generation_config.to_dict(),
        processor=proc.image_processor.to_dict(), chat_template=proc.chat_template,
        logits='Raw first-generation-token A..F logits', confidence_columns=['MSP', 'negative_entropy']))
    started, previous = time.perf_counter(), state['seconds']
    for i in range(state['completed'], len(rows)):
        row = rows[i]
        image = image_for_processor(a.image_root/row['image'], proc.image_processor)
        prompt = proc.apply_chat_template([{'role':'user','content':[
            {'type':'image'}, {'type':'text','text':row['prompt']}]}], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[image], padding=True, truncation=False, return_tensors='pt').to('cuda')
        cached = {}

        def capture(module, args, output):
            if 'z' not in cached:
                cached['z'] = output.logits[:, -1].detach().float().cpu().numpy()[0]

        hook = model.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
        finally:
            hook.remove()
        z = cached['z']
        assert np.isfinite(z).all()
        letter[i] = z[32:38]
        zz = z.astype(np.float64)-float(z.max())
        ex = np.exp(zz)
        total = ex.sum()
        confidence[i] = [1/total, (ex*zz).sum()/total-np.log(total)]
        generated = generated[:, inputs['input_ids'].shape[1]:]
        response = proc.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        letter.flush()
        confidence.flush()
        with (a.output/'answers.jsonl').open('a') as f:
            f.write(json.dumps(dict(sid=row['sid'], response=response,
                generated_token_ids=generated[0].cpu().tolist(), first_token_argmax=int(z.argmax())), ensure_ascii=False)+'\n')
        state.update(completed=i+1, total=len(rows), seconds=previous+time.perf_counter()-started,
                     state='complete' if i+1 == len(rows) else 'running')
        write_json(a.output/'status.json', state)
        if i == 0 or (i+1) % 50 == 0 or i+1 == len(rows):
            print(json.dumps(state), flush=True)
        del inputs, generated, cached


if __name__ == '__main__':
    main()
