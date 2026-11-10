"""Fixed three-epoch answering SFT and one shared, merged Qwen checkpoint."""
import argparse
import json
import math
import random
import time
from pathlib import Path


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def answer_rows(rows):
    result = []
    for row in rows:
        if row['type'] != 'standard':
            continue
        choices = [(letter, text) for letter, text in row['display'] if letter != row['nota']]
        old_gold_text = dict(row['display'])[row['gold']]
        assert row['gold'] != row['nota']
        mapping = {letter: chr(65 + i) for i, (letter, _) in enumerate(choices)}
        first_letter, first_text = row['display'][0]
        question, separator, _ = row['prompt'].partition('\n' + first_letter + '. ' + first_text + '\n')
        assert separator, row['sid']
        display = [(mapping[letter], text) for letter, text in choices]
        gold = mapping[row['gold']]
        assert dict(display)[gold] == old_gold_text
        prompt = question + '\n' + '\n'.join(f'{letter}. {text}' for letter, text in display)
        prompt += "\nAnswer with the option's letter from the given choices directly."
        result.append(dict(sid=row['sid'], image=row['image'], cluster=row['cluster'],
                           prompt=prompt, gold=gold, display=display, original_gold=row['gold']))
    assert len(result) == 677 and len({r['sid'] for r in result}) == 677
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--heldout', type=Path, nargs='+', required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_linear_schedule_with_warmup
    from extract_qwen_baselines import image_for_processor

    seed, epochs, accumulation = 42, 3, 4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = answer_rows(read_rows(a.manifest))
    training_images = {(a.image_root / r['image']).resolve() for r in rows}
    assert all(p.is_file() for p in training_images)
    for path in a.heldout:
        heldout_images = {(a.image_root / r['image']).resolve() for r in read_rows(path)}
        assert not (training_images & heldout_images), f'train/heldout image overlap: {path}'
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output / 'B1' / 'config.json').exists(), 'B1 already exists'
    (a.output / 'train_standard.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')

    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
        target_modules=r'.*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))')
    model = get_peft_model(base, cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('language_model' in name and 'lora_' in name for name, _ in trainable)
    steps_per_epoch = math.ceil(len(rows) / accumulation)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=2e-5, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * .03), total_steps)
    config = dict(model=str(a.model), manifest=str(a.manifest), heldout=[str(p) for p in a.heldout],
        seed=seed, epochs=epochs, rows=len(rows), images=len(training_images), micro_batch=1,
        effective_batch=accumulation, learning_rate=2e-5, weight_decay=.01, warmup_ratio=.03,
        warmup_steps=math.ceil(total_steps * .03), total_steps=total_steps, scheduler='linear',
        lora_r=16, lora_alpha=32, lora_dropout=.05, max_grad_norm=None,
        trainable_parameters=sum(p.numel() for _, p in trainable), gradient_checkpointing=True,
        dtype='bfloat16', adapter_dtype=str(trainable[0][1].dtype), attention='sdpa',
        image_processor=proc.image_processor.to_dict(), sequence_truncation=False,
        target='correct letter plus im_end; prompt and trailing newline ignored',
        torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
        gpu=torch.cuda.get_device_name(), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    (a.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'configured', **{k: config[k] for k in
          ['rows', 'images', 'total_steps', 'trainable_parameters', 'gpu', 'gpu_memory_bytes']}}), flush=True)

    def encode(row):
        image = image_for_processor(a.image_root / row['image'], proc.image_processor)
        prompt = proc.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}],
            tokenize=False, add_generation_prompt=True)
        batch = proc(text=[prompt + row['gold'] + '<|im_end|>\n'], images=[image], return_tensors='pt')
        prefix = proc(text=[prompt], images=[image], return_tensors='pt')['input_ids']
        plen = prefix.shape[1]
        assert torch.equal(batch['input_ids'][:, :plen], prefix)
        assert batch['input_ids'][0, -2].item() == proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
        labels = batch['input_ids'].clone()
        labels[:, :plen] = -100
        labels[:, -1] = -100
        batch['labels'] = labels
        return batch.to('cuda')

    started = time.time()
    step = 0
    model.train()
    with (a.output / 'train.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(epochs):
            order = np.random.default_rng(seed + epoch).permutation(len(rows)).tolist()
            for start in range(0, len(order), accumulation):
                group = order[start:start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                losses, seq_lengths = [], []
                lr_used = optimizer.param_groups[0]['lr']
                for idx in group:
                    batch = encode(rows[idx])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(**batch, use_cache=False).loss
                    assert torch.isfinite(loss).item(), f'nonfinite loss at {rows[idx]["sid"]}'
                    (loss / len(group)).backward()
                    losses.append(float(loss.detach()))
                    seq_lengths.append(batch['input_ids'].shape[1])
                    del batch, loss
                optimizer.step()
                scheduler.step()
                step += 1
                record = dict(step=step, total_steps=total_steps, epoch=epoch + 1,
                    examples=min(start + len(group), len(rows)), loss=sum(losses) / len(losses),
                    lr=lr_used, max_sequence_length=max(seq_lengths),
                    peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    seconds=round(time.time() - started, 1))
                log.write(json.dumps(record) + '\n')
                log.flush()
                (a.output / 'status.json').write_text(json.dumps({'state': 'training', **record}), encoding='utf-8')
                if step == 1 or step % 10 == 0:
                    print(json.dumps(record), flush=True)
            model.save_pretrained(a.output / 'adapter')
            proc.save_pretrained(a.output / 'adapter')

    model.eval()
    model.gradient_checkpointing_disable()
    batch = encode(rows[0])
    answer_position = int((batch.pop('labels')[0] != -100).nonzero()[0]) - 1
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = model(**batch, use_cache=False).logits[:, answer_position, :].float().cpu()
    merged = model.merge_and_unload()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        after = merged(**batch, use_cache=False).logits[:, answer_position, :].float().cpu()
    merge = dict(sid=rows[0]['sid'], max_abs_logit_change=float((after - before).abs().max()),
        before_argmax=int(before.argmax()), after_argmax=int(after.argmax()),
        shared_initialization='All branches load the single saved B1 checkpoint.')
    (a.output / 'merge_check.json').write_text(json.dumps(merge, indent=2), encoding='utf-8')
    merged.config.use_cache = True
    merged.save_pretrained(a.output / 'B1', safe_serialization=True, max_shard_size='4GB')
    proc.save_pretrained(a.output / 'B1')
    state = dict(state='complete', steps=step, epochs=epochs,
        model=str(a.output / 'B1'), seconds=round(time.time() - started, 1), merge=merge)
    (a.output / 'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
