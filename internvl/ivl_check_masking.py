"""T-only checks of the InternVL3-8B SFT rows, recipe, inputs, loss masking and trainable set. No optimizer step is
taken on the model and nothing is written except the report.

Part 0, rows and recipe of the mode (no model, no tokenizer):
  the rows are those of the Qwen script of the stage and their data report equals that of the recorded run
  (recorded_run.json); answer rows carry no abstention choice, letters run from A and the gold option text is
  unchanged; refusal rows are the manifest rows and every epoch is a sequence of (UPD, Standard) records of one pair;
  the learning rate of every step equals the closed form of the recipe; with --qwen-run, rows, weight decay,
  schedule, warmup steps, LoRA rank / alpha / dropout, adapter dtype, seed, epochs and records per step equal that
  Qwen run's config.json and the learning rates its log; no training image is an image of a --heldout manifest.
Part A, every training row of the mode (tokenizer and image tiling, no forward):
  the first ids equal the ids of the inference prompt, rebuilt here with the lines of the InternVL extraction; the
  prompt ends with the assistant header; the only supervised positions are the last two; they hold the gold letter
  (one token) and the end token; the image-context tokens number num_image_token x tiles.
Part B, --gpu-rows rows spread over the set plus the longest row (forward and one backward, no update):
  first-token logits before and after attaching the zero-initialised LoRA agree (--zero-atol); the training loss equals
  two independent computations; replacing future tokens in an equal-length sequence leaves the first supervised
  prediction unchanged; sequence-length numerical differences are recorded; after
  one backward every gradient outside the LoRA matrices is absent, the B gradients are non-zero and the A gradients
  are zero (B starts at zero); the trainable count equals rank x (in + out) summed over the wrapped layers.
Usage: python internvl/ivl_check_masking.py --mode answer|refusal --manifest T.jsonl --heldout CM.jsonl E.jsonl
       --image-root DIR --model DIR --clip-up-root DIR --output REPORT.json [--qwen-code scripts] [--qwen-run DIR]
       [--gpu-rows 8] [--max-num 6] [--no-grad-checkpoint] [--causal-atol 1e-5] [--zero-atol 1e-5]
"""
import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('answer', 'refusal'), required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--heldout', type=Path, nargs='+', required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--clip-up-root', type=Path, required=True)
    ap.add_argument('--qwen-code', type=Path, default=Path(__file__).resolve().parent.parent / 'scripts')
    ap.add_argument('--qwen-run', type=Path)
    ap.add_argument('--recorded', type=Path, default=Path(__file__).resolve().parent / 'recorded_run.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu-rows', type=int, default=8)
    ap.add_argument('--max-num', type=int, default=6)
    ap.add_argument('--no-grad-checkpoint', action='store_true')
    ap.add_argument('--causal-atol', type=float, default=1e-5)
    ap.add_argument('--zero-atol', type=float, default=1e-5)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ivl_recipe as R
    import ivl_common as K
    import numpy as np
    import torch
    U = K.clip_up_utils(args.clip_up_root)

    # Part 0
    mode = args.mode
    source, rows, pairs = R.training_rows(args.manifest, mode, args.qwen_code)
    kind = {r['sid']: r['type'] for r in source}
    orders = R.epoch_orders(mode, rows, pairs)
    steps, warmup = R.schedule_steps(mode, len(rows))
    data = R.data_report(mode, source, rows, pairs, orders)
    recorded = R.assert_recorded(mode, data, args.recorded)
    lr_plan = K.lr_sequence(mode, steps, warmup)
    recorded['learning_rate_closed_form_max_difference'] = K.lr_parity(mode, lr_plan, R.lr_closed_form(mode, steps, warmup))
    parity = None
    if args.qwen_run:
        qwen = R.qwen_run(mode, args.qwen_run)
        parity = R.assert_parity(mode, rows, orders, qwen)
        parity['learning_rate_max_difference'] = K.lr_parity(mode, lr_plan, qwen['lr'])
    heldout = R.heldout_disjoint(args.image_root, rows, args.heldout)
    part_0 = {'data': data, 'recorded_run': recorded, 'qwen_parity': parity, 'heldout': heldout, 'numpy': np.__version__,
              'schedule': {'optimizer_steps': steps, 'warmup_steps': warmup, 'kind': R.RECIPE[mode]['schedule']}}
    print(part_0, flush=True)

    model, tok, eos_id = K.load_model(args.model, U)
    context_id, start_id, end_id = (tok.convert_tokens_to_ids(t) for t in
                                    (U.IMG_CONTEXT_TOKEN, U.IMG_START_TOKEN, U.IMG_END_TOKEN))
    header, end = model.conv_template.roles[1], model.conv_template.sep.strip()

    # Part A
    lengths, tiles_seen, same_as_string, longest = [], Counter(), 0, None
    for row in rows:
        ex = K.encode(model, tok, U, eos_id, row, args.image_root, args.max_num)
        ids, labels, p, tiles = ex['input_ids'], ex['labels'], ex['prompt_len'], ex['pixel_values'].shape[0]
        template = model.conv_template.copy()
        template.append_message(template.roles[0], '<image>\n' + row['prompt'])
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt().replace(
            '<image>', U.IMG_START_TOKEN + U.IMG_CONTEXT_TOKEN * model.num_image_token * tiles + U.IMG_END_TOKEN, 1)
        reference = tok(prompt, return_tensors='pt')['input_ids'][0]
        assert p == len(reference) and torch.equal(ids[:p], reference), row['sid']
        assert prompt.endswith(header) and row['prompt'] in prompt, row['sid']
        gold = tok(row['gold'], add_special_tokens=False)['input_ids']
        assert len(gold) == 1 and tok.decode(gold) == row['gold'], (row['sid'], gold)
        assert (labels != K.IGNORE).nonzero(as_tuple=True)[0].tolist() == [p, p + 1] == list(range(p, len(ids))), row['sid']
        assert ids[p:].tolist() == labels[p:].tolist() == gold + [eos_id], row['sid']
        assert 1 <= tiles <= args.max_num + 1 and int((ids == context_id).sum()) == model.num_image_token * tiles, row['sid']
        assert int((ids == start_id).sum()) == 1 == int((ids == end_id).sum()), row['sid']
        same_as_string += torch.equal(tok(prompt + row['gold'] + end, return_tensors='pt')['input_ids'][0], ids)
        lengths.append(len(ids))
        tiles_seen[tiles] += 1
        if longest is None or len(ids) > longest[0]:
            longest = (len(ids), row)
    part_a = {'rows': len(rows), 'supervised_tokens_per_row': 2, 'assistant_header': header, 'end_token': end,
              'num_image_token': model.num_image_token, 'tiles': {str(k): v for k, v in sorted(tiles_seen.items())},
              'sequence_length': {'min': min(lengths), 'median': statistics.median(lengths), 'max': max(lengths)},
              'rows_equal_to_tokenizing_the_full_string': same_as_string}
    print(part_a, flush=True)

    # Part B
    step = max(1, len(rows) // max(1, args.gpu_rows))
    picks = rows[::step][:args.gpu_rows]
    if longest[1]['sid'] not in {r['sid'] for r in picks}:
        picks.append(longest[1])
    base = {r['sid']: K.first_token_logits(model, tok, U, r, args.image_root, args.max_num) for r in picks}
    torch.manual_seed(R.SEED)
    params, audit = K.prepare_training(model, mode, not args.no_grad_checkpoint)
    assert not model.training and not any(m.training for m in model.language_model.modules())
    checks = []
    for row in picks:
        attached = K.first_token_logits(model, tok, U, row, args.image_root, args.max_num)
        zero_init = float((attached - base[row['sid']]).abs().max())
        assert zero_init <= args.zero_atol, (row['sid'], zero_init)  # same tolerance as the extraction script's witness
        ex = K.encode(model, tok, U, eos_id, row, args.image_root, args.max_num)
        p = ex['prompt_len']
        with torch.inference_mode():
            logits = K.forward_logits(model, ex['input_ids'], ex['pixel_values'])
            loss = float(K.sft_loss(logits, ex['labels']))
            shifted = float(torch.nn.functional.cross_entropy(logits[:-1].float(), ex['labels'][1:].to(logits.device),
                                                              ignore_index=K.IGNORE))
            logp = torch.log_softmax(logits[[p - 1, p]].double(), -1)
            manual = -float(logp[0, int(ex['input_ids'][p])] + logp[1, eos_id]) / 2
            first = logits[p - 1].float().cpu()
            future_changed = ex['input_ids'].clone()
            future_changed[p:] = tok('Z', add_special_tokens=False)['input_ids'][0]
            altered_first = K.forward_logits(model, future_changed, ex['pixel_values'])[p - 1].float().cpu()
        assert abs(loss - shifted) < 1e-4 and abs(loss - manual) < 1e-4, (row['sid'], loss, shifted, manual)
        gap = float((first - base[row['sid']]).abs().max())
        causal_gap = float((first - altered_first).abs().max())
        assert causal_gap <= args.causal_atol, (row['sid'], causal_gap)
        checks.append({'sid': row['sid'], 'type': kind[row['sid']], 'gold': row['gold'], 'tokens': len(ex['input_ids']),
                       'tiles': int(ex['pixel_values'].shape[0]), 'loss': loss, 'loss_shifted_ce': shifted,
                       'loss_manual': manual, 'zero_init_max_error': zero_init,
                       'equal_length_future_token_max_error': causal_gap,
                       'teacher_forced_vs_prompt_only_max_error': gap,
                       'same_first_token_argmax': int(first.argmax()) == int(base[row['sid']].argmax())})

    lm = model.language_model
    lm.train()
    torch.cuda.reset_peak_memory_stats()
    ex = K.encode(model, tok, U, eos_id, longest[1], args.image_root, args.max_num)
    loss = K.sft_loss(K.forward_logits(model, ex['input_ids'], ex['pixel_values']), ex['labels'])
    loss.backward()
    grads_a = [m.lora_A.grad for _, m in K.lora_modules(lm)]
    grads_b = [m.lora_B.grad for _, m in K.lora_modules(lm)]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads_a + grads_b)
    assert all(float(g.abs().max()) == 0.0 for g in grads_a), 'A gradients must vanish while B is zero'
    nonzero_b = sum(float(g.abs().max()) > 0 for g in grads_b)
    assert nonzero_b > 0, 'no gradient reached the LoRA B matrices'
    assert all(q.grad is None for _, q in model.named_parameters() if not q.requires_grad)
    backward = {'sid': longest[1]['sid'], 'tokens': longest[0], 'loss': float(loss),
                'lora_B_with_nonzero_gradient': nonzero_b, 'lora_modules': len(grads_b),
                'gradients_outside_lora': 0, 'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                'vision_model_training_flag': model.vision_model.training, 'mlp1_training_flag': model.mlp1.training}
    assert not model.vision_model.training and not model.mlp1.training
    report = {'status': 'pass', 'mode': mode, 'model': str(args.model.resolve()),
              'manifest': str(args.manifest.resolve()), 'recipe': R.RECIPE[mode], 'part_0': part_0, 'part_a': part_a,
              'audit': audit, 'part_b': checks, 'backward': backward, 'causal_atol': args.causal_atol,
              'zero_atol': args.zero_atol}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    R.write_json(args.output, report)
    print({k: v for k, v in report.items() if k not in ('part_0', 'part_b')}, flush=True)


if __name__ == '__main__':
    main()
