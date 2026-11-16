"""InternVL3-8B answer SFT (B1) and ordinary refusal SFT: LoRA on the language model, vision tower and projector frozen.

Each mode follows the Qwen script of the same stage; the numbers, the rows and the row order are in ivl_recipe.py.
answer : starts from the pinned InternVL3-8B; answer_rows of train_answer_sft.py (the 677 Standard rows of T with
         the abstention choice removed and the letters remapped); rank 16 / alpha 32, float32 adapters; AdamW 2e-5,
         weight decay 0.01; linear schedule, 16 warmup steps of 510; order default_rng(42 + epoch).permutation(677).
refusal: starts from the merged B1 written by ivl_merge.py; all 1,354 rows of T with the original prompt (the
         target of a UPD row is its abstention letter); rank 8 / alpha 16, bfloat16 adapters; AdamW 4e-5, weight
         decay 1e-4; cosine schedule, 30 warmup steps of 1017; 677 pairs permuted per epoch, UPD then Standard.
Both   : seed 42; 3 epochs; one record per forward, 4 consecutive records per optimizer step; LoRA dropout 0.05; no
         gradient clipping; loss = mean cross-entropy over the letter token and the end token.
Before the model is loaded: the data report must equal that of the recorded run (recorded_run.json); the scheduler's
learning rate of every step must equal the closed form of the recipe and, with --qwen-run (an output directory of the
repository's Qwen script of the same stage), the lr column of that run's log, whose rows and recipe values are also
compared; no training image may be an image of a --heldout manifest. Prompt, conversation template, image tiling
and image-token expansion are those of the InternVL extraction. There is no evaluation and no selection here: the
adapter after the last epoch is the result. A run is not resumed; an unfinished output directory is refused.
--smoke-steps N takes N optimizer steps, reports time and memory, and writes neither an adapter nor complete.json.
Usage: python internvl/ivl_train_sft.py --mode answer|refusal --manifest T.jsonl --heldout CM.jsonl E.jsonl
       --image-root DIR --model DIR --clip-up-root DIR --output OUT [--qwen-code scripts] [--qwen-run DIR]
       [--max-num 6] [--no-grad-checkpoint] [--smoke-steps N]
"""
import argparse
import json
import random
import sys
import time
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
    ap.add_argument('--max-num', type=int, default=6)
    ap.add_argument('--no-grad-checkpoint', action='store_true')
    ap.add_argument('--smoke-steps', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ivl_recipe as R
    import ivl_common as K
    import numpy as np
    import torch
    import transformers
    U = K.clip_up_utils(args.clip_up_root)

    mode, recipe = args.mode, R.RECIPE[args.mode]
    source, rows, pairs = R.training_rows(args.manifest, mode, args.qwen_code)
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
    mark = args.model / K.MERGE_MARK
    if mode == 'answer':
        assert not mark.exists(), 'answer SFT starts from the pinned model, not from a merged checkpoint'
        start = None
    else:
        start = json.loads(mark.read_text())
        assert start['role'] == 'b1', 'refusal SFT starts from the merged B1 checkpoint'
    out, smoke = args.output, args.smoke_steps > 0
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('complete', flush=True)
        return
    log_path = out / ('smoke_log.jsonl' if smoke else 'train_log.jsonl')
    assert smoke or not log_path.exists(), 'unfinished run in this output directory; use a new --output'

    model, tok, eos_id = K.load_model(args.model, U)
    random.seed(R.SEED)
    np.random.seed(R.SEED)
    torch.manual_seed(R.SEED)  # after loading: the LoRA initialisation and the dropout masks do not depend on the loading code
    torch.cuda.manual_seed_all(R.SEED)
    params, audit = K.prepare_training(model, mode, not args.no_grad_checkpoint)
    lm = model.language_model
    optimizer = torch.optim.AdamW(params, lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    scheduler = K.make_scheduler(optimizer, mode, steps, warmup)
    config = {'mode': mode, 'recipe': recipe, 'seed': R.SEED, 'epochs': R.EPOCHS, 'records_per_step': R.GROUP,
              'records': len(rows), 'optimizer_steps': steps, 'warmup_steps': warmup,
              'optimizer': 'AdamW (torch defaults for betas and eps)',
              'schedule': f"transformers.get_{recipe['schedule']}_schedule_with_warmup, stepped once per optimizer step",
              'gradient_clipping': None, 'loss': 'mean cross-entropy over the gold-letter token and the end token',
              'lora_targets': K.TARGET.pattern, 'audit': audit, 'data': data, 'recorded_run': recorded,
              'qwen_parity': parity, 'heldout': heldout,
              'model': str(args.model.resolve()), 'start_checkpoint': start, 'manifest': str(args.manifest.resolve()),
              'qwen_code': str(args.qwen_code.resolve()), 'image_root': str(args.image_root.resolve()),
              'max_num_tiles': args.max_num, 'end_token': tok.convert_ids_to_tokens(eos_id),
              'attention': getattr(lm.config, '_attn_implementation', None),
              'attention_dropout': getattr(lm.config, 'attention_dropout', None),
              'environment': {'torch': torch.__version__, 'transformers': transformers.__version__,
                              'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)}}
    R.write_json(out / ('smoke_config.json' if smoke else 'config.json'), config)
    if not smoke:  # the rows trained on, in the layout of the Qwen run's row file
        (out / 'train_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(json.dumps(config), flush=True)

    def log(record):
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    step, t0, epochs = 0, time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    lm.train()  # LoRA dropout and gradient checkpointing; the vision tower and the projector stay in eval mode
    assert not model.vision_model.training and not model.mlp1.training
    for epoch, order in enumerate(orders, 1):
        losses, hits = [], 0
        for begin in range(0, len(order), R.GROUP):
            group = [rows[i] for i in order[begin:begin + R.GROUP]]
            optimizer.zero_grad(set_to_none=True)
            group_loss, longest = 0.0, 0
            for row in group:
                ex = K.encode(model, tok, U, eos_id, row, args.image_root, args.max_num)
                logits = K.forward_logits(model, ex['input_ids'], ex['pixel_values'])
                loss = K.sft_loss(logits, ex['labels'])
                assert torch.isfinite(loss), row['sid']
                (loss / len(group)).backward()
                group_loss += float(loss) / len(group)
                longest = max(longest, len(ex['input_ids']))
                hits += int(logits[ex['prompt_len'] - 1].argmax()) == int(ex['input_ids'][ex['prompt_len']])
                del logits, loss
            norm = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params)))
            assert norm == norm and norm != float('inf'), 'non-finite gradient'
            lr = optimizer.param_groups[0]['lr']
            assert lr == lr_plan[step], (step, lr, lr_plan[step])
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(group_loss)
            log({'event': 'step', 'epoch': epoch, 'step': step, 'records': len(group),
                 'examples': min(begin + len(group), len(rows)), 'loss': group_loss, 'lr': lr, 'grad_norm': norm,
                 'max_sequence_length': longest, 'seconds': time.perf_counter() - t0,
                 'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30})
            if smoke and step == args.smoke_steps:
                R.write_json(out / 'smoke.json', {'steps': step, 'seconds_per_step': (time.perf_counter() - t0) / step,
                                                  'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                                                  'planned_steps': steps})
                return
        epochs.append({'epoch': epoch, 'mean_step_loss': sum(losses) / len(losses),
                       'first_token_hits_while_training': hits, 'records': len(order)})
        log(dict(epochs[-1], event='epoch'))
    assert step == steps, (step, steps)
    K.save_adapter(lm, out / 'adapter', {'mode': mode, 'rank': recipe['rank'], 'alpha': recipe['alpha'],
                                         'dropout': recipe['dropout'], 'adapter_dtype': recipe['adapter_dtype'],
                                         'trained_on': str(args.model.resolve()), 'steps': step, 'seed': R.SEED})
    R.write_json(out / 'complete.json', {'steps': step, 'epochs': epochs, 'seconds': time.perf_counter() - t0,
                                         'peak_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                                         'adapter': str((out / 'adapter').resolve())})
    print(json.dumps({'event': 'training_complete', 'mode': mode, 'steps': step}), flush=True)


if __name__ == '__main__':
    main()
