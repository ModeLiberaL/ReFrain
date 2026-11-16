"""Recipes, rows and row order of the InternVL3-8B answer SFT (B1) and ordinary refusal SFT.

The two recipes are those of the Qwen scripts scripts/train_answer_sft.py and scripts/train_refusal_sft.py. They
differ on purpose and nothing is shared between them except what the two Qwen scripts share.
answer  (train_answer_sft.py) : answer_rows -> the 677 Standard rows of T with the abstention choice removed and the
         letters remapped; LoRA 16 / 32, float32 adapters; AdamW 2e-5, weight decay 0.01; linear schedule, warmup
         ceil(0.03 * 510) = 16; order of epoch e = numpy.random.default_rng(42 + e).permutation(677), e = 0, 1, 2.
refusal (train_refusal_sft.py): refusal_rows -> all 1,354 rows of T with the original prompt; LoRA 8 / 16, bfloat16
         adapters; AdamW 4e-5, weight decay 1e-4; cosine schedule, warmup int(0.03 * 1017) = 30; each epoch takes
         permutation(677 pairs) from numpy's legacy generator seeded once with 42, then UPD and Standard of each pair.
Both    : seed 42; 3 epochs; one record per forward; 4 consecutive records per optimizer step; LoRA dropout 0.05; no
         gradient clipping; target = letter + end token.
answer_rows and refusal_rows are loaded from the repository's Qwen scripts (--qwen-code, default scripts/); they are
not rewritten here. Standard library and numpy only, so everything in this file runs on a CPU without a model.
Run as a script, it prints the data report of a mode and requires it to equal the report of the recorded run
(recorded_run.json next to this file); with --qwen-run (an output directory of the repository's Qwen script of the
same stage) it also compares rows, recipe values, step structure and learning rates with that run:
  python internvl/ivl_recipe.py --mode answer|refusal --manifest manifests/T.jsonl [--qwen-code scripts]
         [--qwen-run DIR] [--recorded internvl/recorded_run.json] [--output REPORT.json]
"""
import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

SEED, EPOCHS, GROUP = 42, 3, 4
HERE = Path(__file__).resolve().parent
QWEN_CODE = HERE.parent / 'scripts'
RECORDED = HERE / 'recorded_run.json'
INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
RECIPE = {
    'answer': {'rank': 16, 'alpha': 32, 'dropout': 0.05, 'lr': 2e-5, 'weight_decay': 0.01, 'schedule': 'linear',
               'adapter_dtype': 'float32', 'records': 677, 'steps': 510, 'warmup_steps': 16},
    'refusal': {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'lr': 4e-5, 'weight_decay': 1e-4, 'schedule': 'cosine',
                'adapter_dtype': 'bfloat16', 'records': 1354, 'steps': 1017, 'warmup_steps': 30}}
# script, row function and row file of the Qwen run of each mode
QWEN = {'answer': ('train_answer_sft', 'answer_rows', 'train_standard.jsonl'),
        'refusal': ('train_refusal_sft', 'refusal_rows', 'train_refusal.jsonl')}


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    temp.replace(path)


def schedule_steps(mode, records):
    """Optimizer steps and warmup steps, with the expressions of the two Qwen scripts (ceil for answer, int for refusal)."""
    total_steps = math.ceil(records / GROUP) * EPOCHS
    warmup = math.ceil(total_steps * .03) if mode == 'answer' else int(total_steps * .03)
    assert (records, total_steps, warmup) == tuple(RECIPE[mode][k] for k in ('records', 'steps', 'warmup_steps'))
    return total_steps, warmup


def lr_closed_form(mode, steps, warmup):
    """Learning rate in force before each optimizer step: base lr times the multiplier of
    transformers.get_linear_schedule_with_warmup (answer) or get_cosine_schedule_with_warmup with 0.5 cycles
    (refusal), evaluated at step index 0 .. steps - 1. Used only as a check of the scheduler the training uses."""
    base, out = RECIPE[mode]['lr'], []
    for s in range(steps):
        if s < warmup:
            factor = float(s) / float(max(1, warmup))
        elif RECIPE[mode]['schedule'] == 'linear':
            factor = max(0.0, float(steps - s) / float(max(1, steps - warmup)))
        else:
            progress = float(s - warmup) / float(max(1, steps - warmup))
            factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
        out.append(base * factor)
    return out


def qwen_function(qwen_code, mode):
    """answer_rows / refusal_rows of the Qwen script. The top level of these scripts only imports the standard
    library and defines functions, so loading one starts nothing."""
    module, name, _ = QWEN[mode]
    spec = importlib.util.spec_from_file_location('qwen_' + module, Path(qwen_code) / (module + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return getattr(loaded, name)


def training_rows(manifest, mode, qwen_code):
    """(source, rows, pairs). source: the rows of the T manifest. rows: what the Qwen script of the mode trains on.
    pairs: for refusal, the row indices {'standard': i, 'upd': j} of each of the 677 pairs; None for answer."""
    source = read_rows(manifest)
    assert len(source) == 1354 == len({r['sid'] for r in source})
    assert {r['split'] for r in source} == {'T'}, 'training reads the T manifest only'
    build = qwen_function(qwen_code, mode)
    rows, pairs = (build(source), None) if mode == 'answer' else build(source)
    rows = json.loads(json.dumps(rows, ensure_ascii=False))  # tuples become lists, as in the row file of the Qwen run
    assert len(rows) == RECIPE[mode]['records']
    return source, rows, pairs


def epoch_orders(mode, rows, pairs, seed=SEED):
    """Row indices of every epoch in training order; each run of GROUP consecutive indices is one optimizer step.
    answer : the expression of train_answer_sft.py.
    refusal: train_refusal_sft.py seeds numpy's global legacy generator with 42 and calls permutation(677) once per
             epoch. RandomState(42) is that stream, and stays that stream whatever else this process draws."""
    import numpy as np
    if mode == 'answer':
        return [np.random.default_rng(seed + epoch).permutation(len(rows)).tolist() for epoch in range(EPOCHS)]
    stream, orders = np.random.RandomState(seed), []
    for _ in range(EPOCHS):
        pair_order = stream.permutation(len(pairs)).tolist()
        orders.append([pairs[i][kind] for i in pair_order for kind in ('upd', 'standard')])
    return orders


def qwen_run(mode, folder):
    """What a run of the repository's Qwen script of this stage recorded: configuration, training rows, and step,
    learning rate and record count of every log line."""
    folder = Path(folder)
    log = read_rows(folder / 'train.jsonl')
    return {'folder': str(folder), 'config': json.loads((folder / 'config.json').read_text()),
            'rows': read_rows(folder / QWEN[mode][2]), 'steps': [r['step'] for r in log], 'lr': [r['lr'] for r in log],
            'examples': [r['examples'] for r in log]}


def assert_parity(mode, rows, orders, ref):
    """Rows, recipe values and step structure equal those the Qwen run recorded. Returns what was compared."""
    recipe = RECIPE[mode]
    steps, warmup = schedule_steps(mode, len(rows))
    assert rows == ref['rows'], 'the training rows differ from the rows of the Qwen run'
    expected = {'seed': SEED, 'epochs': EPOCHS, 'rows': len(rows), 'micro_batch': 1, 'effective_batch': GROUP,
                'learning_rate': recipe['lr'], 'weight_decay': recipe['weight_decay'], 'warmup_ratio': .03,
                'warmup_steps': warmup, 'total_steps': steps, 'scheduler': recipe['schedule'], 'lora_r': recipe['rank'],
                'lora_alpha': recipe['alpha'], 'lora_dropout': recipe['dropout'], 'max_grad_norm': None,
                'adapter_dtype': 'torch.' + recipe['adapter_dtype'], 'sequence_truncation': False}
    differing = {k: (ref['config'].get(k, 'absent'), v) for k, v in expected.items() if ref['config'].get(k, 'absent') != v}
    assert not differing, differing
    assert ref['steps'] == list(range(1, steps + 1)), 'the Qwen log does not hold exactly one line per step'
    examples = [min(begin + GROUP, len(order)) for order in orders for begin in range(0, len(order), GROUP)]
    assert examples == ref['examples'], 'records per step differ from the Qwen log'
    return {'qwen_run': ref['folder'], 'rows_equal': len(rows), 'config_values_equal': expected, 'log_steps': steps}


def assert_recorded(mode, report, recorded=RECORDED):
    """The data report (rows, letters, option counts, first rows and indices of every epoch) equals the report of the
    recorded InternVL run of this mode."""
    reference = json.loads(Path(recorded).read_text())[mode]
    as_written = json.loads(json.dumps(report, ensure_ascii=False))  # integer keys become strings, as in the record
    assert as_written == reference['data'], 'rows or row order differ from the recorded run'
    return {'recorded': Path(recorded).name, 'data_report_equal': True}


def heldout_disjoint(image_root, rows, heldout):
    """As in the Qwen scripts: every training image exists and none is an image of a held-out manifest. The held-out
    manifests are read for their image paths only. Relative image paths are joined to image_root; absolute ones are
    kept."""
    root = Path(image_root)
    training = {(root / r['image']).resolve() for r in rows}
    assert all(p.is_file() for p in training)
    report = {'training_images': len(training)}
    for path in heldout:
        images = {(root / r['image']).resolve() for r in read_rows(path)}
        assert not (training & images), f'train/heldout image overlap: {path}'
        report[Path(path).name] = len(images)
    return report


def option_tail(display):
    return '\n' + '\n'.join(f'{letter}. {text}' for letter, text in display) + INSTRUCTION


def data_report(mode, source, rows, pairs, orders):
    """What the rows and the orders are, checked row by row against the T manifest. No model, no tokenizer."""
    by_sid = {r['sid']: r for r in source}
    for order in orders:
        assert sorted(order) == list(range(len(rows))), 'an epoch does not visit every row exactly once'
    report = {'mode': mode, 'rows': len(rows), 'types': dict(Counter(by_sid[r['sid']]['type'] for r in rows)),
              'gold_letters': dict(sorted(Counter(r['gold'] for r in rows).items())),
              'options_per_row': dict(sorted(Counter(len(r['display']) for r in rows).items())),
              'records_in_last_step_of_an_epoch': len(rows) % GROUP or GROUP,
              'first_rows_of_each_epoch': [[rows[i]['sid'] for i in order[:GROUP]] for order in orders],
              'first_indices_of_each_epoch': [order[:8] for order in orders]}
    if mode == 'answer':
        differs_only_by_the_choice = choice_was_last = 0
        for row in rows:
            src = by_sid[row['sid']]
            texts = {letter: text for letter, text in src['display']}
            kept = [text for letter, text in src['display'] if letter != src['nota']]
            assert src['type'] == 'standard' and src['gold'] != src['nota'] and row['image'] == src['image'], row['sid']
            assert len(kept) == len(src['display']) - 1 and [text for _, text in row['display']] == kept, row['sid']
            assert [letter for letter, _ in row['display']] == [chr(65 + i) for i in range(len(kept))], row['sid']
            assert {letter: text for letter, text in row['display']}[row['gold']] == texts[src['gold']], row['sid']
            assert row['original_gold'] == src['gold'] and row['prompt'].endswith(option_tail(row['display'])), row['sid']
            assert not any(line.endswith('. ' + texts[src['nota']]) for line in row['prompt'].split('\n')), row['sid']
            assert any(line == f"{src['nota']}. {texts[src['nota']]}" for line in src['prompt'].split('\n')), row['sid']
            head, tail = row['prompt'][:-len(option_tail(row['display']))], option_tail(src['display'])
            differs_only_by_the_choice += src['prompt'] == head + tail
            choice_was_last += src['display'][-1][0] == src['nota']
        report.update(order='numpy.random.default_rng(42 + epoch).permutation(677), epoch = 0, 1, 2',
                      prompts='answer_rows of train_answer_sft.py: abstention choice removed, letters remapped',
                      abstention_choice_lines_in_training_prompts=0, letters='consecutive from A',
                      prompts_equal_to_source_minus_the_choice=differs_only_by_the_choice,
                      rows_whose_removed_choice_was_last=choice_was_last,
                      gold_letter_changed_by_remapping=sum(r['gold'] != r['original_gold'] for r in rows))
    else:
        assert rows == source and len(pairs) == 677
        for row in rows:
            texts = {letter: text for letter, text in row['display']}
            assert (row['gold'] == row['nota']) == (row['type'] == 'upd'), row['sid']
            assert any(line == f"{row['nota']}. {texts[row['nota']]}" for line in row['prompt'].split('\n')), row['sid']
        for order in orders:
            for k in range(0, len(order), 2):
                upd, standard = rows[order[k]], rows[order[k + 1]]
                assert (upd['type'], standard['type']) == ('upd', 'standard') and upd['pair'] == standard['pair']
        report.update(prompts='original manifest prompts, abstention choice present in every row', pairs=len(pairs),
                      order='numpy.random.RandomState(42).permutation(677 pairs), one call per epoch; UPD then '
                            'Standard of each pair; two pairs per step',
                      first_pairs_of_each_epoch=[[rows[i]['pair'] for i in order[:GROUP:2]] for order in orders])
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('answer', 'refusal'), required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--qwen-code', type=Path, default=QWEN_CODE)
    ap.add_argument('--qwen-run', type=Path)
    ap.add_argument('--recorded', type=Path, default=RECORDED)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    import numpy
    source, rows, pairs = training_rows(args.manifest, args.mode, args.qwen_code)
    orders = epoch_orders(args.mode, rows, pairs)
    steps, warmup = schedule_steps(args.mode, len(rows))
    data = data_report(args.mode, source, rows, pairs, orders)
    parity = None
    if args.qwen_run:
        ref = qwen_run(args.mode, args.qwen_run)
        parity = assert_parity(args.mode, rows, orders, ref)
        gap = max(abs(a - b) for a, b in zip(lr_closed_form(args.mode, steps, warmup), ref['lr']))
        assert gap <= 1e-9 * RECIPE[args.mode]['lr'], gap
        parity['learning_rate_max_difference'] = gap
    report = {'recipe': RECIPE[args.mode], 'seed': SEED, 'epochs': EPOCHS, 'records_per_step': GROUP,
              'optimizer_steps': steps, 'warmup_steps': warmup, 'numpy': numpy.__version__, 'data': data,
              'recorded_run': assert_recorded(args.mode, data, args.recorded), 'qwen_parity': parity}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
