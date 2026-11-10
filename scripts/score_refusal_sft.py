"""Paired evaluation of ordinary refusal SFT against the shared B1 and other arms."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import Paired, circular, load_reference
from x2_score import native_parser, score

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
parser = native_parser(args.upd_code_root)
ev = load_reference(root, 'qwen', 'E', parser, score)
rows, std, base = ev['rows'], ev['std'], ev['correct']
answers = {r['sid']: r for r in (json.loads(s) for s in
    (root / 'refusal_sft/E/answers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip())}
assert set(answers) == {r['sid'] for r in rows}
parsed = [score(row, answers[row['sid']], parser, 'option') for row in rows]
pred = np.array([bool(p['correct']) if s else bool(p['abstain']) for p, s in zip(parsed, std)])
prior = np.load(root / 'comparison/predictions.npz')
assert prior['sid'].tolist() == [r['sid'] for r in rows]
rep = std & ~prior['R0'] & base
paired = Paired(rows)
delta = pred.astype(int) - base.astype(int)
sci, uci = paired.interval(delta, std), paired.interval(delta, ~std)
arm = dict(method='ordinary_refusal_SFT', standard=float(pred[std].mean()),
    upd=float(pred[~std].mean()), circular_dual=circular(rows, pred)['macro']['dual'],
    gross_loss=int((std & base & ~pred).sum()), repairs=int((std & ~base & pred).sum()),
    new_repair_loss=int((rep & ~pred).sum()), retained_repairs=int((rep & pred).sum()),
    repair_total=int(rep.sum()), standard_change_ci=sci, upd_change_ci=uci,
    passes_original_rule=bool(sci[1] > -.01 and uci[1] > 0))
contrasts = {}
for key in ('R2_loss0', 'R2_loss0.005', 'R3'):
    other = prior[key]
    diff = other.astype(int) - pred.astype(int)
    contrasts[key + '_minus_SFT'] = dict(standard_ci=paired.interval(diff, std),
        upd_ci=paired.interval(diff, ~std),
        gross_loss_reduction_ci=paired.interval(diff * base, std),
        repaired_answer_retention_ci=paired.interval(diff, rep))
result = dict(state='complete', arm=arm, contrasts=contrasts,
    training_config=json.loads((root / 'refusal_sft/config.json').read_text(encoding='utf-8')),
    protocol=dict(shared_base='B1', data='T: 1354 records', epochs=3, seed=42,
        evaluation='historically exposed E; no evaluation-based tuning',
        output_format='common option prompt and NOTA letter',
        relationship_to_EmbLoRA='same B1/data/epochs/LoRA recipe; different answer format, no CLIP injection',
        bootstrap='2000 paired image-cluster draws, seed 0'))
out = root / 'comparison/refusal_sft'
out.mkdir(exist_ok=True)
(out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
np.savez_compressed(out / 'predictions.npz', sid=prior['sid'], refusal_sft=pred)
print(json.dumps(result), flush=True)
