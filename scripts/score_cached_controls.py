"""Score cached C=1, maximum probability and entropy controls on the shared B1."""
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
sets = {s: load_reference(root, 'qwen', s, parser, score) for s in ('T', 'CM', 'E')}
head = np.load(root / 'probe/published_C1.npz')
head_scores = dict(zip(head['sid'].tolist(), head['logits'].tolist()))
scores = {name: {} for name in ('LP_C1_shared_executor', 'MSP', 'negative_entropy')}
for split, data in sets.items():
    scores['LP_C1_shared_executor'][split] = np.array([head_scores[r['sid']] for r in data['rows']])
    x = np.load(root / 'qwen_base' / split / 'logits.npy', mmap_mode='r')
    msp, entropy = [], []
    for start in range(0, len(x), 64):
        z = np.asarray(x[start:start+64], dtype=np.float64)
        z -= z.max(1, keepdims=True)
        ex = np.exp(z)
        total = ex.sum(1)
        msp.extend(1 / total)
        entropy.extend((ex * z).sum(1) / total - np.log(total))
    scores['MSP'][split] = np.asarray(msp)
    scores['negative_entropy'][split] = np.asarray(entropy)
ev, cm = sets['E'], sets['CM']
std, base = ev['std'], ev['correct']
paired = Paired(ev['rows'])
original = np.load(root / 'comparison/predictions.npz')
assert original['sid'].tolist() == [r['sid'] for r in ev['rows']]
rep = std & ~original['R0'] & base
arms, predictions = [], {}
for name, ss in scores.items():
    grid = np.r_[-np.inf, np.unique(ss['T']), np.inf]
    correct_scores = np.sort(ss['CM'][cm['std'] & cm['correct']])
    losses = np.searchsorted(correct_scores, grid, side='left') / cm['std'].sum()
    for budget in (0., .005):
        index = np.flatnonzero(losses <= budget + 1e-12)[-1]
        tau = float(grid[index])
        pred = np.where(ss['E'] >= tau, base, ~std)
        delta = pred.astype(int) - base.astype(int)
        sci, uci = paired.interval(delta, std), paired.interval(delta, ~std)
        ours = original['R2_loss0' if budget == 0 else 'R2_loss0.005']
        diff = ours.astype(int) - pred.astype(int)
        key = name + '/loss_' + str(budget)
        predictions[key] = pred
        arms.append(dict(method=name, budget=budget,
            threshold=tau if np.isfinite(tau) else str(tau), calibration_loss=float(losses[index]),
            standard=float(pred[std].mean()), upd=float(pred[~std].mean()),
            circular_dual=circular(ev['rows'], pred)['macro']['dual'],
            gross_loss=int((std & base & ~pred).sum()), repairs=int((std & ~base & pred).sum()),
            retained_repairs=int((rep & pred).sum()), repair_total=int(rep.sum()),
            standard_change_ci=sci, upd_change_ci=uci,
            passes_original_rule=bool(sci[1] > -.01 and uci[1] > 0),
            ours_minus_control_standard_ci=paired.interval(diff, std),
            ours_minus_control_upd_ci=paired.interval(diff, ~std)))
result = dict(status='complete', scope='post-SFT B1 cached component controls; common deterministic NOTA executor',
    selection='C=1, MSP and negative entropy fixed; thresholds from T and CM only',
    evaluated_after_main_results=True, arms=arms)
out = root / 'comparison/cached_controls'
out.mkdir(exist_ok=True)
(out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
np.savez_compressed(out / 'predictions.npz', sid=original['sid'], **predictions)
print(json.dumps(result))
