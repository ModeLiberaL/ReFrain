"""Budget curves and task breakdowns from frozen predictions and cached scores."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import Paired, load_reference
from x2_score import native_parser, score

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
parser = native_parser(args.upd_code_root)
sets = {s: load_reference(root, 'qwen', s, parser, score) for s in ('T', 'CM', 'E')}
ev, cm = sets['E'], sets['CM']
rows, std, base = ev['rows'], ev['std'], ev['correct']
prior = np.load(root / 'comparison/predictions.npz')
sid = [r['sid'] for r in rows]
assert prior['sid'].tolist() == sid
rep = std & ~prior['R0'] & base
scores = {}
for name, filename in [('logits_CV', 'train_cv.npz'), ('logits_C1', 'published_C1.npz')]:
    data = np.load(root / 'probe' / filename)
    mapping = dict(zip(data['sid'].tolist(), data['logits'].tolist()))
    scores[name] = {s: np.array([mapping[r['sid']] for r in d['rows']]) for s, d in sets.items()}
for name, folder in [('BICR', 'bicr'), ('ACR_MCQ_scaled', 'acr_mcq_standardized')]:
    data = np.load(root / folder / 'scores.npz')
    scores[name] = {s: data[s] for s in sets}
curves = []
for method, ss in scores.items():
    grid = np.r_[-np.inf, np.unique(ss['T']), np.inf]
    losses = np.searchsorted(np.sort(ss['CM'][cm['std'] & cm['correct']]), grid, side='left')
    for budget in (0., .001, .0025, .005, .01):
        tau = float(grid[np.flatnonzero(losses / cm['std'].sum() <= budget + 1e-12)[-1]])
        pred = np.where(ss['E'] >= tau, base, ~std)
        curves.append(dict(method=method, cm_budget=budget,
            threshold=tau if np.isfinite(tau) else str(tau),
            cm_actual_loss=int(((ss['CM'] < tau) & cm['std'] & cm['correct']).sum()),
            standard=float(pred[std].mean()), upd=float(pred[~std].mean()),
            e_loss=int((std & base & ~pred).sum()), retained_repairs=int((rep & pred).sum())))

predictions = {'B1': base, 'CV_0': prior['R2_loss0'], 'CV_0.5': prior['R2_loss0.005'],
               'EmbLoRA': prior['R3']}
for name, folder, key in [('ordinary_SFT', 'refusal_sft', 'refusal_sft'),
    ('BICR_0.5', 'bicr', 'loss_0.005'),
    ('ACR_scaled_0.5', 'acr_mcq_standardized', 'ACR-MCQ standardized/loss_0.005'),
    ('CLIP_CV_0.5', 'clip_classifier', 'train_CV/shared_loss_0.005'),
    ('C1_0.5', 'cached_controls', 'LP_C1_shared_executor/loss_0.005')]:
    data = np.load(root / 'comparison' / folder / 'predictions.npz')
    assert data['sid'].tolist() == sid
    predictions[name] = data[key]
tasks = sorted({r['task'] for r in rows})
task_results = []
paired = Paired(rows)
for task in tasks:
    mask = np.array([r['task'] == task for r in rows])
    for method, pred in predictions.items():
        sm, um = mask & std, mask & ~std
        diff = pred.astype(int) - base.astype(int)
        task_results.append(dict(task=task, method=method, standard_count=int(sm.sum()),
            upd_count=int(um.sum()), standard=float(pred[sm].mean()), upd=float(pred[um].mean()),
            lost=int((sm & base & ~pred).sum()), repaired=int((sm & ~base & pred).sum()),
            retained_repairs=int((mask & rep & pred).sum()), repair_total=int((mask & rep).sum()),
            standard_delta_ci=paired.interval(diff, sm), upd_delta_ci=paired.interval(diff, um)))

conditions = [
    ('EmbLoRA loses a new repair retained by CV', rep & ~predictions['EmbLoRA'] & predictions['CV_0.5']),
    ('SFT loses a new repair retained by CV', rep & ~predictions['ordinary_SFT'] & predictions['CV_0.5']),
    ('SFT repairs a B1 Standard error', std & ~base & predictions['ordinary_SFT']),
    ('CV loses a new repair retained by SFT', rep & ~predictions['CV_0.5'] & predictions['ordinary_SFT']),
    ('SFT detects UPD missed by CV', ~std & ~predictions['CV_0.5'] & predictions['ordinary_SFT']),
    ('CV detects UPD missed by EmbLoRA', ~std & predictions['CV_0.5'] & ~predictions['EmbLoRA'])]
examples, used_clusters = [], set()
for description, mask in conditions:
    candidates = [i for i in np.flatnonzero(mask) if rows[i]['cluster'] not in used_clusters]
    if candidates:
        i = min(candidates, key=lambda j: rows[j]['sid'])
        used_clusters.add(rows[i]['cluster'])
        examples.append(dict(category=description, row=rows[i],
            outcomes={method: bool(p[i]) for method, p in predictions.items()}))
out = root / 'comparison/paper_diagnostics'
out.mkdir(exist_ok=True)
result = dict(state='complete', curves=curves, tasks=task_results, examples=examples,
    protocol='Additional descriptive analysis; T candidates and CM constraints only; E never selects curve thresholds',
    denominators=dict(cm_standard=int(cm['std'].sum()), e_standard=int(std.sum()), repairs=int(rep.sum())))
(out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
lines = ['# 共同 B1 的分任务结果', '', '| 任务 | 方法 | Standard % | UPD % | 损失/修复 | 新修复保持 |',
         '|---|---|---:|---:|---:|---:|']
for r in task_results:
    lines.append(f"| {r['task']} | {r['method']} | {100*r['standard']:.3f} | {100*r['upd']:.3f} | "
                 f"{r['lost']}/{r['repaired']} | {r['retained_repairs']}/{r['repair_total']} |")
(out / 'tasks.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(dict(state='complete', curves=len(curves), task_rows=len(task_results), examples=len(examples))))
