"""Paired comparisons of the two prespecified operating points against EmbLoRA."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation import Paired

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
rows = json.loads((root / 'qwen_base/E/inputs.json').read_text(encoding='utf-8'))['rows']
pred = np.load(root / 'comparison/predictions.npz')
assert pred['sid'].tolist() == [r['sid'] for r in rows]
paired = Paired(rows)
std = paired.std
r0, b1, emb = [pred[k] for k in ('R0', 'R1', 'R3')]
rep = std & ~r0 & b1
result = {'reference': 'R3 EmbLoRA on shared B1', 'unit': 'proportion',
          'bootstrap_draws': 2000, 'bootstrap_seed': 0, 'arms': []}
for name in ('R2_loss0', 'R2_loss0.005'):
    ours = pred[name]
    difference = ours.astype(int) - emb.astype(int)
    saved = (b1 & ~emb & std).astype(int) - (b1 & ~ours & std).astype(int)
    num = np.bincount(paired.ix, weights=difference * rep, minlength=paired.k)
    den = np.bincount(paired.ix, weights=rep, minlength=paired.k)
    bn, bd = paired.w @ num, paired.w @ den
    valid = bd > 0
    retention = [float(num.sum() / den.sum()), *np.quantile(bn[valid] / bd[valid], [.025, .975]).tolist()]
    result['arms'].append({'arm': name,
        'standard_difference_ci': paired.interval(difference, std),
        'upd_difference_ci': paired.interval(difference, ~std),
        'gross_loss_reduction_ci': paired.interval(saved, std),
        'new_repair_retention_difference_ci': retention,
        'bootstrap_nonempty_repair_draws': int(valid.sum())})
path = root / 'comparison/contrasts_vs_emb_lora.json'
path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(result))
