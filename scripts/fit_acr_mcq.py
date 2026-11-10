"""ACR mechanism adapted to shared B1 hidden states and A-E option logits."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from evaluation import Paired, circular, load_reference
from x2_score import native_parser, score

torch.set_num_threads(2)
torch.manual_seed(42)
ap = argparse.ArgumentParser()
ap.add_argument('--standardize', action='store_true')
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
variant = 'acr_mcq_standardized' if args.standardize else 'acr_mcq'
method_name = 'ACR-MCQ standardized' if args.standardize else 'ACR-MCQ'
out = root / variant
out.mkdir(exist_ok=True)
parser = native_parser(args.upd_code_root)
sets = {s: load_reference(root, 'qwen', s, parser, score) for s in ('T', 'CM', 'E')}
tokenizer = AutoTokenizer.from_pretrained(root / 'sft/B1', local_files_only=True)
ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCDE']
assert all(len(seq) == 1 for seq in ids)
ids = [seq[0] for seq in ids]
x, msp = {}, {}
for split, data in sets.items():
    assert all(set(dict(r['display'])) <= set('ABCDE') for r in data['rows'])
    z = np.load(root / 'qwen_base' / split / 'logits.npy', mmap_mode='r')[:, ids].copy()
    h = np.load(root / 'qwen_base' / split / 'hidden.npy')
    x[split] = torch.from_numpy(np.concatenate([h, z], axis=1))
    msp[split] = torch.from_numpy(z).softmax(-1).max(-1).values
if args.standardize:
    mean = x['T'].mean(0)
    scale = x['T'].std(0, correction=0)
    scale[scale == 0] = 1.
    x = {s: (values - mean) / scale for s, values in x.items()}
    torch.save(dict(mean=mean, scale=scale), out / 'feature_scaler.pt')
y = torch.tensor(sets['T']['correct'], dtype=torch.float32)
dim = x['T'].shape[1]


class ACR(nn.Module):
    def __init__(self):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(.1),
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(.1), nn.Linear(dim, 1), nn.Sigmoid())
        self.risk, self.gate = head(), head()

    def forward(self, features, confidence):
        risk = self.risk(features).squeeze(-1)
        gate = self.gate(features).squeeze(-1)
        return gate * confidence + (1 - gate) * risk


model = ACR()
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, betas=(.95, .999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=.1)
started = time.time()
config = dict(method=method_name, source='https://arxiv.org/html/2602.04924',
    feature='2048-d final prompt hidden + 5 A-E raw logits; fixed A-E alphabet',
    base='shared B1', rows=len(y), labels='B1 actual response correctness',
    seed=42, epochs=15, batch_size=32, optimizer='Adam', lr=8e-5, betas=[.95, .999],
    scheduler='StepLR(8, gamma=0.1)', dropout=.1, head_width=dim,
    parameters=sum(p.numel() for p in model.parameters()),
    standardization='T-only mean/std for learned heads; unscaled MSP' if args.standardize else 'none',
    adaptation='image MCQ replaces audio-video QA; answer alphabet and head width change; fixed QA-TIGER schedule',
    selection='no E tuning; all T training; CM only threshold calibration')
(out / 'config.json').write_text(json.dumps(config, indent=2))
for epoch in range(15):
    model.train()
    total = 0.
    for batch in torch.randperm(len(y)).split(32):
        optimizer.zero_grad()
        p = model(x['T'][batch], msp['T'][batch])
        loss = F.binary_cross_entropy(p, y[batch])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    scheduler.step()
    state = dict(state='training', epoch=epoch + 1, loss=total / len(y), seconds=time.time() - started)
    (out / 'status.json').write_text(json.dumps(state))
    print(json.dumps(state), flush=True)
model.eval()
with torch.no_grad():
    confidence = {s: torch.cat([model(x[s][i:i+256], msp[s][i:i+256])
        for i in range(0, len(x[s]), 256)]).numpy() for s in sets}
torch.save(model.state_dict(), out / 'model.pt')
np.savez_compressed(out / 'scores.npz', **confidence)
cm, ev = sets['CM'], sets['E']
std, base = ev['std'], ev['correct']
paired = Paired(ev['rows'])
prior = np.load(root / 'comparison/predictions.npz')
assert prior['sid'].tolist() == [r['sid'] for r in ev['rows']]
rep = std & ~prior['R0'] & base
arms, predictions = [], {}
for method, scores in [(method_name, confidence), ('MSP_A-E', {s: msp[s].numpy() for s in sets})]:
    grid = np.r_[-np.inf, np.unique(scores['T']), np.inf]
    losses = np.searchsorted(np.sort(scores['CM'][cm['std'] & cm['correct']]), grid, side='left')
    for name, budget in [('default_threshold', None), ('loss_0.0', 0.), ('loss_0.005', .005)]:
        tau = .5 if budget is None else float(grid[np.flatnonzero(losses / cm['std'].sum() <= budget + 1e-12)[-1]])
        pred = np.where(scores['E'] >= tau, base, ~std)
        predictions[method + '/' + name] = pred
        diff = pred.astype(int) - base.astype(int)
        sci, uci = paired.interval(diff, std), paired.interval(diff, ~std)
        ours_diff = prior['R2_loss0.005'].astype(int) - pred.astype(int)
        arms.append(dict(method=method, point=name, budget=budget,
            threshold=tau if np.isfinite(tau) else str(tau),
            calibration_lost=int(((scores['CM'] < tau) & cm['std'] & cm['correct']).sum()),
            standard=float(pred[std].mean()), upd=float(pred[~std].mean()),
            circular_dual=circular(ev['rows'], pred)['macro']['dual'],
            gross_loss=int((std & base & ~pred).sum()), repairs=int((std & ~base & pred).sum()),
            retained_repairs=int((rep & pred).sum()), repair_total=int(rep.sum()),
            standard_change_ci=sci, upd_change_ci=uci,
            passes_original_rule=bool(sci[1] > -.01 and uci[1] > 0),
            ours_half_percent_minus_control_standard_ci=paired.interval(ours_diff, std),
            ours_half_percent_minus_control_upd_ci=paired.interval(ours_diff, ~std)))
result = dict(state='complete', config=config, training_seconds=time.time() - started,
    correctness_metrics={s: dict(auroc=float(roc_auc_score(sets[s]['correct'], confidence[s])),
        brier=float(np.mean((confidence[s] - sets[s]['correct']) ** 2))) for s in sets}, arms=arms)
with torch.no_grad():
    gate = model.gate(x['T']).flatten().numpy()
    result['training_gate_quantiles'] = np.quantile(gate, [0, .25, .5, .75, 1]).tolist()
result_dir = root / 'comparison' / variant
result_dir.mkdir(exist_ok=True)
(result_dir / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
np.savez_compressed(result_dir / 'predictions.npz', sid=prior['sid'], **predictions)
(out / 'status.json').write_text(json.dumps(dict(state='complete', seconds=time.time() - started)))
print(json.dumps(result), flush=True)
