"""BICR on shared B1: official loss/search space, T-only model selection and refit."""
import copy
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

from evaluation import Paired, circular, load_reference
from x2_score import native_parser, score

torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
out = root / 'bicr'
parser = native_parser(args.upd_code_root)
sets = {s: load_reference(root, 'qwen', s, parser, score) for s in ('T', 'CM', 'E')}
x = {s: torch.from_numpy(np.load(root / 'qwen_base' / s / 'hidden.npy')) for s in sets}
blank = torch.from_numpy(np.load(out / 'black_T.npy'))
assert json.loads((out / 'black_inputs.json').read_text(encoding='utf-8'))['sid'] == [r['sid'] for r in sets['T']['rows']]
y = torch.tensor(sets['T']['correct'], dtype=torch.float32)
groups = [r['cluster'] for r in sets['T']['rows']]
train, validation = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x['T'], y, groups))
assert not set(np.array(groups)[train]) & set(np.array(groups)[validation])
(out / 'selection_split.json').write_text(json.dumps(dict(seed=42,
    train_sid=[sets['T']['rows'][i]['sid'] for i in train],
    validation_sid=[sets['T']['rows'][i]['sid'] for i in validation])))


def metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    ece = 0.
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities > lo) & (probabilities <= hi)
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return dict(auroc=float(roc_auc_score(labels, probabilities)), ece=float(ece),
                brier=float(np.mean((probabilities - labels) ** 2)))


def create_model(config):
    dims = [x['T'].shape[1]] + ([] if config['layers'] == '0' else [int(v) for v in config['layers'].split(',')])
    modules = []
    for a, b in zip(dims[:-1], dims[1:]):
        modules.extend([nn.Linear(a, b), nn.ReLU()])
        if config['dropout']:
            modules.append(nn.Dropout(config['dropout']))
    modules.append(nn.Linear(dims[-1], 1))
    model = nn.Sequential(*modules)
    assert sum(p.numel() for p in model.parameters()) <= 5000000
    return model


def train_model(config, indices, epochs, trial=None):
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    pos_weight = (1 - y[indices]).sum() / y[indices].sum()
    indices = torch.as_tensor(indices)
    best, stale, best_epoch, best_state = -np.inf, 0, 0, None
    for epoch in range(epochs):
        model.train()
        permutation = indices[torch.randperm(len(indices))]
        for batch in permutation.split(32):
            optimizer.zero_grad()
            z = model(x['T'][batch]).squeeze(-1)
            p = z.sigmoid()
            p_black = model(blank[batch]).squeeze(-1).sigmoid()
            target = y[batch]
            loss = F.binary_cross_entropy_with_logits(z, target, pos_weight=pos_weight)
            loss += config['beta_brier'] * ((p - target) ** 2).mean()
            loss += config['lambda_rank'] * (F.relu(config['margin'] - p + p_black) * target).sum() / (target.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if trial is not None:
            model.eval()
            with torch.no_grad():
                probs = model(x['T'][validation]).squeeze(-1).sigmoid().numpy()
            m = metrics(y[validation].numpy(), probs)
            value = .6 * m['auroc'] + .4 * (1 - m['ece'])
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if value > best:
                best, stale, best_epoch = value, 0, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 20:
                break
    if trial is not None:
        model.load_state_dict(best_state)
        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('best_score', best)
    return model, best


def objective(trial):
    config = dict(layers=trial.suggest_categorical('layers',
        ['0', '256', '512', '128,64', '256,128', '512,256', '1024,512', '1024,512,256']),
        dropout=trial.suggest_categorical('dropout', [0., .1, .3, .5]),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        beta_brier=trial.suggest_float('beta_brier', 0., .5),
        lambda_rank=trial.suggest_float('lambda_rank', .01, .3),
        margin=trial.suggest_float('margin', .05, .25))
    _, value = train_model(config, train, 200, trial)
    return value


started = time.time()
np.random.seed(42)
torch.manual_seed(42)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5))
def record_progress(study, trial):
    (out / 'fit_status.json').write_text(json.dumps(dict(state='searching', completed_trials=len(study.trials),
        total_trials=50, seconds=time.time() - started, best_value=study.best_value)))
study.optimize(objective, n_trials=50, callbacks=[record_progress])
epochs = study.best_trial.user_attrs['best_epoch']
torch.manual_seed(42)
model, _ = train_model(study.best_params, np.arange(len(y)), epochs)
model.eval()
torch.save(dict(state_dict=model.state_dict(), config=study.best_params), out / 'model.pt')
with torch.no_grad():
    scores = {s: torch.cat([model(batch).squeeze(-1).sigmoid() for batch in values.split(256)]).numpy()
              for s, values in x.items()}
fit = dict(method='BICR', base='shared B1', labels='B1 actual response correctness, including correct UPD refusal',
    selection='T image-cluster 80/20 split; 50 Optuna trials; AUROC/ECE composite; full-T refit',
    seed=42, trials=50, epochs=epochs, params=study.best_params, validation_score=study.best_value,
    train_rows=len(train), validation_rows=len(validation),
    correctness_metrics={s: metrics(sets[s]['correct'], scores[s]) for s in sets},
    training_seconds=time.time() - started,
    source='https://github.com/Ledengary/BICR',
    adaptation='Qwen2.5-VL-3B B1, T1354, one seed; paper layer candidates including 128,64; common NOTA executor')
(out / 'fit.json').write_text(json.dumps(fit, indent=2))
np.savez_compressed(out / 'scores.npz', **scores)
cm, ev = sets['CM'], sets['E']
std, base = ev['std'], ev['correct']
prior = np.load(root / 'comparison/predictions.npz')
assert prior['sid'].tolist() == [r['sid'] for r in ev['rows']]
rep = std & ~prior['R0'] & base
paired = Paired(ev['rows'])
grid = np.r_[-np.inf, np.unique(scores['T']), np.inf]
losses = np.searchsorted(np.sort(scores['CM'][cm['std'] & cm['correct']]), grid, side='left')
arms, predictions = [], {}
for name, budget in [('default_threshold', None), ('loss_0.0', 0.), ('loss_0.005', .005)]:
    tau = .5 if budget is None else float(grid[np.flatnonzero(losses / cm['std'].sum() <= budget + 1e-12)[-1]])
    pred = np.where(scores['E'] >= tau, base, ~std)
    predictions[name] = pred
    delta = pred.astype(int) - base.astype(int)
    sci, uci = paired.interval(delta, std), paired.interval(delta, ~std)
    diff = prior['R2_loss0.005'].astype(int) - pred.astype(int)
    arms.append(dict(method='BICR_common_NOTA', point=name, budget=budget,
        threshold=tau if np.isfinite(tau) else str(tau),
        calibration_lost=int(((scores['CM'] < tau) & cm['std'] & cm['correct']).sum()),
        standard=float(pred[std].mean()), upd=float(pred[~std].mean()),
        circular_dual=circular(ev['rows'], pred)['macro']['dual'],
        gross_loss=int((std & base & ~pred).sum()), repairs=int((std & ~base & pred).sum()),
        retained_repairs=int((rep & pred).sum()), repair_total=int(rep.sum()),
        standard_change_ci=sci, upd_change_ci=uci,
        passes_original_rule=bool(sci[1] > -.01 and uci[1] > 0),
        ours_half_percent_minus_BICR_standard_ci=paired.interval(diff, std),
        ours_half_percent_minus_BICR_upd_ci=paired.interval(diff, ~std)))
results = dict(state='complete', fit=fit, arms=arms)
result_dir = root / 'comparison/bicr'
result_dir.mkdir(exist_ok=True)
(result_dir / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
np.savez_compressed(result_dir / 'predictions.npz', sid=prior['sid'], **predictions)
(out / 'fit_status.json').write_text(json.dumps(dict(state='complete', seconds=time.time() - started)))
print(json.dumps(results), flush=True)
