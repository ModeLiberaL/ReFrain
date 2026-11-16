"""Fit, calibrate, and apply ReFrain scenario adaptation from cached features."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ARMS = {
    'R0': ([], True, [0]),
    'R1': (['final_state'], True, [0] + GRID),
    'R1b': (['final_state', 'question_state'], True, [0] + GRID),
    'R2': (['final_state'], False, GRID),
    'R2b': (['final_state', 'question_state'], False, GRID),
    'R1v': (['final_state', 'image_state'], True, [0] + GRID),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value):
    return float(value) if np.isfinite(value) else ('inf' if value > 0 else '-inf')


def load_features(path, blocks):
    with np.load(path) as data:
        result = {key: data[key] for key in ['sid', 'source_score'] + list(blocks)}
    n = len(result['sid'])
    if any(len(value) != n for value in result.values()):
        raise ValueError('All feature arrays must have the same row count.')
    if len(set(result['sid'].tolist())) != n:
        raise ValueError('Sample IDs must be unique.')
    if not all(np.isfinite(result[key]).all() for key in ['source_score'] + list(blocks)):
        raise ValueError('Features and source scores must be finite.')
    return result


def matrix(data, blocks, part=slice(None)):
    if not blocks:
        return np.zeros((len(data['sid'][part]), 0))
    return np.concatenate([np.asarray(data[key][part], dtype=np.float64) for key in blocks], axis=1)


def standardize(x):
    return x.mean(0), np.maximum(x.std(0, ddof=1), 1e-6)


def intercept(offset, y):
    limit = float(np.max(np.abs(offset))) + 50.
    return float(brentq(lambda c: (expit(offset + c) - y).sum(), -limit, limit, xtol=1e-12))


def optimize(x, y, offset, c):
    if c == 0:
        return np.zeros(x.shape[1]), intercept(offset, y), dict(converged=True, iterations=0)
    lam = 1 / c
    scale = np.sqrt(lam + .25 * np.square(x).sum(0))
    bscale = np.sqrt(len(y) * .25)

    def objective(u):
        w, b = u[:-1] / scale, u[-1] / bscale
        z = offset + x @ w + b
        error = expit(z) - y
        loss = np.sum(np.logaddexp(0., z) - y * z) + .5 * lam * (w @ w)
        gradient = np.r_[(x.T @ error + lam * w) / scale, error.sum() / bscale]
        return float(loss), gradient

    initial = np.zeros(x.shape[1] + 1)
    initial[-1] = intercept(offset, y) * bscale
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
        options=dict(maxiter=500, ftol=1e-12, gtol=1e-7, maxls=40))
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(str(result.message))
    return result.x[:-1] / scale, float(result.x[-1] / bscale), dict(
        converged=True, iterations=int(result.nit), objective=float(result.fun),
        gradient_inf=float(np.abs(result.jac).max()))


def predict(data, head):
    blocks = head['blocks'].tolist()
    z = (data['source_score'].astype(np.float64).copy() if bool(head['use_source'])
         else np.zeros(len(data['sid'])))
    z += float(head['intercept'])
    if float(head['C']) > 0:
        for start in range(0, len(z), 128):
            part = slice(start, start + 128)
            x = matrix(data, blocks, part)
            z[part] += ((x - head['mean']) / head['scale']) @ head['coef']
    return z


def calibrate(a_scores, b_scores, std, correct, budgets):
    if not std.any():
        raise ValueError('Calibration requires answerable examples.')
    candidates = np.r_[-np.inf, np.unique(a_scores), np.inf]
    losses = np.searchsorted(np.sort(b_scores[std & correct]), candidates, side='left')
    policies = {}
    for budget in budgets:
        if not np.isfinite(budget) or not 0 <= budget <= 1:
            raise ValueError('Each loss budget must be in [0, 1].')
        j = np.flatnonzero(losses <= budget * std.sum() + 1e-12)[-1]
        policies[str(float(budget))] = dict(threshold=number(candidates[j]),
            calibration_losses=int(losses[j]), calibration_answerable=int(std.sum()))
    return policies


def load_head(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def fit_command(args):
    started = time.monotonic()
    blocks, use_source, grid = ARMS[args.arm]
    a = load_features(args.adaptation, blocks)
    b = load_features(args.calibration, blocks)
    with np.load(args.adaptation) as data:
        y, groups, folds = data['std'].astype(float), data['groups'].astype(str), data['folds'].astype(int)
    with np.load(args.calibration) as data:
        std, correct, b_groups = data['std'].astype(bool), data['correct'].astype(bool), data['groups'].astype(str)
    if set(groups) & set(b_groups):
        raise ValueError('Adaptation and calibration image groups must be disjoint.')
    if len(y) != len(a['sid']) or len(folds) != len(y) or set(folds) != set(range(5)):
        raise ValueError('Adaptation requires one label and one fold in 0..4 per row.')
    x = matrix(a, blocks)
    offset = a['source_score'].astype(float) if use_source else np.zeros(len(y))
    losses = {c: [] for c in grid}
    args.output.mkdir(parents=True, exist_ok=False)
    for k in range(5):
        tr, va = folds != k, folds == k
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('An image group appears in both fold training and validation.')
        if set(y[tr]) != {0., 1.}:
            raise ValueError('Every fold training set must contain both answerability classes.')
        mean, sd = standardize(x[tr])
        xt, xv = (x[tr] - mean) / sd, (x[va] - mean) / sd
        for c in grid:
            w, bias, _ = optimize(xt, y[tr], offset[tr], c)
            z = offset[va] + xv @ w + bias
            losses[c].append(float(np.mean(np.logaddexp(0., z) - y[va] * z)))
        print(json.dumps(dict(arm=args.arm, fold=k, seconds=time.monotonic() - started)), flush=True)
    cv = [dict(C=c, lambda_value=number(1 / c if c else np.inf),
               fold_bce=losses[c], mean_bce=float(np.mean(losses[c]))) for c in grid]
    chosen = min(cv, key=lambda item: (item['mean_bce'], item['C']))
    mean, sd = standardize(x)
    coef, bias, optimizer = optimize((x - mean) / sd, y, offset, chosen['C'])
    head = dict(coef=coef, intercept=np.array(bias), mean=mean, scale=sd,
        C=np.array(chosen['C']), blocks=np.array(blocks, dtype=str), use_source=np.array(use_source))
    a_scores, b_scores = predict(a, head), predict(b, head)
    np.savez_compressed(args.output / 'head.npz', **head)
    np.savez_compressed(args.output / 'A_scores.npz', sid=a['sid'], score=a_scores)
    np.savez_compressed(args.output / 'B_scores.npz', sid=b['sid'], score=b_scores)
    write_json(args.output / 'selection.json', dict(chosen, arm=args.arm, blocks=blocks,
        use_source=use_source, dimensions=x.shape[1], cv=cv, optimizer=optimizer))
    write_json(args.output / 'policy.json', calibrate(a_scores, b_scores, std, correct, args.budgets))
    write_json(args.output / 'calibration_frozen.json', dict(state='complete',
        frozen_at=datetime.now(timezone.utc).isoformat(), adaptation_sid=a['sid'].tolist(),
        calibration_sid=b['sid'].tolist(), folds=folds.tolist(),
        adaptation_rows=len(y), calibration_rows=len(std), seconds=time.monotonic() - started,
        objective='sum BCE + lambda ||v||^2 / 2; intercept unpenalized',
        normalization='training-fold mean and sample std, floor 1e-6',
        selection='mean of five image-group fold BCEs; ties prefer stronger shrinkage',
        calibration='A-score candidates plus both infinities; strict score < threshold veto'))
    print(json.dumps(chosen), flush=True)


def calibrate_command(args):
    head = load_head(args.head)
    blocks = head['blocks'].tolist()
    a, b = load_features(args.adaptation, blocks), load_features(args.calibration, blocks)
    with np.load(args.calibration) as data:
        std, correct = data['std'].astype(bool), data['correct'].astype(bool)
    write_json(args.output, calibrate(predict(a, head), predict(b, head), std, correct, args.budgets))


def apply_command(args):
    head = load_head(args.head)
    data = load_features(args.features, head['blocks'].tolist())
    scores = predict(data, head)
    policy = read_json(args.policy)[str(float(args.budget))]
    threshold = float(policy['threshold'])
    answers = [json.loads(line) for line in args.answers.read_text(encoding='utf-8').splitlines() if line.strip()]
    by = {record['sid']: record for record in answers}
    if set(by) != set(data['sid'].tolist()):
        raise ValueError('Answer IDs must match feature IDs.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for sid, value in zip(data['sid'].tolist(), scores):
            answer = by[sid]
            keep = bool(value >= threshold)
            record = dict(sid=sid, response=answer['response'] if keep else answer.get('nota', args.abstention),
                          score=float(value), decision='answer' if keep else 'abstain')
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Select and fit a scenario head on A, then calibrate on B.')
    fit.add_argument('--arm', choices=ARMS, default='R1b')
    fit.add_argument('--adaptation', type=Path, required=True)
    fit.add_argument('--calibration', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--budgets', type=float, nargs='+', default=[0., .005])
    fit.set_defaults(run=fit_command)
    calibration = commands.add_parser('calibrate', help='Change the loss budget without fitting the head.')
    calibration.add_argument('--head', type=Path, required=True)
    calibration.add_argument('--adaptation', type=Path, required=True)
    calibration.add_argument('--calibration', type=Path, required=True)
    calibration.add_argument('--budgets', type=float, nargs='+', required=True)
    calibration.add_argument('--output', type=Path, required=True)
    calibration.set_defaults(run=calibrate_command)
    apply = commands.add_parser('apply', help='Route cached original answers without target labels.')
    apply.add_argument('--head', type=Path, required=True)
    apply.add_argument('--features', type=Path, required=True)
    apply.add_argument('--policy', type=Path, required=True)
    apply.add_argument('--budget', type=float, required=True)
    apply.add_argument('--answers', type=Path, required=True)
    apply.add_argument('--abstention', default='unanswerable', help='Open-answer refusal; per-row nota takes precedence.')
    apply.add_argument('--output', type=Path, required=True)
    apply.set_defaults(run=apply_command)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
