"""Fit full-vocabulary LVLM-LP and hidden-state controls on a common split."""
import argparse
import json
from pathlib import Path
import random
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def save(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def fit(x, y, c, standardize):
    mean = x.mean(0) if standardize else np.zeros(x.shape[1])
    sd = np.maximum(x.std(0, ddof=1), 1e-6) if standardize else np.ones(x.shape[1])
    xx = (x - mean) / sd if standardize else x
    model = LogisticRegression(C=c, max_iter=2000, tol=1e-4, solver='lbfgs')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(xx, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert list(model.classes_) == [0, 1]
    w = model.coef_[0] / sd
    b = float(model.intercept_[0] - mean @ w)
    return w, b, {'C': c, 'iterations': int(model.n_iter_[0]), 'converged': converged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--train-split', default='T')
    ap.add_argument('--feature', choices=['logits', 'hidden', 'hidden_raw'], required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    rows, arrays, training_arrays = [], [], []
    for cache in args.cache:
        rr = json.loads((cache / 'inputs.json').read_text(encoding='utf-8'))['rows']
        state = json.loads((cache / 'status.json').read_text(encoding='utf-8'))
        assert state['completed'] == len(rr), (cache, 'feature extraction incomplete')
        feature_file = 'hidden' if args.feature == 'hidden_raw' else args.feature
        x = np.load(cache / f'{feature_file}.npy', mmap_mode='r')
        assert len(x) == len(rr)
        train_mask = np.array([r['split'] == args.train_split for r in rr])
        if train_mask.any():
            training_arrays.append(np.asarray(x[train_mask], dtype=np.float64, order='C'))
        arrays.append(x)
        rows.extend(rr)
    assert len({r['sid'] for r in rows}) == len(rows)
    idx = np.array([i for i, r in enumerate(rows) if r['split'] == args.train_split])
    y = np.array([int(rows[i]['type'] == 'standard') for i in idx])
    clusters = [str(rows[i]['cluster']) for i in idx]
    groups = sorted(set(clusters))
    assert len(groups) >= 5 and set(y) == {0, 1}
    random.Random(42).shuffle(groups)
    folds = {g: i % 5 for i, g in enumerate(groups)}
    fold = np.array([folds[g] for g in clusters])
    xt = training_arrays[0] if len(training_arrays) == 1 else np.concatenate(training_arrays)
    standardize = args.feature == 'hidden'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'results.json').exists(), 'completed fit exists'
    save(out / 'folds.json', {'train_sid': [rows[i]['sid'] for i in idx], 'cluster_fold': folds})
    default = fit(xt, y, 1., standardize)
    cv = []
    for c in [.01, .1, 1., 10., 100.]:
        losses, optim = [], []
        for k in range(5):
            train, val = fold != k, fold == k
            w, b, details = fit(xt[train], y[train], c, standardize)
            z = xt[val] @ w + b
            # Stable binary cross entropy on uncalibrated logits.
            losses.append(float(np.mean(np.logaddexp(0, z) - y[val] * z)))
            optim.append(details)
        cv.append({'C': c, 'mean_bce': float(np.mean(losses)), 'fold_bce': losses, 'optimizer': optim})
        print(json.dumps({'feature': args.feature, 'C': c, 'cv_bce': cv[-1]['mean_bce'],
                          'seconds': time.perf_counter() - started}), flush=True)
    best = min(cv, key=lambda r: (r['mean_bce'], r['C']))['C']
    selected = default if best == 1 else fit(xt, y, best, standardize)
    outputs = {}
    for name, (w, b, details) in [('published_C1', default), ('train_cv', selected)]:
        scores = np.empty(len(rows), np.float64)
        offset = 0
        for x in arrays:
            for start in range(0, len(x), 128):
                stop = min(start + 128, len(x))
                scores[offset + start:offset + stop] = np.asarray(x[start:stop], dtype=np.float64) @ w + b
            offset += len(x)
        assert np.isfinite(scores).all()
        np.savez_compressed(out / f'{name}.npz', coef=w, intercept=np.array(b), logits=scores,
                            sid=np.array([r['sid'] for r in rows]))
        outputs[name] = details
    save(out / 'results.json', {'feature': args.feature, 'standardized': standardize,
        'train_rows': len(idx), 'train_clusters': len(groups), 'selected_C': best,
        'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-4,
        'cv': cv, 'outputs': outputs, 'seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
