"""Recompute component results and paired intervals from released row records."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ['R0', 'R1', 'R1b', 'R2', 'R2b', 'R1v']
CAPS = [0., .0025, .005, .01]
PAIRS = [('R1b', 'R2b'), ('R1b', 'R1v'), ('R1b', 'R1')]
SEED = 20261116


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def order(score):
    ix = np.argsort(score, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(score[ix]) != 0), len(ix) - 1]
    return ix, ends, np.r_[score[ix][ends], np.inf]


def curve(data, ordering, weights):
    ix, ends, thresholds = ordering
    return (np.r_[0, np.cumsum((weights * data['correct'])[ix])[ends]],
            np.r_[0, np.cumsum((weights * data['gain'])[ix])[ends]], thresholds)


def point(data, values, weights, cap):
    losses, gains, thresholds = values
    allowed = np.flatnonzero(losses <= cap * (weights @ data['std']) + 1e-12)
    j = allowed[np.argmax(gains[allowed])]
    return dict(loss=int(losses[j]), gain=int(gains[j]), threshold=float(thresholds[j]),
                conversion_rate=float(gains[j] / (weights @ data['gain'])))


def common(curves, left, right, ceiling):
    maps = []
    for name in [left, right]:
        losses, gains, _ = curves[name]
        ends = np.r_[np.flatnonzero(np.diff(losses) != 0), len(losses) - 1]
        ends = ends[losses[ends] <= ceiling + 1e-12]
        maps.append(dict(zip(losses[ends].tolist(), gains[ends].tolist())))
    loss = max(set(maps[0]) & set(maps[1]))
    return dict(loss=int(loss), left_gain=int(maps[0][loss]), right_gain=int(maps[1][loss]),
                gain_difference=int(maps[0][loss] - maps[1][loss]))


def circular(data, predictions, draws):
    units = defaultdict(list)
    for i, key in enumerate(zip(data['task'], data['type'], data['family'])):
        units[key].append(i)
    families = sorted(set(zip(data['task'], data['family'])))
    tasks = sorted(set(data['task']))
    clusters = np.array([str(data['groups'][units[(task, 'standard', family)][0]]) for task, family in families])
    groups, ix = np.unique(clusters, return_inverse=True)
    weights = np.random.default_rng(SEED).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), draws)
    values, samples, per_type = {}, {}, {}
    for name, pred in predictions.items():
        hits = np.array([[pred[units[(task, kind, family)]].all() for kind in ['standard', 'upd']]
                         for task, family in families])
        hits = np.column_stack([hits, hits.all(1)])
        values[name] = np.zeros(3)
        samples[name] = np.zeros((draws, 3))
        per_type[name] = {}
        for task in tasks:
            mask = np.array([t == task for t, _ in families])
            denominator = weights @ np.bincount(ix, weights=mask, minlength=len(groups))
            numerator = np.column_stack([np.bincount(ix, weights=mask * hits[:, j], minlength=len(groups)) for j in range(3)])
            per_type[name][task] = hits[mask].mean(0)
            values[name] += per_type[name][task] / len(tasks)
            samples[name] += (weights @ numerator) / denominator[:, None] / len(tasks)
    return values, samples, per_type


def close(actual, expected, context):
    if not np.allclose(actual, expected, rtol=0, atol=1e-10):
        raise AssertionError((context, actual, expected))


def verify_case(root, model, split, expected, draws):
    folder = root / model
    data = load(folder / (split + '_records.npz'))
    n, pool = int(data['std'].sum()), int(data['gain'].sum())
    one = np.ones(len(data['sid']), dtype=int)
    orders = {arm: order(data[arm]) for arm in ARMS}
    curves = {arm: curve(data, orders[arm], one) for arm in ARMS}
    frontier = {arm: {cap: point(data, curves[arm], one, cap) for cap in CAPS} for arm in ARMS}
    policies = {arm: json.loads((folder / 'heads' / arm / 'policy.json').read_text()) for arm in ARMS}
    masks = {arm + '/' + budget: data[arm] < float(value['threshold'])
             for arm in ARMS for budget, value in policies[arm].items()}
    predictions = {'B1': data['base_correct']}
    predictions.update({name: np.where(mask, ~data['std'], data['base_correct']) for name, mask in masks.items()})
    predictions.update({key.removeprefix('prediction::'): value for key, value in data.items() if key.startswith('prediction::')})
    values, circ_samples, typed = circular(data, predictions, draws)
    count = 0
    for row in expected['summary']:
        arm, cap = row['method'], float(row['budget'])
        if row['protocol'] == 'realized_loss_ceiling':
            actual = frontier[arm][cap]
        else:
            name = arm + '/' + str(cap)
            mask, pred = masks[name], predictions[name]
            actual = dict(loss=int((mask & data['correct']).sum()), gain=int((mask & data['gain']).sum()),
                conversion_rate=float((mask & data['gain']).sum() / pool), row_standard=pred[data['std']].mean(),
                row_upd=pred[~data['std']].mean(), **dict(zip(['circular_standard', 'circular_upd', 'circular_dual'], values[name])))
        for key in ['loss', 'gain', 'conversion_rate', 'row_standard', 'row_upd', 'circular_standard', 'circular_upd', 'circular_dual']:
            if row.get(key):
                close(actual[key], float(row[key]), (model, split, arm, row['protocol'], key))
                count += 1
    kinds = np.array([task.rsplit('_', 1)[-1].lower() for task in data['task']])
    for row in expected['by_type']:
        arm, kind = row['method'], row['kind']
        threshold = frontier[arm][.005]['threshold'] if row['protocol'] == 'frontier_0.005' else float(policies[arm]['0.005']['threshold'])
        mask = data[arm] < threshold
        subset = kinds == kind
        pred = np.where(mask, ~data['std'], data['base_correct'])
        _, _, cv = circular(data, {'p': pred}, 1)
        task = next(task for task in cv['p'] if task.rsplit('_', 1)[-1].lower() == kind)
        actual = dict(answerable=int((subset & data['std']).sum()), gain_pool=int((subset & data['gain']).sum()),
            new_abstentions=int((subset & data['gain'] & mask).sum()), loss=int((subset & data['correct'] & mask).sum()),
            conversion_rate=float((subset & data['gain'] & mask).sum() / (subset & data['gain']).sum()),
            row_standard=pred[subset & data['std']].mean(), row_upd=pred[subset & ~data['std']].mean(),
            **dict(zip(['circular_standard', 'circular_upd', 'circular_dual'], cv['p'][task])))
        for key, value in actual.items():
            close(value, float(row[key]), (model, split, arm, kind, key)); count += 1
    for row in expected['circular']:
        name = row['method']
        for j, metric in enumerate(['standard', 'upd', 'dual']):
            close(values[name][j], float(row['circular_' + metric]), (model, split, name, metric)); count += 1
        if draws == 2000:
            close(np.quantile(circ_samples[name][:, 2], [.025, .975]), json.loads(row['circular_dual_ci95']), (name, 'Dual CI')); count += 2
    # A single seeded set of image-cluster weights is shared by all score contrasts.
    groups, ix = np.unique(data['groups'], return_inverse=True)
    weights = np.random.default_rng(SEED).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), draws)
    paired_samples = defaultdict(list)
    for group_weight in weights:
        w = group_weight[ix]
        current = {arm: curve(data, orders[arm], w) for arm in ARMS}
        points = {arm: {cap: point(data, current[arm], w, cap) for cap in CAPS} for arm in ARMS}
        for left, right in PAIRS:
            for cap in CAPS:
                paired_samples[(left, right, 'realized_loss_ceiling', cap, 'conversion_rate')].append(
                    points[left][cap]['conversion_rate'] - points[right][cap]['conversion_rate'])
            paired_samples[(left, right, 'exact_common_loss', .005, 'conversion_rate')].append(
                common(current, left, right, np.floor(.005 * (w @ data['std']) + 1e-12))['gain_difference'] / (w @ data['gain']))
            for metric, label in [('correct', 'loss_rate'), ('gain', 'gain_rate')]:
                denom = w @ (data['std'] if metric == 'correct' else data['gain'])
                paired_samples[(left, right, 'frozen_deployment', .005, label)].append(
                    (w @ (data[metric] & masks[left + '/0.005']) - w @ (data[metric] & masks[right + '/0.005'])) / denom)
            for kind in sorted(set(kinds)):
                a, b = [data[arm] < points[arm][.005]['threshold'] for arm in [left, right]]
                gain = data['gain'] & (kinds == kind)
                paired_samples[(left, right, 'typed_realized_loss_ceiling', .005, kind + '_conversion_rate')].append(
                    (w @ (gain & a) - w @ (gain & b)) / (w @ gain))
    for row in expected['paired']:
        left, right, protocol, cap, metric = row['left'], row['right'], row['protocol'], float(row['budget']), row['metric']
        if metric.startswith('circular_'):
            j = ['standard', 'upd', 'dual'].index(metric.removeprefix('circular_'))
            samples = circ_samples[left + '/' + str(cap)][:, j] - circ_samples[right + '/' + str(cap)][:, j]
            actual = values[left + '/' + str(cap)][j] - values[right + '/' + str(cap)][j]
        else:
            samples = paired_samples[(left, right, protocol, cap, metric)]
            if protocol == 'realized_loss_ceiling':
                actual = frontier[left][cap]['conversion_rate'] - frontier[right][cap]['conversion_rate']
            elif protocol == 'exact_common_loss':
                value = common(curves, left, right, np.floor(cap * n + 1e-12))
                for key in ['loss', 'left_gain', 'right_gain', 'gain_difference']:
                    close(value[key], float(row[key]), (left, right, key)); count += 1
                actual = value['gain_difference'] / pool
            elif protocol == 'frozen_deployment':
                key, denom = ('correct', n) if metric == 'loss_rate' else ('gain', pool)
                actual = ((data[key] & masks[left + '/' + str(cap)]).sum() - (data[key] & masks[right + '/' + str(cap)]).sum()) / denom
            else:
                kind = metric.removesuffix('_conversion_rate')
                gain = data['gain'] & (kinds == kind)
                a, b = [data[arm] < frontier[arm][cap]['threshold'] for arm in [left, right]]
                actual = ((gain & a).sum() - (gain & b).sum()) / gain.sum()
        close(actual, float(row['difference']), (model, split, left, right, protocol, metric)); count += 1
        if draws == 2000:
            close(np.quantile(samples, [.025, .975]), json.loads(row['ci95']), (model, split, protocol, metric, 'CI')); count += 2
    for row in expected['ledger']:
        name = row['method'].split(' / ')[0]
        if name == 'B1 fresh':
            pred = data['base_correct']
        elif name in predictions:
            pred = predictions[name]
        else:
            continue
        reference = data['base_correct'] if row['reference'] == 'fresh' else predictions['B1 archived']
        actual = dict(original_correct=int((reference & data['std']).sum()),
            loss=int((data['std'] & reference & ~pred).sum()), repairs=int((data['std'] & ~reference & pred).sum()),
            row_standard=pred[data['std']].mean(), row_upd=pred[~data['std']].mean(), circular_dual=values['B1' if name == 'B1 fresh' else name][2])
        for key, value in actual.items():
            close(value, float(row[key]), (model, split, name, row['reference'], key)); count += 1
    for row in expected['native_paired_contrasts']:
        left, right, metric = row['method'], row['comparator'], row['metric']
        j = ['standard', 'upd', 'dual'].index(metric)
        close(values[left][j] - values[right][j], float(row['difference']), (left, right, metric)); count += 1
        if draws == 2000:
            close(np.quantile(circ_samples[left][:, j] - circ_samples[right][:, j], [.025, .975]),
                  [float(row['ci_low']), float(row['ci_high'])], (left, right, metric, 'CI')); count += 2
    return dict(model=model, split=split, comparisons=count, rows=len(data['sid']), bootstrap_draws=draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/two_block'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    tables = {name: read_csv(args.root / 'expected' / (name + '.csv')) for name in
        ['summary', 'paired', 'by_type', 'circular', 'ledger', 'native_paired_contrasts']}
    for model in ['qwen', 'internvl']:
        for split in ['E', 'I']:
            expected = {name: [row for row in rows if row['model'] == model and row['split'] == split]
                        for name, rows in tables.items()}
            result = verify_case(args.root, model, split, expected, args.bootstrap)
            print(json.dumps(result), flush=True)
    message = ('Released row records reproduce the component tables, ledgers and paired intervals.'
               if args.bootstrap == 2000 else
               'Released row records reproduce the point estimates; full interval verification uses --bootstrap 2000.')
    print(message, flush=True)


if __name__ == '__main__':
    main()
