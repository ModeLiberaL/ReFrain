"""Recompute the complete-module VizWiz confirmation from compact row records."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_scenario_records import order, curve, point, common, close, load

CAPS = [0., .0025, .005, .01]
ARMS = ['R0', 'R1', 'R1b', 'R2b']
CONTROLS = ['R0', 'R1', 'R2b', 'MSP', 'negative_entropy', 'LVLM_LP_C1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/vizwiz_full'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    d = load(args.root / 'H_records.npz')
    expected = json.loads((args.root / 'results.json').read_text())
    policy = json.loads((args.root / 'calibration.json').read_text())
    names = ARMS + CONTROLS[3:]
    n, pool = int(d['std'].sum()), int(d['gain'].sum())
    close([len(d['sid']), n, d['correct'].sum(), pool], [4000, 3043, 1630, 602], 'counts')
    if len(set(d['sid'].tolist())) != len(d['sid']):
        raise AssertionError('Each image must contribute one unique question.')
    one = np.ones(len(d['sid']), dtype=int)
    orders = {name: order(d[name]) for name in names}

    def points(weights):
        return {name: {str(cap): point(d, curve(d, orders[name], weights), weights, cap)
                       for cap in CAPS} for name in names}

    full = points(one)
    curves = {name: curve(d, orders[name], one) for name in names}
    for name in names:
        for cap in CAPS:
            actual, ref = full[name][str(cap)], expected['frontier'][name][str(cap)]
            close([actual['gain'], actual['loss'], actual['conversion_rate']],
                  [ref['gain'], ref['loss'], ref['rate']], (name, cap))
    masks = {name+'/'+cap: d[name] < float(policy[name][cap]['threshold'])
             for name in ARMS for cap in ['0.0', '0.005']}
    wrong = d['std'] & ~d['correct'] & ~d['abstain']
    for name, mask in masks.items():
        p = expected['deployment'][name]
        close([(mask & d['correct']).sum(), (mask & d['gain']).sum(), (mask & wrong).sum()],
              [p['loss'], p['gain'], p['wrong_answers_withheld']], name)
    for control in CONTROLS:
        p = common(curves, 'R1b', control, int(.005*n))
        e = expected['comparisons']['R1b-'+control]['exact_common_loss']
        close([p['loss'], p['left_gain'], p['right_gain']],
              [e['loss'], e['left_gain'], e['right_gain']], 'exact/'+control)
    front = {(control, str(cap)): [] for control in CONTROLS for cap in CAPS}
    deployed = {(control, cap, metric): [] for control in CONTROLS[:3]
                for cap in ['0.0', '0.005'] for metric in ['loss', 'gain']}
    rng = np.random.default_rng(20261116)
    for _ in range(args.bootstrap):
        w = rng.multinomial(len(one), np.full(len(one), 1/len(one)))
        p = points(w)
        for control, cap in front:
            front[control, cap].append(p['R1b'][cap]['conversion_rate'] - p[control][cap]['conversion_rate'])
        for control, cap, metric in deployed:
            left, right = masks['R1b/'+cap], masks[control+'/'+cap]
            hit = d['correct'] if metric == 'loss' else d['gain']
            denominator = w @ (d['std'] if metric == 'loss' else d['gain'])
            deployed[control, cap, metric].append((w @ (left & hit) - w @ (right & hit))/denominator)
    if args.bootstrap == 2000:
        for (control, cap), values in front.items():
            close(np.quantile(values, [.025, .975]),
                  expected['comparisons']['R1b-'+control]['frontier'][cap]['ci95'], ('frontier CI', control, cap))
        for (control, cap, metric), values in deployed.items():
            key = 'loss_rate_ci95' if metric == 'loss' else 'rate_ci95'
            close(np.quantile(values, [.025, .975]),
                  expected['comparisons']['R1b-'+control]['deployment/'+cap][key], ('frozen CI', control, cap, metric))
    print('PASS: all VizWiz H frontiers, exact-loss counts and frozen points' +
          (', including all paired 95% intervals.' if args.bootstrap == 2000 else '.'))


if __name__ == '__main__':
    main()
