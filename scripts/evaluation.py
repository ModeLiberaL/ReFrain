"""Shared score calibration and paired evaluation on a fixed backbone."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def load_reference(root, backbone, split, parser, scorer, features=False):
    parts = ['E_part1', 'E_part2'] if backbone == 'ivl' and split == 'E' else [split]
    rows, answers, hidden, logits = [], [], [], []
    for part in parts:
        folder = root / f'{backbone}_base' / part
        rr = json.loads((folder / 'inputs.json').read_text(encoding='utf-8'))['rows']
        assert json.loads((folder / 'status.json').read_text(encoding='utf-8'))['completed'] == len(rr)
        aa = jsonl(folder / 'answers.jsonl')
        assert [r['sid'] for r in rr] == [r['sid'] for r in aa]
        rows.extend(rr)
        answers.extend(aa)
        if features:
            hidden.append(np.load(folder / 'hidden.npy', mmap_mode='r'))
            logits.append(np.load(folder / 'logits.npy', mmap_mode='r'))
    assert len({r['sid'] for r in rows}) == len(rows)
    scored = [scorer(r, a, parser, 'option') for r, a in zip(rows, answers)]
    std = np.array([r['type'] == 'standard' for r in rows])
    correct = np.array([bool(s['correct']) if is_std else bool(s['abstain'])
                        for s, is_std in zip(scored, std)])
    return {'rows': rows, 'std': std, 'correct': correct, 'parsed': scored,
            'hidden': hidden, 'logits': logits}


def circular(rows, correct):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['task'], r['type'], r['family'])].append(i)
    hit = {key: bool(correct[ix].all()) for key, ix in groups.items()}
    tasks = {}
    for task in sorted({r['task'] for r in rows}):
        sf = {key[2] for key in hit if key[:2] == (task, 'standard')}
        uf = {key[2] for key in hit if key[:2] == (task, 'upd')}
        assert sf == uf
        tasks[task] = {'families': len(sf),
            'standard': float(np.mean([hit[(task, 'standard', f)] for f in sf])),
            'upd': float(np.mean([hit[(task, 'upd', f)] for f in sf])),
            'dual': float(np.mean([hit[(task, 'standard', f)] and hit[(task, 'upd', f)] for f in sf]))}
    return {'tasks': tasks, 'macro': {m: float(np.mean([v[m] for v in tasks.values()]))
                                     for m in ['standard', 'upd', 'dual']}}


class Paired:
    def __init__(self, rows):
        self.std = np.array([r['type'] == 'standard' for r in rows])
        groups, self.ix = np.unique([r['cluster'] for r in rows], return_inverse=True)
        self.k = len(groups)
        self.w = np.random.default_rng(0).multinomial(self.k, np.full(self.k, 1 / self.k), size=2000)

    def interval(self, delta, mask):
        n = np.bincount(self.ix, weights=delta * mask, minlength=self.k)
        d = np.bincount(self.ix, weights=mask, minlength=self.k)
        draws = (self.w @ n) / (self.w @ d)
        return [float(n.sum() / d.sum()), *[float(v) for v in np.quantile(draws, [.025, .975])]]
