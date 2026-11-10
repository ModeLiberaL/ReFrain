"""Freeze a selector threshold using T and CM only."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation import load_reference
from selector import select_threshold
from x2_score import native_parser, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--upd-code-root', type=Path, required=True)
    ap.add_argument('--budget', type=float, default=.005)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    parser = native_parser(args.upd_code_root)
    data = {s: load_reference(args.root, 'qwen', s, parser, score) for s in ('T', 'CM')}
    probe = np.load(args.root / 'probe/train_cv.npz')
    mapping = dict(zip(probe['sid'].tolist(), probe['logits'].tolist()))
    scores = {s: np.array([mapping[r['sid']] for r in d['rows']]) for s,d in data.items()}
    tau, lost = select_threshold(scores['T'], scores['CM'], data['CM']['std'],
                                 data['CM']['correct'], args.budget)
    result = dict(budget=args.budget, threshold=tau if np.isfinite(tau) else str(tau),
                  standard_count=int(data['CM']['std'].sum()), lost_correct=lost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
