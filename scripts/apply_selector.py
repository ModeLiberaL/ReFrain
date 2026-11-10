"""Apply a frozen probe and threshold to cached responses without target labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from selector import route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((args.cache / 'inputs.json').read_text(encoding='utf-8'))['rows']
    state = json.loads((args.cache / 'status.json').read_text(encoding='utf-8'))
    if state['completed'] != len(rows):
        raise ValueError('Feature extraction must finish before applying the selector.')
    answers = [json.loads(s) for s in (args.cache / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    logits = np.load(args.cache / 'logits.npy', mmap_mode='r')
    probe = np.load(args.probe)
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    scores = np.empty(len(rows))
    for start in range(0, len(rows), 128):
        scores[start:start+128] = np.asarray(logits[start:start+128], dtype=np.float64) @ probe['coef'] + float(probe['intercept'])
    if not np.isfinite(scores).all():
        raise ValueError('Probe scores must be finite.')
    output = route(rows, answers, scores, float(policy['threshold']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output), encoding='utf-8')


if __name__ == '__main__':
    main()
