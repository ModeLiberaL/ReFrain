"""Assemble scenario NPZ inputs from extracted blocks and optional A/B records."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--final', default='L36.npy')
    parser.add_argument('--question', default='Q27.npy')
    parser.add_argument('--source-score', default='s0.npy')
    parser.add_argument('--image', help='Optional same-layer image block for the R1v control.')
    parser.add_argument('--records', type=Path, help='A/B JSON list: sid, group, answerable, reference_correct, optional fold.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.features / 'inputs.json').read_text())
    rows = metadata['rows']
    status = json.loads((args.features / 'status.json').read_text())
    if status['state'] != 'complete' or status['completed'] != len(rows):
        raise ValueError('Feature extraction must finish before packing.')
    lookup = {record['sid']: i for i, record in enumerate(rows)}
    records = json.loads(args.records.read_text()) if args.records else rows
    ix = np.array([lookup[record['sid']] for record in records])
    result = dict(sid=np.array([record['sid'] for record in records]),
        source_score=np.asarray(np.load(args.features / args.source_score, mmap_mode='r')[ix], dtype=np.float64),
        final_state=np.asarray(np.load(args.features / args.final, mmap_mode='r')[ix], dtype=np.float32),
        question_state=np.asarray(np.load(args.features / args.question, mmap_mode='r')[ix], dtype=np.float32))
    if args.image:
        result['image_state'] = np.asarray(np.load(args.features / args.image, mmap_mode='r')[ix], dtype=np.float32)
    if args.records:
        result.update(groups=np.array([str(record['group']) for record in records]),
                      std=np.array([record['answerable'] for record in records], dtype=bool),
                      correct=np.array([record['reference_correct'] for record in records], dtype=bool))
        if 'fold' in records[0]:
            result['folds'] = np.array([record['fold'] for record in records], dtype=int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)


if __name__ == '__main__':
    main()
