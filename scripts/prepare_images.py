"""Materialize manifest image paths from the official dataset downloads."""
import argparse
import json
import shutil
from pathlib import Path
from mmupd_adapter import image_bytes, load_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifests', type=Path, default=Path('manifests'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--clipup-images', type=Path)
    ap.add_argument('--mmupd-root', type=Path)
    ap.add_argument('--ai2d-images', type=Path)
    args = ap.parse_args()
    refs = ({r['sample_id']: r for r in load_items(args.mmupd_root, 'option')}
            if args.mmupd_root else {})
    for split in ('T', 'CM', 'E', 'I'):
        if (split == 'T' and not args.clipup_images or
            split in ('CM','E') and not args.mmupd_root or
            split == 'I' and not args.ai2d_images):
            continue
        path = args.manifests / f'{split}.jsonl'
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        for row in rows:
            dst = args.output / row['image']
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if split in ('CM','E'):
                sample = refs[row['sid'].split('/opt/',1)[1]]
                if sample['prompt'] != row['prompt']:
                    raise ValueError(f"Dataset prompt differs for {row['sid']}")
                dst.write_bytes(image_bytes(sample['image_ref']))
            else:
                source = (args.clipup_images / row['image'].removeprefix('clipup/')
                          if split == 'T' else args.ai2d_images / Path(row['image']).name)
                shutil.copyfile(source, dst)
        print(f'{split}: {len(rows)} image references ready')


if __name__ == '__main__':
    main()
