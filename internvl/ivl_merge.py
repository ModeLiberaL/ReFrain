"""Merge a trained LoRA into the InternVL3-8B weights and store one checkpoint that every scheme loads.

--role b1      : base = the pinned InternVL3-8B, adapter = answer SFT            -> the shared B1
--role refusal : base = the merged B1,           adapter = ordinary refusal SFT -> the refusal-SFT model
Merged weight = float32(W) + (alpha / rank) B A, rounded once to the stored dtype (bfloat16). The output directory
holds the new weight files plus an unchanged copy of every other top-level file of the base directory
(configuration, remote code, conversation template, tokenizer), so the InternVL extraction scripts read it with
--model.
Checks, on first-token logits of --check-rows rows of T with their manifest prompts (the evaluation format, both
row types; no generation, nothing from CM or E):
  hard : parameter names, shapes and dtypes are unchanged by the merge and by the reload; the checkpoint reloaded
         from disk gives the logits of the in-memory merged model (--reload-atol).
  reported only: base / adapter-attached / merged logits, argmax and letter agreement, letter hits by row type, and
         how much of the float32 update survives the rounding to bfloat16.
ivl_merge.json is written last; its presence marks a finished merge. An unfinished output directory is refused.
Usage: python internvl/ivl_merge.py --role b1|refusal --base DIR --adapter DIR --output DIR --manifest T.jsonl
       --image-root DIR --clip-up-root DIR [--check-rows 32] [--max-num 6] [--reload-atol 1e-4]
"""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth', '.msgpack', '.h5')
WEIGHT_INDEXES = ('model.safetensors.index.json', 'pytorch_model.bin.index.json')


def is_weight(name):
    return name.endswith(WEIGHT_SUFFIXES) or name in WEIGHT_INDEXES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=('b1', 'refusal'), required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--clip-up-root', type=Path, required=True)
    ap.add_argument('--check-rows', type=int, default=32)
    ap.add_argument('--max-num', type=int, default=6)
    ap.add_argument('--reload-atol', type=float, default=1e-4)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ivl_recipe as R
    import ivl_common as K
    import torch
    U = K.clip_up_utils(args.clip_up_root)

    out = args.output
    if (out / K.MERGE_MARK).exists():
        print('complete', flush=True)
        return
    assert not out.exists() or not any(out.iterdir()), 'unfinished merge in this output directory; use a new --output'
    meta = json.loads((args.adapter / 'adapter.json').read_text())
    assert meta['mode'] == {'b1': 'answer', 'refusal': 'refusal'}[args.role], (args.role, meta['mode'])
    assert meta['trained_on'] == str(args.base.resolve()), 'the adapter was trained on another checkpoint'
    mark = args.base / K.MERGE_MARK
    parent = json.loads(mark.read_text()) if mark.exists() else None
    if args.role == 'b1':
        assert parent is None, 'B1 is merged into the pinned model, not into a merged checkpoint'
    else:
        assert parent is not None and parent['role'] == 'b1', 'the refusal adapter is merged into the merged B1'
    weight_bytes = sum(p.stat().st_size for p in args.base.iterdir() if p.is_file() and is_weight(p.name))
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    assert free > weight_bytes + 2 ** 30, f'free disk {free / 2 ** 30:.1f} GiB, weights {weight_bytes / 2 ** 30:.1f} GiB'
    rows = R.read_rows(args.manifest)  # every row of T, both types, manifest prompts
    assert len(rows) == 1354 and {r['split'] for r in rows} == {'T'}, 'the check reads the T manifest only'
    picks = rows[::max(1, len(rows) // args.check_rows)][:args.check_rows]

    def spec_of(m):
        return {n: (tuple(t.shape), str(t.dtype)) for n, t in m.state_dict().items()}

    def logits_of(m, t):
        return torch.stack([K.first_token_logits(m, t, U, r, args.image_root, args.max_num) for r in picks])

    model, tok, _ = K.load_model(args.base, U)
    lm = model.language_model
    spec = spec_of(model)
    base = logits_of(model, tok)
    assert K.load_adapter(lm, args.adapter) == meta
    assert not any(m.training for m in lm.modules())  # LoRA dropout is off for the comparison
    attached = logits_of(model, tok)
    rounding = K.merge_lora(lm)
    merged = logits_of(model, tok)
    assert spec_of(model) == spec, 'the merge changed parameter names, shapes or dtypes'
    model.save_pretrained(out, safe_serialization=True)
    copied = []
    for p in sorted(args.base.iterdir()):  # the pinned configuration, remote code and tokenizer files, unchanged
        if p.is_file() and not is_weight(p.name) and p.name != K.MERGE_MARK:
            shutil.copyfile(p, out / p.name)
            copied.append(p.name)
    extra = sorted(p.name for p in out.iterdir() if p.is_file() and not is_weight(p.name) and p.name not in copied)
    del model, lm
    gc.collect()
    torch.cuda.empty_cache()

    model, tok, _ = K.load_model(out, U)
    assert spec_of(model) == spec, 'the reloaded checkpoint differs in parameter names, shapes or dtypes'
    reloaded = logits_of(model, tok)
    reload_error = float((reloaded - merged).abs().max())
    assert reload_error <= args.reload_atol, reload_error

    letters = {k: [K.letter_prediction(tok, r, x) for r, x in zip(picks, v)]
               for k, v in (('base', base), ('adapter', attached), ('merged', merged))}

    def hits(pred):
        table = {}
        for kind in ('standard', 'upd'):
            index = [i for i, r in enumerate(picks) if r['type'] == kind]
            table[kind] = [sum(pred[i] == picks[i]['gold'] for i in index), len(index)]
        return table

    check = {'rows': len(picks), 'sids': [r['sid'] for r in picks], 'reload_max_error': reload_error,
             'adapter_vs_base_max_abs_logit': float((attached - base).abs().max()),
             'merged_vs_adapter_max_abs_logit': float((merged - attached).abs().max()),
             'merged_vs_adapter_same_argmax': int((merged.argmax(1) == attached.argmax(1)).sum()),
             'merged_vs_adapter_same_letter': sum(a == b for a, b in zip(letters['merged'], letters['adapter'])),
             'gold_letter_hits_by_type': {k: hits(v) for k, v in letters.items()}}
    R.write_json(out / K.MERGE_MARK, {
        'role': args.role, 'base': str(args.base.resolve()), 'adapter': str(args.adapter.resolve()),
        'adapter_meta': meta, 'parent': parent, 'merge': 'float32(W) + (alpha / rank) B A, rounded once to the stored dtype',
        'rounding': rounding, 'check': check, 'copied_from_base': copied, 'files_not_in_base': extra,
        'environment': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__}})
    print(json.dumps({'event': 'merge_complete', 'role': args.role, 'output': str(out.resolve()), 'rounding': rounding,
                      'check': {k: v for k, v in check.items() if k != 'sids'}}), flush=True)


if __name__ == '__main__':
    main()
