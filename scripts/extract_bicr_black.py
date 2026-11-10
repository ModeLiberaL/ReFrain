"""Extract final prompt states for BICR's same-size black-image training view."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from extract_qwen_baselines import image_for_processor

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--upd-code-root', type=Path, required=True)
args = ap.parse_args()
root = args.root.resolve()
out = root / 'bicr'
out.mkdir(exist_ok=True)
binding = json.loads((root / 'qwen_base/T/inputs.json').read_text(encoding='utf-8'))
rows = binding['rows']
model_path = root / 'sft/B1'
assert Path(binding['model']).resolve() == model_path.resolve()
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
    dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
model.requires_grad_(False)
dim = model.get_output_embeddings().weight.shape[1]
hidden = np.lib.format.open_memmap(out / 'black_T.npy', mode='w+', dtype=np.float32,
                                 shape=(len(rows), dim))
started = time.time()
for i, row in enumerate(rows):
    real_image = image_for_processor(Path(binding['image_root']) / row['image'], processor.image_processor)
    black = Image.new('RGB', real_image.size, (0, 0, 0))
    prompt = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[black], return_tensors='pt').to('cuda')
    with torch.inference_mode():
        result = model.model(**inputs, use_cache=False)
    hidden[i] = result.last_hidden_state[0, -1].float().cpu().numpy()
    if (i + 1) % 50 == 0 or i + 1 == len(rows):
        hidden.flush()
        status = dict(state='complete' if i + 1 == len(rows) else 'running',
            completed=i + 1, total=len(rows), seconds=time.time() - started)
        (out / 'black_status.json').write_text(json.dumps(status))
        print(json.dumps(status), flush=True)
    del inputs, result
assert np.isfinite(hidden).all()
(out / 'black_inputs.json').write_text(json.dumps(dict(model=str(model_path),
    sid=[r['sid'] for r in rows], image_view='same-size RGB black',
    representation='final normalized hidden state at last prompt token')))
