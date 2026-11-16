"""Model-side functions of the InternVL3-8B answer SFT (B1) and ordinary refusal SFT scripts.

Plain functions: the prompt of the InternVL extraction (native conversation template, CLIP-UP tiling; the same prompt
scripts/extract_scenario_internvl.py builds), the training example built on it, a LoRA for the language model, the
training forward, the loss, the scheduler of each recipe, and a merge computed in float32.
The recipes, the training rows and the row order are in ivl_recipe.py.
Nothing is loaded at import time. Import this module before anything from CLIP-UP: it imports transformers' modeling
and generation modules first, the order the recorded runs used.
"""
import json
import math
import re
import sys
from pathlib import Path

import transformers.generation  # noqa: F401  (import order: transformers' modeling modules before scikit-learn,
import transformers.modeling_utils  # noqa: F401  which CLIP-UP's package imports)
import torch
from torch import nn

import ivl_recipe as R

IGNORE = -100
TARGET = re.compile(r'(^|\.)layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$')
MERGE_MARK = 'ivl_merge.json'


def clip_up_utils(root):
    """CLIP-UP's InternVL image loader and image-token constants (the import the InternVL extraction uses)."""
    sys.path.insert(0, str(root))
    from clip_up.vlms.internvl import clip_up_internvl_utils
    return clip_up_internvl_utils


def load_model(path, U):
    """The loading calls of the InternVL extraction. Everything frozen, eval mode. Returns model, tokenizer, end-token id."""
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=False, trust_remote_code=True, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, local_files_only=True)
    model.img_context_token_id = tok.convert_tokens_to_ids(U.IMG_CONTEXT_TOKEN)
    end = model.conv_template.sep.strip()
    eos_id = tok.convert_tokens_to_ids(end)
    assert isinstance(eos_id, int) and tok.convert_ids_to_tokens(eos_id) == end, (end, eos_id)
    return model, tok, eos_id


def load_pixels(U, image_root, row, max_num):
    from PIL import Image
    image = Image.open(Path(image_root) / row['image']).convert('RGB')
    return U.load_image_internvl(image, max_num=max_num).to(torch.bfloat16)


def inference_prompt(model, U, row, tiles):
    """The prompt string the InternVL extraction builds for a row whose image gives `tiles` tiles."""
    template = model.conv_template.copy()
    template.append_message(template.roles[0], '<image>\n' + row['prompt'])
    template.append_message(template.roles[1], None)
    return template.get_prompt().replace(
        '<image>', U.IMG_START_TOKEN + U.IMG_CONTEXT_TOKEN * model.num_image_token * tiles + U.IMG_END_TOKEN, 1)


def encode(model, tok, U, eos_id, row, image_root, max_num):
    """Training example: inference prompt ids, then the gold letter and the end token. Only the appended ids are supervised."""
    pixel_values = load_pixels(U, image_root, row, max_num)
    prompt_ids = tok(inference_prompt(model, U, row, pixel_values.shape[0]), return_tensors='pt')['input_ids'][0]
    target = tok(row['gold'], add_special_tokens=False)['input_ids'] + [eos_id]
    input_ids = torch.cat([prompt_ids, torch.tensor(target, dtype=prompt_ids.dtype)])
    assert len(input_ids) <= tok.model_max_length, (row['sid'], len(input_ids))
    labels = torch.full_like(input_ids, IGNORE)
    labels[len(prompt_ids):] = input_ids[len(prompt_ids):]
    return {'pixel_values': pixel_values, 'input_ids': input_ids, 'labels': labels, 'prompt_len': len(prompt_ids)}


def forward_logits(model, input_ids, pixel_values):
    """Language-model logits [positions, vocabulary] for one sequence. The vision tower and projector run without
    gradient; their output replaces the image-context embeddings, as in the model's own forward."""
    lm = model.language_model
    ids = input_ids.unsqueeze(0).cuda()
    with torch.no_grad():
        vit = model.extract_feature(pixel_values.cuda())
        emb = lm.get_input_embeddings()(ids)
        sel = ids[0] == model.img_context_token_id
        assert int(sel.sum()) == vit.shape[0] * vit.shape[1], (int(sel.sum()), tuple(vit.shape))
        emb[0, sel] = vit.reshape(-1, vit.shape[-1]).to(emb.dtype)
    return lm(inputs_embeds=emb, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]


def sft_loss(logits, labels):
    """Mean next-token cross-entropy over the supervised positions, in float32. Position p is predicted from p - 1."""
    pos = (labels != IGNORE).nonzero(as_tuple=True)[0].to(logits.device)
    assert len(pos) and int(pos[0]) > 0
    return nn.functional.cross_entropy(logits[pos - 1].float(), labels.to(logits.device)[pos])


def first_token_logits(model, tok, U, row, image_root, max_num):
    """Raw logits at the last prompt position (the first-token logits the extraction stores), float32 on the CPU."""
    pixel_values = load_pixels(U, image_root, row, max_num)
    prompt_ids = tok(inference_prompt(model, U, row, pixel_values.shape[0]), return_tensors='pt')['input_ids'][0]
    with torch.inference_mode():
        return forward_logits(model, prompt_ids, pixel_values)[-1].float().cpu()


def letter_prediction(tok, row, logits):
    """Displayed option letter with the largest first-token logit."""
    ids = {}
    for letter, _ in row['display']:
        piece = tok(letter, add_special_tokens=False)['input_ids']
        assert len(piece) == 1, (letter, piece)
        ids[letter] = piece[0]
    return max(ids, key=lambda letter: float(logits[ids[letter]]))


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / rank) B A dropout(x). The base layer is frozen. A and B use the usual LoRA layout and
    initialisation (A: Kaiming uniform as nn.Linear, drawn in float32; B: zeros), so the update starts at zero.
    They are stored, and updated by the optimizer, in `dtype`: float32 for answer and bfloat16 for refusal, as in the
    two Qwen runs. The LoRA branch is computed in the dtype of the layer input (bfloat16), which is what the
    bfloat16 autocast of the Qwen scripts did for both."""

    def __init__(self, base, rank, alpha, dropout, dtype):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        self.base, self.scale = base, alpha / rank
        self.drop = nn.Dropout(dropout)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a.to(base.weight.device, dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        z = nn.functional.linear(nn.functional.linear(self.drop(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return y + z * self.scale


def lora_modules(lm):
    return [(n, m) for n, m in lm.named_modules() if isinstance(m, LoRALinear)]


def add_lora(lm, rank, alpha, dropout, dtype):
    """Wrap q/k/v/o and gate/up/down of every language layer. Returns the wrapped module names."""
    names = [n for n, _ in lm.named_modules() if TARGET.search(n)]
    assert len(names) == 7 * lm.config.num_hidden_layers, (len(names), lm.config.num_hidden_layers)
    for name in names:
        parent, _, child = name.rpartition('.')
        parent = lm.get_submodule(parent)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha, dropout, dtype).train(parent.training))
    return names


def set_trainable(model):
    """Freeze everything, then enable the LoRA matrices of the language model only. Returns (parameters, audit)."""
    model.requires_grad_(False)
    params, expected = [], 0
    for _, mod in lora_modules(model.language_model):
        params += [mod.lora_A.requires_grad_(True), mod.lora_B.requires_grad_(True)]
        expected += mod.lora_A.shape[0] * (mod.base.in_features + mod.base.out_features)
    named = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    assert named and all(n.startswith('language_model.') and n.rsplit('.', 1)[1] in ('lora_A', 'lora_B') for n, _ in named)
    count = sum(k for _, k in named)
    assert count == expected == sum(p.numel() for p in params), (count, expected)
    outside = {k: sum(c for n, c in named if n.startswith(k + '.')) for k in ('vision_model', 'mlp1')}
    assert not any(outside.values()), outside
    return params, {'trainable_parameters': count, 'trainable_tensors': len(named),
                    'lora_modules': len(params) // 2, 'trainable_in_vision_model': outside['vision_model'],
                    'trainable_in_mlp1': outside['mlp1'], 'all_parameters': sum(p.numel() for p in model.parameters()),
                    'adapter_dtype': str(params[0].dtype), 'base_dtype': str(model.language_model.get_input_embeddings().weight.dtype)}


def prepare_training(model, mode, grad_checkpoint):
    """LoRA of the mode's recipe on the language model, trainable set, gradient checkpointing. Seed before calling."""
    recipe = R.RECIPE[mode]
    add_lora(model.language_model, recipe['rank'], recipe['alpha'], recipe['dropout'], getattr(torch, recipe['adapter_dtype']))
    params, audit = set_trainable(model)
    assert audit['adapter_dtype'] == 'torch.' + recipe['adapter_dtype'], audit['adapter_dtype']
    if grad_checkpoint:
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    audit['gradient_checkpointing'] = bool(grad_checkpoint)
    return params, audit


def make_scheduler(optimizer, mode, steps, warmup):
    """get_linear_schedule_with_warmup for answer, get_cosine_schedule_with_warmup for refusal, as the Qwen scripts."""
    import transformers
    make = {'linear': transformers.get_linear_schedule_with_warmup, 'cosine': transformers.get_cosine_schedule_with_warmup}
    return make[R.RECIPE[mode]['schedule']](optimizer, warmup, steps)


def lr_sequence(mode, steps, warmup):
    """Learning rate in force at every optimizer step, read before the step as both Qwen scripts log it. The
    scheduler runs on a one-element parameter; no model is involved."""
    probe = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=R.RECIPE[mode]['lr'])
    scheduler, sequence = make_scheduler(probe, mode, steps, warmup), []
    for _ in range(steps):
        sequence.append(probe.param_groups[0]['lr'])
        probe.step()
        scheduler.step()
    return sequence


def lr_parity(mode, sequence, recorded):
    """Largest difference between this scheduler's learning rates and the lr column of the Qwen run's log."""
    assert len(sequence) == len(recorded), (len(sequence), len(recorded))
    gap = max(abs(a - b) for a, b in zip(sequence, recorded))
    assert gap <= 1e-9 * R.RECIPE[mode]['lr'], gap
    return gap


def save_adapter(lm, path, meta):
    """A and B as float32 (exact for bfloat16 values). meta records the dtype they were trained in."""
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, mod in lora_modules(lm):
        tensors[name + '.lora_A'] = mod.lora_A.detach().float().cpu().contiguous()
        tensors[name + '.lora_B'] = mod.lora_B.detach().float().cpu().contiguous()
    save_file(tensors, str(path / 'adapter.safetensors'))
    R.write_json(path / 'adapter.json', dict(meta, modules=len(tensors) // 2))


def load_adapter(lm, path):
    """Attach the saved LoRA, as float32 copies, to an unwrapped language model. Returns the adapter's metadata."""
    from safetensors.torch import load_file
    meta = json.loads((path / 'adapter.json').read_text())
    names = add_lora(lm, meta['rank'], meta['alpha'], meta['dropout'], torch.float32)
    tensors = load_file(str(path / 'adapter.safetensors'))
    assert set(tensors) == {n + s for n in names for s in ('.lora_A', '.lora_B')}
    for name, mod in lora_modules(lm):
        for key in ('lora_A', 'lora_B'):
            t = tensors[f'{name}.{key}']
            assert t.shape == getattr(mod, key).shape and t.dtype == torch.float32, (name, key)
            getattr(mod, key).data.copy_(t)
    return meta


def merge_lora(lm):
    """W <- (float32(W) + (alpha / rank) B A) rounded once to W's dtype; then the wrappers are removed.
    Returns how much of the float32 update survives the rounding to the stored dtype."""
    s = {'modules': 0, 'elements': 0, 'changed_elements': 0, 'update_sq': 0.0, 'realised_sq': 0.0, 'error_sq': 0.0,
         'max_abs_update': 0.0}
    wrapped = lora_modules(lm)
    assert wrapped, 'no LoRA modules to merge'
    with torch.no_grad():
        for name, mod in wrapped:
            w = mod.base.weight
            update = (mod.lora_B.float() @ mod.lora_A.float()) * mod.scale
            old = w.float()
            new = (old + update).to(w.dtype)
            realised = new.float() - old
            s['modules'] += 1
            s['elements'] += w.numel()
            s['changed_elements'] += int((realised != 0).sum())
            s['update_sq'] += float(update.double().pow(2).sum())
            s['realised_sq'] += float(realised.double().pow(2).sum())
            s['error_sq'] += float((realised - update).double().pow(2).sum())
            s['max_abs_update'] = max(s['max_abs_update'], float(update.abs().max()))
            w.copy_(new)
            parent, _, child = name.rpartition('.')
            setattr(lm.get_submodule(parent), child, mod.base)
    assert not lora_modules(lm)
    norm = math.sqrt(s['update_sq'])
    return {'modules': s['modules'], 'weight_dtype': str(w.dtype), 'elements': s['elements'],
            'changed_elements': s['changed_elements'], 'changed_fraction': s['changed_elements'] / s['elements'],
            'update_norm': norm, 'realised_norm': math.sqrt(s['realised_sq']),
            'relative_rounding_error': math.sqrt(s['error_sq']) / norm if norm else None,
            'max_abs_update': s['max_abs_update']}
