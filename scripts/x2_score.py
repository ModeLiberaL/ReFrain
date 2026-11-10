"""X2 frozen scorer (CPU). Deterministic: no LLM judge. Mutually exclusive outcome per row:
  answer (pred label) | abstain | invalid

Abstention is detected in EVERY cell (in forced cells it is a protocol violation, still recorded):
  normalized response starts with "i cannot answer" and the remainder names no option
  (a remainder that parses to an option label or option text -> invalid, "contradictory").
NOTA cells (option setting): choosing the NOTA label = abstain.
Letter formats: official MM-UPD direct parser (can_infer_option) on the displayed label->text map.
Text rendering: normalized exact match to exactly one option text; duplicate normalized texts,
  truncated generations (budget reached) and anything else -> invalid. No prefix matching.
"""
import math
import re
import sys
from pathlib import Path

ABSTAIN_RE = re.compile(r"^\W*i\s+cannot\s+answer\b", re.I)
LABEL_RE = re.compile(r"(?:^|[\s(\[])([A-F])(?:[).:\]]|\s*$)")


def norm(t):
    t = t.replace('\\n', ' ').lower()
    t = re.sub(r'^\s*\(?[a-f][\).:]\s+', '', t)  # a stray leading label
    t = re.sub(r'[^0-9a-z]+', ' ', t)
    return t.strip()


def native_parser(upd_code_root):
    """Load the official MM-UPD direct option parser from its source checkout."""
    from mmupd_adapter import _native
    return _native(str(upd_code_root))


def _names_option(text, display):
    n = norm(text)
    if not n:
        return False
    if any(L in display for L in LABEL_RE.findall(text.strip())):
        return True
    return any(norm(t) == n for t in display.values())


def score(sample, out, native, setting='base'):
    """Return dict(outcome, pred, correct, abstain, invalid, content, reason)."""
    resp = str(out.get('response', ''))
    display = dict(sample['display'])
    gold = sample['gold']
    nota = sample.get('nota')
    res = {'outcome': 'invalid', 'pred': None, 'abstain': False, 'invalid': True, 'correct': 0,
           'content': None, 'reason': None}
    m = ABSTAIN_RE.match(resp)
    if m:
        if _names_option(resp[m.end():], display):
            res['reason'] = 'contradictory_abstention'
            return res
        res.update(outcome='abstain', pred='ABSTAIN', abstain=True, invalid=False, correct=int(gold == 'ABSTAIN'))
        return res
    if sample.get('rendering') == 'text':
        if out.get('budget_reached'):
            res['reason'] = 'truncated'
            return res
        r = norm(resp)
        normed = {L: norm(t) for L, t in display.items()}
        if len(set(normed.values())) != len(normed):
            res['reason'] = 'duplicate_option_text'
            return res
        hits = [L for L, t in normed.items() if t == r]
        if len(hits) != 1:
            res['reason'] = 'no_exact_text_match'
            return res
        pred = hits[0]
    else:
        pred = native['can_infer_option'](resp, display, question_type=setting)
        if not pred or pred not in display:
            res['reason'] = 'unparsed_letter'
            return res
    res.update(pred=pred, invalid=False)
    if nota and pred == nota:
        res.update(outcome='abstain', abstain=True, correct=int(gold == nota))
        return res
    res.update(outcome='answer', correct=int(pred == gold))
    if 'content_id' in sample:
        res['content'] = sample['content_id'][list(display).index(pred)]
    return res


def letter_lp(out, L):
    """First-step log-prob of label L: logsumexp of the bare ("A") and space-prefixed (" A") tokens."""
    lp = out['first_lp']
    a, b = lp[L], lp.get('_' + L, float('-inf'))
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def first_token_pred(sample, out, exclude_nota=False):
    """Argmax of first-step letter log-prob over the displayed labels (optionally excluding NOTA)."""
    labels = [L for L, _ in sample['display'] if not (exclude_nota and L == sample.get('nota'))]
    return max(labels, key=lambda L: letter_lp(out, L))


def option_mass(sample, out):
    """Total first-step probability on the displayed labels (both token forms)."""
    return sum(math.exp(letter_lp(out, L)) for L, _ in sample['display'])
