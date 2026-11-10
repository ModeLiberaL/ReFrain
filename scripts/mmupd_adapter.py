"""MM-UPD prompts, lazy TSV images, and official direct choice extraction."""

from __future__ import annotations

import ast
import base64
import csv
import io
import math
import string
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


TASKS = {"aad": 820, "iasd": 919, "ivqd": 356}
ROWS_PER_SETTING = {"aad": 5537, "iasd": 7052, "ivqd": 2656}


def _present(value):
    return value is not None and str(value).strip().lower() not in {"", "nan", "none"}


def _prompt(row, setting, upd_type):
    text = row["question"]
    if _present(row.get("hint")):
        text = row["hint"] + "\n" + text
    options = {}
    for letter in "ABCDE":
        if not _present(row.get(letter)):
            break
        options[letter] = row[letter]
        text += "\n" + letter + ". " + row[letter]
    if setting == "base":
        text += "\n"
    elif setting == "option":
        text += "\nAnswer with the option's letter from the given choices directly."
    elif setting == "inst":
        if upd_type == "ivqd":
            text += '\nIf the given image is irrelevant to the question, answer "F. The image and question are irrelevant.".'
        else:
            text += '\nIf all the options are incorrect, answer "F. None of the above".'
    else:
        raise ValueError(f"Unknown MM-UPD setting: {setting}")
    return text, options


class _Lines:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        line = self.stream.readline()
        if not line:
            raise StopIteration
        return line.decode("utf-8")


def load_items(root, setting="option"):
    """Read all three combined TSVs; retain byte references instead of image data.

    ``root`` is the MM-UPD dataset directory containing ``data/``. Every official
    precomputed circular variant is returned once. ``inst`` reuses the base TSVs.
    """
    if setting not in {"base", "option", "inst"}:
        raise ValueError(f"Unknown MM-UPD setting: {setting}")
    root = Path(root).resolve()
    items = []
    csv.field_size_limit(64_000_000)
    data_setting = "base" if setting == "inst" else setting
    for upd_type in TASKS:
        path = root / "data" / f"mm{upd_type}_20240303_{data_setting}.tsv"
        with path.open("rb") as stream:
            reader = csv.reader(_Lines(stream), delimiter="\t")
            columns = next(reader)
            image_column = columns.index("image")
            while True:
                offset = stream.tell()
                values = next(reader, None)
                if values is None:
                    break
                length = stream.tell() - offset
                row = dict(zip(columns, values))
                prompt, options = _prompt(row, setting, upd_type)
                index = int(row["index"])
                items.append({
                    "dataset": "MM-UPD",
                    "task": "mmupd_" + upd_type,
                    "sample_id": f"mmupd/{upd_type}/{setting}/{row['type']}/{index}",
                    "setting": setting,
                    "image_ref": {"kind": "mmupd_tsv", "path": str(path),
                                  "offset": offset, "length": length,
                                  "image_column": image_column},
                    "prompt": prompt,
                    "question": row["question"],
                    "gold_label": row["answer"],
                    "masked_answer": row.get("masked_answer", ""),
                    "index": index,
                    "type": row["type"],
                    "options": options,
                    "category": row["category"],
                    "l2-category": row["l2-category"],
                    "split": row["split"],
                })
    group_sizes = Counter((item["task"], item["type"], item["index"] % 1_000_000)
                          for item in items)
    for item in items:
        item["circular_group_size"] = group_sizes[
            (item["task"], item["type"], item["index"] % 1_000_000)]
    return items


def image_bytes(image_ref):
    """Return one referenced image as bytes for ``PIL.Image.open(BytesIO(...))``."""
    with open(image_ref["path"], "rb") as stream:
        stream.seek(image_ref["offset"])
        record = stream.read(image_ref["length"]).decode("utf-8")
    values = next(csv.reader(io.StringIO(record, newline=""), delimiter="\t"))
    return base64.b64decode(values[image_ref["image_column"]])


@lru_cache(maxsize=2)
def _native(code_root):
    """Load the upstream pure functions without importing its API client stack."""
    import pandas as pd

    source_path = Path(code_root) / "automatic_eval" / "calculate_scores.py"
    if not source_path.exists():
        source_path = Path(code_root) / "UPD" / "automatic_eval" / "calculate_scores.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"is_none", "can_infer_option", "build_option_str_w_gt", "build_prompt"}
    selected = [node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"pd": pd, "math": math, "string": string}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace
