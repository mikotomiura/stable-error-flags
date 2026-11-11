"""MCL-WiC En-En for the replication study (rep_preregistration.md): loaders and the WiC layout.

Inputs and references are read by separate functions, as in eval_data. The evaluation set "mcl" is the dev
and test splits pooled; train is used for development only.

- Layout: the character span of the target is turned into WiC's layout (space-joined tokens and the target
  token index): s[:start].split() + [s[start:end]] + s[end:].split(). All target spans of the release are one
  whole word (checked before the freeze), so the target token is exactly the span.
- Parts of speech: only NOUN and VERB items are kept, mapped to "N" and "V" as in WiC (the main evaluation
  and its inventory code cover nouns and verbs only).
- Ids: "<split>.en-en.<n>" as released (unique across splits).
- Units: the lemma, as in the main evaluation (eval_data.wic_unit).
"""
from __future__ import annotations

import json
import zipfile

ALL_ZIP = "SemEval-2021_MCL-WiC_all-datasets.zip"
GOLD_ZIP = "SemEval-2021_MCL-WiC_test-gold-data.zip"
DATA = {"train": "MCL-WiC/training/training.en-en.data", "dev": "MCL-WiC/dev/multilingual/dev.en-en.data",
        "test": "MCL-WiC/test/multilingual/test.en-en.data"}
GOLD = {"train": (ALL_ZIP, "MCL-WiC/training/training.en-en.gold"), "dev": (ALL_ZIP, "MCL-WiC/dev/multilingual/dev.en-en.gold"),
        "test": (GOLD_ZIP, "test.en-en.gold")}
N_RAW = {"train": 8000, "dev": 1000, "test": 1000}
POS_MAP = {"NOUN": "N", "VERB": "V"}
EVAL_SPLITS = ("dev", "test")


def to_wic(it: dict) -> dict:
    """One MCL-WiC item in WiC's layout. Raises if a span is not a whole word."""
    r = {"id": it["id"], "lemma": it["lemma"], "pos": POS_MAP[it["pos"]]}
    for k in (1, 2):
        s, a, b = it[f"sentence{k}"], int(it[f"start{k}"]), int(it[f"end{k}"])
        word = s[a:b]
        assert word and " " not in word and not (a > 0 and s[a - 1].isalnum()) and not (b < len(s) and s[b].isalnum()), it["id"]
        left = s[:a].split()
        r[f"sentence{k}"] = " ".join(left + [word] + s[b:].split())
        r[f"idx{k}"] = len(left)
    return r


def _raw(raw_dir, split):
    items = json.loads(zipfile.ZipFile(f"{raw_dir}/{ALL_ZIP}").read(DATA[split]))
    assert len(items) == N_RAW[split], (split, len(items))
    return items


def load_inputs(raw_dir, splits=EVAL_SPLITS):
    """Inputs (no labels) of the given splits, nouns and verbs only, in WiC's layout."""
    rows = [to_wic(it) for s in splits for it in _raw(raw_dir, s) if it["pos"] in POS_MAP]
    assert len({r["id"] for r in rows}) == len(rows)
    return rows


def load_gold(raw_dir, splits=EVAL_SPLITS):
    """References {id: "T" | "F"} of the given splits (all parts of speech; callers keep the ids they use)."""
    out = {}
    for s in splits:
        z, name = GOLD[s]
        g = json.loads(zipfile.ZipFile(f"{raw_dir}/{z}").read(name))
        assert len(g) == N_RAW[s] and {x["tag"] for x in g} <= {"T", "F"}, s
        out.update({x["id"]: x["tag"] for x in g})
    return out


def unit(item):
    return item["lemma"]
