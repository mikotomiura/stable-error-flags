"""Evaluation data: loaders, input forms, and independent units (preregistration v4, Secs. 2 and 4).

Inputs and references are read by separate functions. Step 1 of the post-freeze order (Sec. 0) reads only
the inputs; the references are read in Step 4 (eval_analysis.count).

Forms (Sec. 4):
- WiC, set S: orig, swap (sentence 1 and 2 exchanged), para1 (both sentences, paraphrase 1).
  Set B (b2, secondary): para2 (both sentences, paraphrase 2) and para2_swap.
- NLI, set S: orig, prem_para1 (premise only), hyp_para1 (hypothesis only). Set B: both_para2.
"""
from __future__ import annotations

import json
import re
import zipfile

from common import find_word

WIC_SPLITS = {"wic_dev": "dev/dev", "wic_test": "test/test"}
WIC_N = {"wic_dev": 638, "wic_test": 1400}
CHAOS_FILES = {"chaos_s": "chaosNLI_snli.jsonl", "chaos_m": "chaosNLI_mnli_m.jsonl"}
CHAOS_N = {"chaos_s": 1514, "chaos_m": 1599}
CHAOS_LABEL = {"e": "entailment", "n": "neutral", "c": "contradiction"}
WIC_S = ("orig", "swap", "para1")
WIC_B = ("para2", "para2_swap")
NLI_S = ("orig", "prem_para1", "hyp_para1")
NLI_B = ("both_para2",)


# ---------- WiC ----------

def _zip_lines(z, name):
    return z.read(name).decode("utf-8").rstrip("\n").split("\n")


def load_wic_inputs(zip_path, data):
    """Inputs of WiC dev or test (no labels). Same layout as train: lemma, pos, "i-j", sentence 1, sentence 2."""
    z = zipfile.ZipFile(zip_path)
    lines = _zip_lines(z, WIC_SPLITS[data] + ".data.txt")
    assert len(lines) == WIC_N[data], (data, len(lines))
    rows = []
    for i, line in enumerate(lines):
        lemma, pos, span, s1, s2 = line.split("\t")
        a, b = span.split("-")
        rows.append({"id": f"{data}_{i}", "row": i, "lemma": lemma, "pos": pos,
                     "idx1": int(a), "idx2": int(b), "sentence1": s1, "sentence2": s2})
    assert len({r["id"] for r in rows}) == len(rows)
    return rows


def load_wic_gold(zip_path, data):
    """References of WiC dev or test: {id: "T" | "F"}. Read only in Step 4 of Sec. 0."""
    z = zipfile.ZipFile(zip_path)
    gold = [g.strip() for g in _zip_lines(z, WIC_SPLITS[data] + ".gold.txt")]
    assert len(gold) == WIC_N[data] and set(gold) <= {"T", "F"}, (data, len(gold), set(gold))
    return {f"{data}_{i}": g for i, g in enumerate(gold)}


def wic_word(item, k):
    return item[f"sentence{k}"].split(" ")[item[f"idx{k}"]]


def retokenize(text, word):
    """Make the unique occurrence of `word` its own space-separated token. Returns (sentence, idx) or None."""
    span = find_word(text, word)
    if span is None:
        return None
    before, mid, after = text[: span[0]].split(), text[span[0]: span[1]], text[span[1]:].split()
    return " ".join(before + [mid] + after), len(before)


def wic_forms(item, para1=None, para2=None):
    """Build the WiC forms of one item. para1 / para2: (text of sentence 1, text of sentence 2), or None if the
    paraphrase could not be produced. A form that cannot be built is absent from the result."""
    base = {k: item[k] for k in ("id", "lemma", "pos")}
    out = {"orig": dict(base, sentence1=item["sentence1"], sentence2=item["sentence2"], idx1=item["idx1"], idx2=item["idx2"]),
           "swap": dict(base, sentence1=item["sentence2"], sentence2=item["sentence1"], idx1=item["idx2"], idx2=item["idx1"])}
    for name, para in (("para1", para1), ("para2", para2)):
        if para is None:
            continue
        r1, r2 = retokenize(para[0], wic_word(item, 1)), retokenize(para[1], wic_word(item, 2))
        if r1 is None or r2 is None:
            continue
        out[name] = dict(base, sentence1=r1[0], sentence2=r2[0], idx1=r1[1], idx2=r2[1])
        if name == "para2":
            out["para2_swap"] = dict(base, sentence1=r2[0], sentence2=r1[0], idx1=r2[1], idx2=r1[1])
    return out


# ---------- GrainWiC coarse (secondary) ----------

GRAIN_N = 2798
TRUE_LABELS, FALSE_LABELS = {"T", "True", "true", "TRUE", "1"}, {"F", "False", "false", "FALSE", "0"}


def parse_target_index(v):
    """target_index of GrainWiC. Its layout was not inspected before freeze 2 (only the column types), so the
    accepted layouts are fixed here and anything else stops the run: "i-j", "i,j", "i j", "[i, j]"."""
    m = re.fullmatch(r"\s*\[?\s*(\d+)\s*[-, ]\s*(\d+)\s*\]?\s*", str(v))
    if m is None:
        raise ValueError(f"unexpected target_index layout: {v!r}")
    return int(m.group(1)), int(m.group(2))


def _grain_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    assert len(rows) == GRAIN_N, len(rows)
    return rows


def load_grain_inputs(path):
    """Inputs of GrainWiC coarse (no labels). All 20 target words are nouns ([20] Table 2). The sentences are
    assumed to be space-tokenized as in WiC; the token at each index must contain the target word for at
    least 95% of the sentences, otherwise the run stops (a machine check that needs no reading of the data)."""
    out, hit = [], 0
    for r in _grain_rows(path):
        i1, i2 = parse_target_index(r["target_index"])
        t1, t2 = r["sentence1"].split(" "), r["sentence2"].split(" ")
        assert i1 < len(t1) and i2 < len(t2), r["id"]
        w = r["word"].lower()
        hit += (w[:4] in t1[i1].lower()) + (w[:4] in t2[i2].lower())
        out.append({"id": f"grain_{r['id']}", "row": r["id"], "lemma": r["word"], "pos": "N",
                    "idx1": i1, "idx2": i2, "sentence1": r["sentence1"], "sentence2": r["sentence2"]})
    assert hit >= 0.95 * 2 * len(out), f"target_index does not point at the target word ({hit} of {2 * len(out)})"
    assert len({r["id"] for r in out}) == len(out)
    return out


def load_grain_gold(path):
    out = {}
    for r in _grain_rows(path):
        v = str(r["label"]).strip()
        assert v in TRUE_LABELS | FALSE_LABELS, f"unexpected label {v!r}"
        out[f"grain_{r['id']}"] = "T" if v in TRUE_LABELS else "F"
    return out


# ---------- ChaosNLI ----------

def chaos_bytes(path, data):
    """Raw bytes of one ChaosNLI part, from the release zip or from a directory holding the two jsonl files
    (the preservation mirror used here, Sec. 13 #3). The digests in config.json are over these bytes."""
    from pathlib import Path
    if Path(path).is_dir():
        return (Path(path) / CHAOS_FILES[data]).read_bytes()
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if n.endswith("/" + CHAOS_FILES[data]) or n == CHAOS_FILES[data]]
    assert len(names) == 1, (data, names)
    return z.read(names[0])


def _chaos_rows(path, data):
    text = chaos_bytes(path, data).decode("utf-8")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(rows) == CHAOS_N[data], (data, len(rows))
    return rows


def load_chaos_inputs(zip_path):
    """Inputs of ChaosNLI-S and -M as one data set (no labels). id = "<part>_<uid>"."""
    rows = []
    for data in CHAOS_FILES:
        for r in _chaos_rows(zip_path, data):
            ex = r["example"]
            rows.append({"id": f"{data}_{r['uid']}", "part": data, "premise": ex["premise"], "hypothesis": ex["hypothesis"]})
    assert len({r["id"] for r in rows}) == len(rows) == sum(CHAOS_N.values())
    return rows


def chaos_reference(label_count):
    """Majority of the 100 judgements in the order [E, N, C]; None when the top count is tied (Sec. 2)."""
    top = max(label_count)
    winners = [j for j, c in enumerate(label_count) if c == top]
    return ["entailment", "neutral", "contradiction"][winners[0]] if len(winners) == 1 else None


def load_chaos_labels(zip_path):
    """References and agreement of ChaosNLI: {id: {"ref", "max_frac", "entropy", "label_count"}}. Step 4 only."""
    out = {}
    for data in CHAOS_FILES:
        for r in _chaos_rows(zip_path, data):
            lc = list(r["label_count"])
            assert len(lc) == 3 and sum(lc) > 0
            # The order [E, N, C] is checked against the named counts and the distributed majority label.
            assert lc == [r["label_counter"].get(k, 0) for k in ("e", "n", "c")], r["uid"]
            ref = chaos_reference(lc)
            assert ref is None or CHAOS_LABEL[r["majority_label"]] == ref, r["uid"]
            out[f"{data}_{r['uid']}"] = {"ref": chaos_reference(lc), "max_frac": max(lc) / sum(lc),
                                         "entropy": r["entropy"], "label_count": lc}
    return out


def nli_forms(item, prem_para1=None, hyp_para1=None, prem_para2=None, hyp_para2=None):
    base = {"id": item["id"]}
    out = {"orig": dict(base, premise=item["premise"], hypothesis=item["hypothesis"])}
    if prem_para1 is not None:
        out["prem_para1"] = dict(base, premise=prem_para1, hypothesis=item["hypothesis"])
    if hyp_para1 is not None:
        out["hyp_para1"] = dict(base, premise=item["premise"], hypothesis=hyp_para1)
    if prem_para2 is not None and hyp_para2 is not None:
        out["both_para2"] = dict(base, premise=prem_para2, hypothesis=hyp_para2)
    return out


# ---------- Independent units (Sec. 2) ----------

def wic_unit(item):
    return item["lemma"]


def nli_unit(item):
    """Premises with the same string (leading and trailing spaces removed, case-sensitive) form one unit."""
    return item["premise"].strip()
