"""Full paraphrase check of the replication study (rep_preregistration.md, Sec. 4; Sec. 0, Steps 2 and 3).

  python rep_check.py judge-sheet --out O                              -> O/check/hardneg_judgment.csv (blank `valid` and `meaning_changed`)
  python rep_check.py build  --config ../rep_config.json --out O     -> O/check/sheet_rater{1,2}.csv, key.json
  python rep_check.py collect --out O --rater rater1 --file RETURNED.csv  -> verified kept into O/check/sheet_rater1.csv
  python rep_check.py score  --config ../rep_config.json --out O

Hard negatives and meaning controls (Sec. 4; decisions #12, #15, #18): before any sheet exists, a judge who is
not a rater fills two columns for every parsed candidate, in the seeded candidate order: `valid` (1 = the marked
word's sense really changed) and `meaning_changed` (1 = the rewritten sentence clearly says something different).
build records the SHA-256 of the filled judgment, takes the first n_hardneg candidates with valid = 1 (hard
negatives: the sense changed) and the first n_meaning with valid = 0 and meaning_changed = 1 (meaning controls:
the sense did not change but the sentence meaning did); nothing is dropped after the rating.

Every evaluation item whose paraphrase 1 could be formed contributes two rows (sentence 1 and sentence 2 with
their paraphrases). The controls are mixed in: n_other sentences of another item with a different lemma, and the
valid hard negatives, and the meaning controls. The rows are shuffled with a fixed seed; both raters get the same sheet, which does not
show which rows are controls. Each rater fills `kept` with 1 (meaning kept) or 0, alone and without any
automatic or LLM help.

Scoring: an item is excluded (counted as not transformable: kept in the flag ranking and in the precision
denominator, removed from the stable and unstable errors) if either rater marks either of its two sentences 0.
The attention check fails if a rater keeps more than max_kept_other (2, i.e. fails at 3 of 20; decisions #21) of the
other-item controls, or if the two
raters BOTH keep at least ceil(hardneg_fail_share x n) of the n hard negatives, or at least
ceil(meaning_fail_share x m) of the m meaning controls (the pair is the unit, as in the exclusion rule: a sentence
is excluded when either rater marks it 0; decisions #19). A part is skipped, and reported, if it has fewer than
min_valid_hardneg / min_meaning rows. Each rater's counts are reported; a rater who alone keeps at least the same
share of a used part is listed in pair.lenient_raters (reported, the decision is unchanged; decisions #22). Each
rater's kept counts on the first and second half of the sheet (by row number) are reported, descriptive only.

collect (Sec. 4): a returned sheet (csv, UTF-8) is accepted only if it has exactly the built rows and its row, word,
original and rewritten columns equal key.json on every row, and every kept is 0 or 1; its SHA-256 is recorded in
check/collected.json and its kept column is written into check/sheet_<rater>.csv (the blank sheet is rebuilt from key.json). A failed attention check makes H1, H2 and
H1b exploratory, and so does a rater who does
not declare in check/declarations.json that the sheet was rated alone without automatic or LLM help
({"rater1": true, "rater2": true}; a false or missing entry counts as not declared).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import eval_data as D
from check_sheet import read_sheet
from common import item_seed, read_jsonl
from eval_run import load_config
from rep_run import DATA, forms_of

COLS = ["row", "word", "original", "rewritten", "kept"]
RATERS = ("rater1", "rater2")
JUDGE_COLS = ["cand", "id", "word", "original", "rewritten", "valid", "meaning_changed"]


def _rng(what):
    return np.random.default_rng(item_seed(DATA, "check", what))


def pairs(cfg, out):
    t = Path(out) / "transform"
    forms = forms_of(t)
    items = {it["id"]: it for it in read_jsonl(t / f"{DATA}_inputs.jsonl")}
    row = lambda kind, i, k, new: {"kind": kind, "id": i, "k": k, "word": D.wic_word(items[i], k),
                                   "original": items[i][f"sentence{k}"], "rewritten": new}
    S = sorted(i for i, fs in forms.items() if "para1" in fs)
    out_rows = [row("S_para1", i, k, forms[i]["para1"][f"sentence{k}"]) for i in S for k in (1, 2)]
    c = cfg["check"]
    ids = sorted(items)
    for j in _rng("other").choice(len(ids), size=c["n_other"], replace=False):
        i = ids[int(j)]
        others = sorted(x for x in ids if items[x]["lemma"] != items[i]["lemma"])
        o = others[int(_rng("other_" + i).integers(len(others)))]
        out_rows.append(row("C_other", i, 1, items[o]["sentence1"]))
    out_rows += [row("C_hardneg", r["id"], 1, r["text"]) for r in valid_hardnegs(out)[: c["n_hardneg"]]]
    out_rows += [row("C_meaning", r["id"], 1, r["text"]) for r in meaning_controls(out)[: c["n_meaning"]]]
    return out_rows


def judge_sheet(out):
    """Write the hard-negative candidates (parsed only, seeded order) for the judge, before any rating sheet."""
    t, d = Path(out) / "transform", Path(out) / "check"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "hardneg_judgment.csv"
    assert not f.exists(), f"{f} exists; the judgment sheet is written once"
    assert not (d / "key.json").exists(), "the rating sheets already exist; the judgment must come first"
    items = {it["id"]: it for it in read_jsonl(t / f"{DATA}_inputs.jsonl")}
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=JUDGE_COLS)
        w.writeheader()
        for n, r in enumerate(r for r in read_jsonl(t / "wic_hardneg.jsonl") if r["parsed"]):
            it = items[r["id"]]
            w.writerow({"cand": n, "id": r["id"], "word": D.wic_word(it, 1), "original": it["sentence1"], "rewritten": r["text"],
                        "valid": "", "meaning_changed": ""})
    print("JUDGE_SHEET", f, flush=True)


def _judged(out):
    with open(Path(out) / "check" / "hardneg_judgment.csv", encoding="utf-8") as fh:
        judged = list(csv.DictReader(fh))
    assert all(r["valid"] in ("0", "1") and r["meaning_changed"] in ("0", "1") for r in judged), \
        "fill `valid` and `meaning_changed` with 0 or 1 on every candidate first"
    cand = {r["id"]: r for r in read_jsonl(Path(out) / "transform" / "wic_hardneg.jsonl") if r["parsed"]}
    return judged, cand


def valid_hardnegs(out):
    """Hard negatives (valid = 1) in the candidate order, from the filled judgment."""
    judged, cand = _judged(out)
    return [cand[r["id"]] for r in judged if r["valid"] == "1"]


def meaning_controls(out):
    """Meaning controls (valid = 0, meaning_changed = 1) in the candidate order, from the filled judgment."""
    judged, cand = _judged(out)
    return [cand[r["id"]] for r in judged if r["valid"] == "0" and r["meaning_changed"] == "1"]


def build(cfg, out):
    d = Path(out) / "check"
    d.mkdir(parents=True, exist_ok=True)
    files = [d / f"sheet_{r}.csv" for r in RATERS] + [d / "key.json"]
    for f in files:
        assert not f.exists(), f"{f} exists; the sheet is built once and never overwritten"
    jf = d / "hardneg_judgment.csv"
    assert jf.exists(), "judge the hard-negative candidates first (judge-sheet, then fill `valid`)"
    n_valid, n_meaning = len(valid_hardnegs(out)), len(meaning_controls(out))
    ps = pairs(cfg, out)
    order = _rng("shuffle").permutation(len(ps))
    key = {}
    rows = []
    for r, j in enumerate(order, 1):
        p = dict(ps[int(j)], row=r, kept="")
        key[r] = p
        rows.append(p)
    for rater in RATERS:
        with open(d / f"sheet_{rater}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    (d / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = {"rows": len(ps), "items": len({p["id"] for p in ps if p["kind"] == "S_para1"}),
           "controls_other": sum(p["kind"] == "C_other" for p in ps), "controls_hardneg": sum(p["kind"] == "C_hardneg" for p in ps),
           "controls_meaning": sum(p["kind"] == "C_meaning" for p in ps), "meaning_available": n_meaning,
           "hardneg_candidates_judged": len(jf.read_text(encoding="utf-8").splitlines()) - 1, "hardneg_valid": n_valid,
           "hardneg_judgment_sha256": hashlib.sha256(jf.read_bytes()).hexdigest(),
           "control_originals_also_on_S_rows": len({p["original"] for p in ps if p["kind"].startswith("C_")} & {p["original"] for p in ps if p["kind"] == "S_para1"})}
    (d / "build.json").write_text(json.dumps(rep, indent=1))
    print("SHEET", json.dumps(rep), flush=True)


def collect(out, rater, src):
    """Accept a returned sheet: same rows and texts as key.json, kept in {0, 1}; write it into check/sheet_<rater>.csv."""
    assert rater in RATERS
    d = Path(out) / "check"
    key = {int(r): p for r, p in json.loads((d / "key.json").read_text(encoding="utf-8")).items()}
    raw = Path(src).read_bytes()
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    got = {int(r["row"]): r for r in rows}
    assert len(got) == len(rows) and set(got) == set(key), f"{src}: rows differ from the built sheet"
    for n, p in key.items():
        for c in ("word", "original", "rewritten"):
            assert got[n][c] == p[c], f"{src}: row {n} column {c} differs from the built sheet"
        assert got[n]["kept"].strip() in ("0", "1"), f"{src}: row {n} kept must be 0 or 1"
    with open(d / f"sheet_{rater}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(dict(key[n], kept=got[n]["kept"].strip()) for n in sorted(key))
    log = d / "collected.json"
    rec = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    rec[rater] = {"file": Path(src).name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    log.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec[rater]


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else None


def score(key, kept_by_rater, check, declarations=None):
    """key: {row: pair}; kept_by_rater: {rater: {row: 0/1}}; check: cfg["check"]; declarations: {rater: bool}."""
    declarations = declarations or {}
    S = [r for r, p in key.items() if p["kind"] == "S_para1"]
    CO = [r for r, p in key.items() if p["kind"] == "C_other"]
    CH = [r for r, p in key.items() if p["kind"] == "C_hardneg"]
    CM = [r for r, p in key.items() if p["kind"] == "C_meaning"]
    use_hn = len(CH) >= check["min_valid_hardneg"]
    use_m = len(CM) >= check["min_meaning"]
    hn_limit = math.ceil(check["hardneg_fail_share"] * len(CH))
    m_limit = math.ceil(check["meaning_fail_share"] * len(CM))
    res = {"raters": {}, "controls_other": len(CO), "controls_hardneg": len(CH), "controls_meaning": len(CM),
           "hardneg_part_used": use_hn, "hardneg_fail_at": hn_limit if use_hn else None,
           "meaning_part_used": use_m, "meaning_fail_at": m_limit if use_m else None, "other_fail_above": check["max_kept_other"]}
    excluded = set()
    for rater, kept in kept_by_rater.items():
        assert set(kept) == set(key), f"{rater}: every row must be rated"
        bad_items = {key[r]["id"] for r in S if kept[r] == 0}
        excluded |= bad_items
        ko, kh, km = (sum(kept[r] for r in rows) for rows in (CO, CH, CM))
        res["raters"][rater] = {"S_rows_kept": sum(kept[r] for r in S), "S_rows": len(S), "items_marked_changed": len(bad_items),
                                "other_kept": ko, "hardneg_kept": kh, "meaning_kept": km,
                                "attention_fail": ko > check["max_kept_other"]}
    assert set(kept_by_rater) == set(RATERS), "both raters' sheets are needed"
    k1, k2 = (kept_by_rater[x] for x in RATERS)
    res["agreement_S_rows"] = {"percent": float(np.mean([k1[r] == k2[r] for r in S])), "kappa": cohen_kappa([k1[r] for r in S], [k2[r] for r in S])}
    both = lambda rows: sum(k1[r] and k2[r] for r in rows)
    lenient = {"hardneg": [x for x in RATERS if use_hn and res["raters"][x]["hardneg_kept"] >= hn_limit],
               "meaning": [x for x in RATERS if use_m and res["raters"][x]["meaning_kept"] >= m_limit]}
    res["pair"] = {"hardneg_kept_by_both": both(CH), "meaning_kept_by_both": both(CM),
                   "attention_fail": (use_hn and both(CH) >= hn_limit) or (use_m and both(CM) >= m_limit),
                   "lenient_by_part": lenient,
                   "lenient_raters": [x for x in RATERS if x in lenient["hardneg"] or x in lenient["meaning"]]}
    rows = sorted(key)
    halves = {"first": rows[:len(rows) // 2], "second": rows[len(rows) // 2:]}
    res["halves"] = {h: {x: {"S_rows": sum(key[r]["kind"] == "S_para1" for r in rr),
                             "S_rows_kept": sum(kept_by_rater[x][r] for r in rr if key[r]["kind"] == "S_para1"),
                             "controls": sum(key[r]["kind"] != "S_para1" for r in rr),
                             "controls_kept": sum(kept_by_rater[x][r] for r in rr if key[r]["kind"] != "S_para1"),
                             **{f"{k}_kept": sum(kept_by_rater[x][r] for r in rr if key[r]["kind"] == k)
                                for k in ("C_other", "C_hardneg", "C_meaning")}}
                         for x in RATERS} for h, rr in halves.items()}
    items = {key[r]["id"] for r in S}
    res["items_checked"] = len(items)
    res["items_excluded"] = len(excluded)
    res["excluded"] = sorted(excluded)
    for rater in kept_by_rater:
        res["raters"][rater]["declared_no_llm"] = declarations.get(rater) is True
    res["exploratory"] = res["pair"]["attention_fail"] or any(v["attention_fail"] or not v["declared_no_llm"] for v in res["raters"].values())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge-sheet", "build", "collect", "score"])
    ap.add_argument("--config")
    ap.add_argument("--rater")
    ap.add_argument("--file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.out) / "check"
    if a.cmd == "judge-sheet":
        judge_sheet(a.out)
    elif a.cmd == "build":
        build(load_config(a.config), a.out)
    elif a.cmd == "collect":
        print(json.dumps(collect(a.out, a.rater, a.file)))
    else:
        cfg = load_config(a.config)
        key = {int(r): p for r, p in json.loads((d / "key.json").read_text(encoding="utf-8")).items()}
        kept = {r: read_sheet(d / f"sheet_{r}.csv") for r in RATERS}
        dec = json.loads((d / "declarations.json").read_text(encoding="utf-8")) if (d / "declarations.json").exists() else {}
        res = score(key, kept, cfg["check"], dec)
        (d / "result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "excluded"}, indent=1))
