"""Human adjudication of DeBERTa's stable errors (post-hoc, NOT preregistered; attack-me round 1, R2 point 5).

  python -X utf8 experiments/20261110-adjudication/scripts/adjudication.py build
  python -X utf8 experiments/20261110-adjudication/scripts/adjudication.py collect --rater rater1 --file RETURNED.csv
  python -X utf8 experiments/20261110-adjudication/scripts/adjudication.py analyze
(from the PJ root, with .venv-win). Design: .steering/20261110-paper-write/adjudication-design.md.

build
  Rebuilds, with the frozen code (src/eval_analysis.py, src/rep_analysis.py, src/analysis.py) and without changing
  it, the stable errors of DeBERTa on WiC test (study 1) and MCL-WiC (study 2), checks that they equal the frozen
  counts.json lists, and checks that the frozen detection rates and A - control intervals of the four DeBERTa cells
  (x Qwen, x OLMo, both studies) are reproduced within 1e-9. It then draws, per study, as many items that DeBERTa
  answered correctly and stably (the original prediction equals the reference and every form of set S gives the
  same prediction; same eligibility as the errors; restricted to the items of the DeBERTa x Qwen table) as there
  are stable errors, stratified so that the reference labels of the controls have the same counts as those of the
  errors (seed in config.json). All items of both studies are shuffled together into one sheet.
  Written once, never overwritten:
    private/sheet.csv   rows for the raters: row, word, pos, sentence1, sentence2, judgement (blank).
                        The target token is wrapped in [[ ]]. Contains WiC / MCL-WiC sentences (CC BY-NC):
                        only for the human raters; never to an external LLM, OSF or the paper.
    private/key.json    row -> id, study, role (error / control), reference, prediction, lemma. Never to raters.
    results/flags.json  per DeBERTa cell: the frozen fractional flags of every method on each stable error, and
                        its unit (lemma). Ids only, no sentences.
    results/build.json  counts and SHA-256 of the files above.
collect
  Accepts a returned sheet only if its rows and texts equal private/sheet.csv and every judgement is S, D or U.
  Writes results/ratings_<rater>.json (row -> judgement, no sentences) and its SHA-256 to results/collected.json.
analyze
  Uses the raters listed in config.json that have been collected. Writes results/analysis.json:
  item classes (certain / contested / suspected reference error), detection and A - c on each class per DeBERTa
  cell (lemma cluster bootstrap, 10,000 draws, frozen seed), the same by predicted label, agreement of each rater
  with the reference on controls and on errors, rater-rater agreement and Cohen's kappa, and the fixed reading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PRIV, RES = EXP / "private", EXP / "results"
SHEET, KEY, FLAGS, BUILD = PRIV / "sheet.csv", PRIV / "key.json", RES / "flags.json", RES / "build.json"
COLS = ["row", "word", "pos", "sentence1", "sentence2", "judgement"]
JUDGE = {"S": "T", "D": "F", "U": "U"}  # same meaning -> reference T, different -> F, cannot decide
PRIM = ("i", "A", "ii", "iii", "iv")
CONTROLS = ("i", "ii", "iii", "iv")
TOL = 1e-9


def cfg():
    return json.loads((EXP / "config.json").read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- build ----------------

def _frozen():
    sys.path.insert(0, str(ROOT / "src"))
    import analysis as AN  # noqa: F401
    import eval_analysis as EA
    import eval_data as D
    import rep_data as R
    from common import read_jsonl
    from eval_run import build_forms
    from rep_run import forms_of
    return AN, EA, D, R, read_jsonl, build_forms, forms_of


def stable_split(EA, AN, out, data, forms, ref, wn_excl, forms_S):
    """count_task's classification for DeBERTa, returning stable errors AND stable correct items."""
    preds = {f: {i: r["pred"] for i, r in EA.base_rows(out, data, "deberta", f).items()} for f in forms_S}
    ids = sorted(preds["orig"])
    non_tr = {i for i in ids if any(f not in forms[i] or i not in preds[f] for f in forms_S)}
    for f in forms_S:
        for i in ids:
            preds[f].setdefault(i, preds["orig"][i])
    ref_in = {i: ref[i] for i in ids if i not in set(wn_excl)}
    stable, _ = AN.classify_errors(preds, ref_in, excluded=non_tr | set(wn_excl))
    correct = sorted(i for i, y in ref_in.items() if y is not None and i not in non_tr
                     and preds["orig"][i] == y and all(preds[f][i] == y for f in forms_S))
    return stable, correct


def studies():
    AN, EA, D, R, read_jsonl, build_forms, forms_of = _frozen()
    s1 = ROOT / "experiments/20261103-evaluation/results/eval"
    s2 = ROOT / "experiments/20261109-rep-main/rep"
    out = {}
    # study 1: WiC test
    c1 = json.loads((s1 / "counts.json").read_text())["wic_test/deberta"]
    ref1 = json.loads((s1 / "references.json").read_text())["refs"]["wic_test"]
    items1 = {it["id"]: it for it in read_jsonl(s1 / "transform/wic_test_inputs.jsonl")}
    forms1 = build_forms(s1 / "transform", data=("wic_test",))["wic_test"]
    st1, co1 = stable_split(EA, AN, s1, "wic_test", forms1, ref1, c1["wordnet_excluded"], D.WIC_S)
    out["1"] = dict(out=s1, data="wic_test", cfg=json.loads((ROOT / "config.json").read_text(encoding="utf-8")),
                    counts=c1, ref=ref1, items=items1, stable=st1, correct=co1, unit=D.wic_unit)
    # study 2: MCL-WiC (items excluded by the paraphrase check lose para1, as in rep_analysis.count)
    c2 = json.loads((s2 / "counts.json").read_text())["mcl/deberta"]
    ref2 = json.loads((s2 / "references.json").read_text())["refs"]["mcl"]
    items2 = {it["id"]: it for it in read_jsonl(s2 / "transform/mcl_inputs.jsonl")}
    forms2 = forms_of(s2 / "transform")
    for i in json.loads((s2 / "check/result.json").read_text())["excluded"]:
        forms2[i].pop("para1", None)
    st2, co2 = stable_split(EA, AN, s2, "mcl", forms2, ref2, c2["wordnet_excluded"], D.WIC_S)
    out["2"] = dict(out=s2, data="mcl", cfg=json.loads((ROOT / "rep_config.json").read_text(encoding="utf-8")),
                    counts=c2, ref=ref2, items=items2, stable=st2, correct=co2, unit=R.unit)
    for k, s in out.items():
        assert s["stable"] == s["counts"]["stable"], f"study {k}: stable errors differ from the frozen counts.json"
    return AN, EA, out


def cell_flags(AN, EA, S):
    """Frozen flags of the DeBERTa cells (x Qwen, x OLMo restricted to the Qwen items), checked against analysis.json."""
    frozen = json.loads((S["out"] / "analysis.json").read_text())["results"]
    res, qwen_ids = {}, None
    for llm in ("qwen", "olmo"):
        tb = EA.Table(S["cfg"], S["out"], S["data"], "wic", "deberta", llm, S["counts"], S["ref"],
                      restrict=None if llm == "qwen" else qwen_ids)
        if llm == "qwen":
            qwen_ids = set(tb.ids)
        ids = tb.ids
        flags = EA.flags_over({m: tb.scores(m, ids) for m in PRIM}, ids)
        pos = set(ids)
        st = [i for i in S["stable"] if i in pos]
        u = AN.unit_index([S["unit"](S["items"][i]) for i in st])
        x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
        key = f"{S['data']}/deberta/{llm}"
        fr = frozen[key]
        w = AN.boot_weights(int(u.max()) + 1)
        ok = len(st) == fr["n_stable"] and all(abs(x[m].mean() - fr["detection"][m]) < TOL for m in PRIM)
        for c in CONTROLS:
            d = AN.diff_ci(x["A"], x[c], u, w)
            ok &= all(abs(d[k] - fr["diff_qmean"][c][k]) < TOL for k in ("d", "lo", "hi"))
        assert ok, f"does not reproduce the frozen analysis: {key}"
        res[key] = {"stable": {i: {"unit": S["unit"](S["items"][i]), "yhat": tb.yhat[i], "ref": S["ref"][i],
                                   "flags": {m: float(flags[m][i]) for m in PRIM}} for i in st}}
    return res, qwen_ids


def mark(sentence, idx):
    tok = sentence.split(" ")
    assert 0 <= idx < len(tok), (sentence[:20], idx)
    tok[idx] = f"[[{tok[idx]}]]"
    return " ".join(tok)


def draw_controls(rng, correct, ref, errors):
    """As many controls as errors, with the same counts of reference labels (stratified, without replacement)."""
    out = []
    for y in ("T", "F"):
        need = sum(ref[i] == y for i in errors)
        pool = sorted(i for i in correct if ref[i] == y)
        assert len(pool) >= need, f"not enough stable correct items with reference {y}: {len(pool)} < {need}"
        out += [pool[j] for j in sorted(rng.choice(len(pool), size=need, replace=False))]
    return out


def build(_a):
    for f in (SHEET, KEY, FLAGS, BUILD):
        assert not f.exists(), f"{f} exists; the sheet is built once and never overwritten"
    c = cfg()
    AN, EA, S = studies()
    entries, flags, rep = [], {}, {"studies": {}}
    for k in ("1", "2"):
        s = S[k]
        fl, qwen_ids = cell_flags(AN, EA, s)
        flags.update(fl)
        errors = list(s["stable"])
        correct = [i for i in s["correct"] if i in qwen_ids]
        rng = np.random.default_rng([c["seed"], int(k)])
        ctrl = draw_controls(rng, correct, s["ref"], errors)
        for role, ids in (("error", errors), ("control", ctrl)):
            for i in ids:
                entries.append({"id": i, "study": k, "role": role, "ref": s["ref"][i],
                                "pred": ("F" if s["ref"][i] == "T" else "T") if role == "error" else s["ref"][i],
                                "item": s["items"][i]})
        rep["studies"][k] = {"data": s["data"], "stable_errors": len(errors), "stable_correct_pool": len(correct),
                             "controls": len(ctrl), "errors_ref_T": sum(s["ref"][i] == "T" for i in errors),
                             "controls_ref_T": sum(s["ref"][i] == "T" for i in ctrl),
                             "error_units": len({s["unit"](s["items"][i]) for i in errors})}
    order = np.random.default_rng([c["seed"], 0]).permutation(len(entries))
    rows, key = [], {}
    for r, j in enumerate(order, 1):
        e = entries[int(j)]
        it = e["item"]
        rows.append({"row": r, "word": it["lemma"], "pos": {"N": "noun", "V": "verb"}[it["pos"]],
                     "sentence1": mark(it["sentence1"], it["idx1"]), "sentence2": mark(it["sentence2"], it["idx2"]),
                     "judgement": ""})
        key[r] = {k2: e[k2] for k2 in ("id", "study", "role", "ref", "pred")} | {"lemma": it["lemma"]}
    PRIV.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="", encoding="utf-8-sig") as f:  # BOM so that Excel opens it as UTF-8
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    FLAGS.write_text(json.dumps(flags, indent=1), encoding="utf-8")
    rep |= {"rows": len(rows), "seed": c["seed"], "reproduces_frozen_cells": sorted(flags),
            "sha256": {"private/sheet.csv": sha(SHEET), "private/key.json": sha(KEY), "results/flags.json": sha(FLAGS)}}
    BUILD.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


# ---------------- collect ----------------

def read_returned(src):
    with open(src, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate(built, rows, name="returned sheet"):
    """built: {row: sheet row}; rows: returned rows. Returns {row: S/D/U}; raises on any difference."""
    got = {int(r["row"]): r for r in rows}
    if not (len(got) == len(rows) and set(got) == set(built)):
        raise ValueError(f"{name}: rows differ from the built sheet")
    for n, b in built.items():
        for col in ("word", "pos", "sentence1", "sentence2"):
            if got[n][col] != b[col]:
                raise ValueError(f"{name}: row {n} column {col} differs from the built sheet")
        if got[n]["judgement"].strip().upper() not in JUDGE:
            raise ValueError(f"{name}: row {n} judgement must be S, D or U (got {got[n]['judgement']!r})")
    return {n: got[n]["judgement"].strip().upper() for n in sorted(got)}


def collect(a):
    assert a.rater in cfg()["raters"], f"unknown rater {a.rater}"
    with open(SHEET, encoding="utf-8-sig", newline="") as f:
        built = {int(r["row"]): r for r in csv.DictReader(f)}
    ratings = validate(built, read_returned(a.file), a.file)
    out = RES / f"ratings_{a.rater}.json"
    assert not out.exists(), f"{out} exists; collect once per rater"
    out.write_text(json.dumps(ratings, indent=0), encoding="utf-8")
    log = RES / "collected.json"
    rec = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    rec[a.rater] = {"file": Path(a.file).name, "sha256": sha(a.file), "ratings_sha256": sha(out)}
    log.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec[a.rater]))


# ---------------- analyze (pure functions; tested in test_adjudication.py) ----------------

def classify(ref, judgements):
    """ref: T/F; judgements: list of T/F/U, one per rater. certain = every rater agrees with the reference;
    contested = otherwise; also 'suspect' = every rater gives the opposite of the reference."""
    certain = all(j == ref for j in judgements)
    opp = "F" if ref == "T" else "T"
    return {"class": "certain" if certain else "contested", "suspect": all(j == opp for j in judgements)}


def cohen_kappa(a, b, cats=("T", "F", "U")):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None


def agreement(judgements, refs):
    n = len(refs)
    if n == 0:
        return None
    return {"n": n, "agree": sum(j == r for j, r in zip(judgements, refs)) / n,
            "opposite": sum(j not in (r, "U") for j, r in zip(judgements, refs)) / n,
            "undecided": sum(j == "U" for j in judgements) / n}


def subset_stats(AN, cell, ids):
    """Detection of every method and A - c with lemma cluster bootstrap (frozen seed and draws per subset)."""
    if not ids:
        return {"n": 0}
    u = AN.unit_index([cell[i]["unit"] for i in ids])
    x = {m: np.array([cell[i]["flags"][m] for i in ids]) for m in PRIM}
    w = AN.boot_weights(int(u.max()) + 1)
    out = {"n": len(ids), "units": int(u.max()) + 1, "detection": {m: float(x[m].mean()) for m in PRIM},
           "A_minus": {c: AN.diff_ci(x["A"], x[c], u, w) for c in CONTROLS}}
    out["by_yhat"] = {y: {"n": len(s), "detection": {m: float(np.mean([cell[i]["flags"][m] for i in s])) for m in PRIM}}
                      for y in ("T", "F") for s in [[i for i in ids if cell[i]["yhat"] == y]] if s}
    return out


def reading(cert, cont):
    """Fixed reading of A - (ii) (adjudication-design.md). Uses the lower ends of the 95% intervals."""
    if cert.get("n", 0) == 0 or cont.get("n", 0) == 0:
        return "not computed (an empty class)"
    lc, lt = cert["A_minus"]["ii"]["lo"], cont["A_minus"]["ii"]["lo"]
    if lc > 0 and lc > lt:
        return "the structured gain remained on the errors the reference supports"
    if lt > 0 and lt > lc:
        return "the structured gain depends on the contested items"
    return "indistinguishable"


def analyze_ratings(AN, key, ratings, flags):
    """key: {row: entry}; ratings: {rater: {row: S/D/U}}; flags: results/flags.json."""
    raters = sorted(ratings)
    J = {r: {n: JUDGE[v] for n, v in ratings[r].items()} for r in raters}
    res = {"raters": raters, "n_raters": len(raters)}
    by_id = {}
    for n, e in key.items():
        if e["role"] == "error":
            by_id[(e["study"], e["id"])] = classify(e["ref"], [J[r][n] for r in raters])
    res["classes"] = {}
    for st in ("1", "2"):
        v = [c for (s, _), c in by_id.items() if s == st]
        res["classes"][st] = {"errors": len(v), "certain": sum(c["class"] == "certain" for c in v),
                              "contested": sum(c["class"] == "contested" for c in v),
                              "suspected_reference_error": sum(c["suspect"] for c in v)}
    res["agreement_with_reference"] = {}
    for r in raters:
        res["agreement_with_reference"][r] = {
            f"study{st}_{role}": agreement([J[r][n] for n, e in key.items() if e["study"] == st and e["role"] == role],
                                           [e["ref"] for n, e in key.items() if e["study"] == st and e["role"] == role])
            for st in ("1", "2") for role in ("control", "error")}
    if len(raters) >= 2:
        a, b = raters[0], raters[1]
        res["rater_agreement"] = {}
        for name, sel in (("all", lambda e: True), ("controls", lambda e: e["role"] == "control"),
                          ("errors", lambda e: e["role"] == "error")):
            rows = [n for n, e in key.items() if sel(e)]
            ja, jb = [J[a][n] for n in rows], [J[b][n] for n in rows]
            res["rater_agreement"][name] = {"n": len(rows), "percent": sum(x == y for x, y in zip(ja, jb)) / len(rows),
                                            "kappa3": cohen_kappa(ja, jb)}
    res["cells"] = {}
    for cell_name, cell in flags.items():
        st = "1" if cell_name.startswith("wic_test/") else "2"
        cell = cell["stable"]
        cert = [i for i in cell if by_id[(st, i)]["class"] == "certain"]
        cont = [i for i in cell if by_id[(st, i)]["class"] == "contested"]
        sus = [i for i in cell if by_id[(st, i)]["suspect"]]
        c = {"certain": subset_stats(AN, cell, cert), "contested": subset_stats(AN, cell, cont),
             "suspected_reference_error": subset_stats(AN, cell, sus)}
        c["reading"] = reading(c["certain"], c["contested"])
        res["cells"][cell_name] = c
    return res


def analyze(_a):
    c = cfg()
    sys.path.insert(0, str(ROOT / "src"))
    import analysis as AN
    assert sha(SHEET) == json.loads(BUILD.read_text())["sha256"]["private/sheet.csv"], "sheet changed after build"
    key = {int(n): e for n, e in json.loads(KEY.read_text(encoding="utf-8")).items()}
    flags = json.loads(FLAGS.read_text(encoding="utf-8"))
    ratings = {}
    for r in c["raters"]:
        p = RES / f"ratings_{r}.json"
        if p.exists():
            ratings[r] = {int(n): v for n, v in json.loads(p.read_text(encoding="utf-8")).items()}
    assert ratings, "no ratings collected"
    decl = json.loads((RES / "declarations.json").read_text(encoding="utf-8")) if (RES / "declarations.json").exists() else {}
    res = analyze_ratings(AN, key, ratings, flags)
    res["declared_alone_without_llm"] = {r: bool(decl.get(r)) for r in ratings}
    res["note"] = "post-hoc, not preregistered (adjudication-design.md)"
    if len(ratings) < len(c["raters"]):
        res["note"] += f"; only {len(ratings)} of {len(c['raters'])} raters collected"
    (RES / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"classes": res["classes"], "readings": {k: v["reading"] for k, v in res["cells"].items()}}, indent=1))


FROZEN_INPUTS = [
    "experiments/20261103-evaluation/results/eval/counts.json",
    "experiments/20261103-evaluation/results/eval/references.json",
    "experiments/20261103-evaluation/results/eval/analysis.json",
    "experiments/20261103-evaluation/results/eval/transform/wic_test_inputs.jsonl",
    "experiments/20261109-rep-main/rep/counts.json",
    "experiments/20261109-rep-main/rep/references.json",
    "experiments/20261109-rep-main/rep/analysis.json",
    "experiments/20261109-rep-main/rep/check/result.json",
    "experiments/20261109-rep-main/rep/transform/mcl_inputs.jsonl",
]
FREEZE_FILES = [
    ".steering/20261110-paper-write/adjudication-design.md",
    "experiments/20261110-adjudication/scripts/adjudication.py",
    "experiments/20261110-adjudication/scripts/test_adjudication.py",
    "experiments/20261110-adjudication/rater_guide.md",
    "experiments/20261110-adjudication/config.json",
    "experiments/20261110-adjudication/SEED",
    "experiments/20261110-adjudication/results/build.json",
]


def manifest(_a):
    """SHA-256 of everything fixed before the rating starts (for the OSF registration). The sheet, the key and the
    flags (per-item reference labels and lemmas of WiC / MCL-WiC) are listed by hash only (never uploaded)."""
    m = {"freeze": {p: sha(ROOT / p) for p in FREEZE_FILES},
         "by_hash_only": {"private/sheet.csv": sha(SHEET), "private/key.json": sha(KEY), "results/flags.json": sha(FLAGS)},
         "frozen_inputs": {p: sha(ROOT / p) for p in FROZEN_INPUTS}}
    mf = EXP / "freeze_manifest.json"
    mf.write_text(json.dumps(m, indent=1), encoding="utf-8")
    import zipfile
    z = EXP / "freeze_bundle.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:  # no file with WiC / MCL-WiC sentences
        zf.write(mf, "freeze_manifest.json")
        for p in FREEZE_FILES:
            b = (ROOT / p).read_bytes()
            assert hashlib.sha256(b).hexdigest() == m["freeze"][p], p
            zf.writestr(p, b)
    info = {"manifest_sha256": sha(mf), "zip": z.name, "zip_bytes": z.stat().st_size, "zip_sha256": sha(z),
            "files_in_zip": len(FREEZE_FILES) + 1}
    (EXP / "bundle.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    sp.add_parser("manifest")
    p = sp.add_parser("collect")
    p.add_argument("--rater", required=True)
    p.add_argument("--file", required=True)
    sp.add_parser("analyze")
    a = ap.parse_args()
    {"build": build, "manifest": manifest, "collect": collect, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
