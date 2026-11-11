"""Post-hoc descriptive analyses requested in attack-me round 1 (NOT preregistered; reported in S8 with that label).
Reuses the frozen analysis code (src/eval_analysis.py, src/rep_analysis.py, src/analysis.py) without changing it:
items, scores, fractional flags, stable errors and units are rebuilt exactly as in the frozen analysis, and the
script first checks that it reproduces the frozen detection rates and A - control intervals.
  python -X utf8 papers/bics/scripts/posthoc_round1.py   (from the PJ root; needs .nltk-win/corpora/wordnet.zip,
  SHA-256 equal to config.json wordnet_zip_sha256)
Analyses:
  P1 between-study difference of (A - c), c in (i),(ii),(iii), for the three classifier x LLM pairs run in both studies
     (independent cluster bootstraps per study, 10,000 draws, seeds 20261102 / 20261103).
  P2 share of A - (i) matched by the unstructured judgement: ((ii) - (i)) / (A - (i)), per configuration, paired cluster
     bootstrap (frozen seed); also (iii).
  P3 granularity: detection by predicted label (= by gold label for errors), within-stratum A - (ii), and the rate at
     which each method predicts "different" over all ranked items.
  P4 (reviewer R2) WiC test split by whether both sentences are WordNet 3.0 usage examples of the target lemma-POS
     (a proxy for the WordNet-derived part of WiC; [19] built WiC from WordNet, VerbNet and Wiktionary examples).
Sentences are read only to compute the P4 proxy; no sentence is written to the outputs."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import analysis as AN  # noqa: E402
import eval_analysis as EA  # noqa: E402
import eval_data as D  # noqa: E402
import inventory  # noqa: E402
import rep_data as R  # noqa: E402

S1 = ROOT / "experiments/20261103-evaluation/results/eval"
S2 = ROOT / "experiments/20261109-rep-main/rep"
OUT = ROOT / "papers/bics/tables"
PRIM = ("i", "A", "ii", "iii", "iv")
TOL = 1e-9


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def cells():
    """Yield (study, data, base, llm, tb, stable ids, unit labels) for every WiC-type cell, as the frozen code builds them."""
    cfg1 = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cnt1 = json.loads((S1 / "counts.json").read_text())
    ref1 = json.loads((S1 / "references.json").read_text())["refs"]
    units1 = {it["id"]: D.wic_unit(it) for it in jl(S1 / "transform/wic_test_inputs.jsonl")}
    prim_ids = {}
    for llm in ("qwen", "olmo"):
        for base in (("deberta", "flan") if llm == "qwen" else ("deberta",)):
            c = cnt1[f"wic_test/{base}"]
            tb = EA.Table(cfg1, S1, "wic_test", "wic", base, llm, c, ref1["wic_test"],
                          restrict=None if llm == "qwen" else prim_ids.get(base))
            if llm == "qwen":
                prim_ids[base] = set(tb.ids)
            yield "1", "wic_test", base, llm, tb, c["stable"], units1
    cfg2 = json.loads((ROOT / "rep_config.json").read_text(encoding="utf-8"))
    cnt2 = json.loads((S2 / "counts.json").read_text())
    ref2 = json.loads((S2 / "references.json").read_text())["refs"]["mcl"]
    units2 = {it["id"]: R.unit(it) for it in jl(S2 / "transform/mcl_inputs.jsonl")}
    prim_ids = {}
    for llm in ("qwen", "olmo"):
        for base in ("deberta", "flan"):
            c = cnt2[f"mcl/{base}"]
            tb = EA.Table(cfg2, S2, "mcl", "wic", base, llm, c, ref2, restrict=None if llm == "qwen" else prim_ids.get(base))
            if llm == "qwen":
                prim_ids[base] = set(tb.ids)
            yield "2", "mcl", base, llm, tb, c["stable"], units2


def ci(est):
    lo, hi = np.quantile(est, [0.025, 0.975])
    return float(lo), float(hi)


def unit_sums(v, u, n_u):
    return np.bincount(u, weights=v, minlength=n_u), np.bincount(u, minlength=n_u).astype(float)


def main():
    p = ROOT / ".nltk-win/corpora/wordnet.zip"
    frozen_sha = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["wordnet_zip_sha256"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == frozen_sha, "WordNet zip differs from the frozen one"
    frozen = {"1": json.loads((S1 / "analysis.json").read_text())["results"],
              "2": json.loads((S2 / "analysis.json").read_text())["results"]}
    W = {}  # per cell: x (flags on stable errors), unit index, flags over all items, tb
    res = {"check_reproduces_frozen": {}, "P1_between_study": {}, "P2_share": {}, "P3_granularity": {}, "P4_wordnet_examples": {}}
    for study, data, base, llm, tb, stable, units in cells():
        ids = tb.ids
        pos = {i: j for j, i in enumerate(ids)}
        prim = {m: tb.scores(m, ids) for m in PRIM}
        flags = EA.flags_over(prim, ids)
        st = [i for i in stable if i in pos]
        u = AN.unit_index([units[i] for i in st])
        x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
        key = f"{data}/{base}/{llm}"
        # reproduction check against the frozen analysis.json
        fr = frozen[study][key]
        w = AN.boot_weights(int(u.max()) + 1)
        ok = len(st) == fr["n_stable"] and all(abs(x[m].mean() - fr["detection"][m]) < TOL for m in PRIM)
        for c in ("i", "ii", "iii", "iv"):
            d = AN.diff_ci(x["A"], x[c], u, w)
            ok &= all(abs(d[k] - fr["diff_qmean"][c][k]) < TOL for k in ("d", "lo", "hi"))
        res["check_reproduces_frozen"][f"study{study}:{key}"] = bool(ok)
        assert ok, f"does not reproduce the frozen analysis: {study} {key}"
        W[(study, base, llm)] = dict(x=x, u=u, flags=flags, tb=tb, st=st, ids=ids, w=w)

        # P2: share of A-(i) matched by (ii) (and (iii)), paired cluster bootstrap with the frozen draws
        n_u = int(u.max()) + 1
        sA_i, nu = unit_sums(x["A"] - x["i"], u, n_u)
        out = {}
        for c in ("ii", "iii"):
            sc_i, _ = unit_sums(x[c] - x["i"], u, n_u)
            num, den = (w @ sc_i) / (w @ nu), (w @ sA_i) / (w @ nu)
            point = float((x[c] - x["i"]).mean() / (x["A"] - x["i"]).mean())
            ok_den = den > 0
            lo, hi = ci(num[ok_den] / den[ok_den])
            out[c] = {"share": point, "lo": lo, "hi": hi, "denominator_le0_frac": float(1 - ok_den.mean())}
        res["P2_share"][f"study{study}:{key}"] = out

        # P3: granularity. Stable errors with yhat = T have gold F (and vice versa).
        g = {}
        for y in ("T", "F"):
            sid = [i for i in st if tb.yhat[i] == y]
            g[f"yhat_{y}"] = {"n_stable": len(sid),
                              "detection": {m: float(np.mean([flags[m][i] for i in sid])) if sid else None for m in PRIM}}
        g["share_predict_different_all_items"] = {m: float(np.mean([EA.method_pred(tb.rows[m][i], "wic") == "F" for i in ids]))
                                                  for m in ("A", "ii", "iii", "iv")}
        g["share_predict_different_all_items"]["base"] = float(np.mean([tb.yhat[i] == "F" for i in ids]))
        g["share_gold_different_all_items"] = float(np.mean([tb.ref[i] == "F" for i in ids]))
        res["P3_granularity"][f"study{study}:{key}"] = g

    # P1: between-study differences for pairs present in both studies
    for base, llm in (("deberta", "qwen"), ("flan", "qwen"), ("deberta", "olmo")):
        a, b = W[("1", base, llm)], W[("2", base, llm)]
        wa = a["w"]
        wb = AN.boot_weights(int(b["u"].max()) + 1, seed=AN.SEED + 1)
        out = {}
        for c in ("i", "ii", "iii"):
            ests = []
            for v, w_ in ((a, wa), (b, wb)):
                n_u = int(v["u"].max()) + 1
                s, nu = unit_sums(v["x"]["A"] - v["x"][c], v["u"], n_u)
                ests.append((w_ @ s) / (w_ @ nu))
            dd = ests[0] - ests[1]
            point = float((a["x"]["A"] - a["x"][c]).mean() - (b["x"]["A"] - b["x"][c]).mean())
            lo, hi = ci(dd)
            out[c] = {"study1_minus_study2": point, "lo": lo, "hi": hi}
        res["P1_between_study"][f"{base}/{llm}"] = out

    # P4: WiC test, both sentences are WordNet usage examples of the lemma-POS vs not (study 1 cells only)
    wn = inventory.get_wn(str(ROOT / ".nltk-win"))
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    pmap = {"N": "n", "V": "v"}
    items = {it["id"]: it for it in jl(S1 / "transform/wic_test_inputs.jsonl")}
    in_wn = {}
    for i, it in items.items():
        ex = {norm(e) for s in wn.synsets(it["lemma"], pmap[it["pos"]]) for e in s.examples()}
        in_wn[i] = all(norm(it[f"sentence{j}"]) in ex for j in (1, 2))
    res["P4_wordnet_examples"]["items_both_sentences_in_wordnet_examples"] = int(sum(in_wn.values()))
    res["P4_wordnet_examples"]["items_total"] = len(in_wn)
    for (study, base, llm), v in W.items():
        if study != "1":
            continue
        out = {}
        for name, keep in (("wordnet_examples", True), ("other", False)):
            sid = [i for i in v["st"] if in_wn[i] == keep]
            if not sid:
                continue
            uu = AN.unit_index([D.wic_unit(items[i]) for i in sid])
            xs = {m: np.array([v["flags"][m][i] for i in sid]) for m in PRIM}
            ww = AN.boot_weights(int(uu.max()) + 1)
            out[name] = {"n_stable": len(sid), "units": int(uu.max()) + 1,
                         "detection": {m: float(xs[m].mean()) for m in PRIM},
                         "A_minus": {c: AN.diff_ci(xs["A"], xs[c], uu, ww) for c in ("i", "ii", "iii")}}
        res["P4_wordnet_examples"][f"study1:wic_test/{base}/{llm}"] = out

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "posthoc_round1.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["check_reproduces_frozen"], indent=0))
    print("->", OUT / "posthoc_round1.json")


if __name__ == "__main__":
    main()
