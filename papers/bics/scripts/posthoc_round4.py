"""Post-hoc descriptive analyses requested in attack-me round 4 (NOT preregistered; reported in S8/S9 with [P]).

Reuses posthoc_round1.cells() / posthoc_round2.debias() and checks the frozen detection rates and A - control
intervals again (1e-9) before computing anything new.
  python -X utf8 papers/bics/scripts/posthoc_round4.py   (from the PJ root)
Analyses:
  S1 adjudication, the 68 (Study 1) / 8 (Study 2) stable errors on which both raters chose the label opposite to the
     reference: detection of A, (i), (ii), (iii), debiased (ii); A - (ii) and A - debiased (ii) (unit bootstrap).
     The shared-inventory explanation predicts A - (ii) > 0 here (A siding with the WordNet-derived reference).
  S2 AURC: (i)+A - A with the frozen risk_coverage interval (lemma resampling, flags fixed), for the seven WiC-type
     cells and ChaosNLI; A - (i) and (i)+A - (i) are already in posthoc_round2.json / posthoc_round3.json.
  S3 confidence against correct items: AUROC of (i) and A for stable errors against correctly classified ranked items
     only (unstable errors left out), per positive so that the unit bootstrap applies; and (i)'s detection rate
     minus the random expectation 0.30 with the unit bootstrap."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ph1 = load("ph1", ROOT / "papers/bics/scripts/posthoc_round1.py")
ph2 = load("ph2", ROOT / "papers/bics/scripts/posthoc_round2.py")
AN, EA, RA = ph1.AN, ph1.EA, ph2.RA
PRIM = ph1.PRIM
TOL = 1e-9
RANDOM = float(np.mean(AN.QS))


def ci(d):
    return {"d": float(d["d"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def per_positive_auc(scores, pos_mask, neg_mask):
    s = np.asarray(scores, float)
    neg = np.sort(s[neg_mask])
    p = s[pos_mask]
    lt = np.searchsorted(neg, p, side="left")
    le = np.searchsorted(neg, p, side="right")
    return (lt + 0.5 * (le - lt)) / len(neg)


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    adj = ROOT / "experiments/20261110-adjudication"
    key = json.loads((adj / "private/key.json").read_text(encoding="utf-8"))
    R = {r: json.loads((adj / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
    J = {"S": "T", "D": "F", "U": "U"}
    opposite = {"T": "F", "F": "T"}
    both_opp = set()
    for k, v in key.items():
        if v["role"] == "error" and all(J[R[r][k]] == opposite[v["ref"]] for r in R):
            both_opp.add(v["id"])

    res = {"note": "post hoc [P], not preregistered", "S1_both_opposite": {}, "S2_aurc_iA_minus_A": {},
           "S3_confidence_vs_correct": {}}
    for study, data, base, llm, tb, stable, units in ph1.cells():
        ids = tb.ids
        pos = {i: j for j, i in enumerate(ids)}
        prim = {m: tb.scores(m, ids) for m in PRIM}
        st = [i for i in stable if i in pos]
        u = AN.unit_index([units[i] for i in st])
        w = AN.boot_weights(int(u.max()) + 1)
        k = f"{data}/{base}/{llm}"
        fr = frozen[study][k]
        flags = EA.flags_over(prim, ids)
        x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
        ok = len(st) == fr["n_stable"] and all(abs(x[m].mean() - fr["detection"][m]) < TOL for m in PRIM)
        for c in ("i", "ii", "iii", "iv"):
            d = AN.diff_ci(x["A"], x[c], u, w)
            ok &= all(abs(d[q] - fr["diff_qmean"][c][q]) < TOL for q in ("d", "lo", "hi"))
        assert ok, f"does not reproduce the frozen analysis: {study} {k}"
        cell = f"study{study}:{k}"

        # S1 both raters opposite (DeBERTa only: the adjudicated classifier)
        if base == "deberta":
            shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
            fl = {m: AN.q_mean_flags(prim[m]) for m in ("A", "i", "ii", "iii")}
            fl["ii_deb"] = AN.q_mean_flags(ph2.debias(tb, ids, "ii", shareA)[0])
            sid = [i for i in st if i in both_opp]
            uu = AN.unit_index([units[i] for i in sid])
            ww = AN.boot_weights(int(uu.max()) + 1)
            xs = {m: np.array([fl[m][pos[i]] for i in sid]) for m in fl}
            res["S1_both_opposite"][cell] = {
                "n": len(sid), "units": int(uu.max()) + 1,
                "detection": {m: float(v.mean()) for m, v in xs.items()},
                "A_minus_ii": ci(AN.diff_ci(xs["A"], xs["ii"], uu, ww)),
                "A_minus_ii_deb": ci(AN.diff_ci(xs["A"], xs["ii_deb"], uu, ww))}

        # S2 AURC (i)+A - A (frozen risk_coverage; keys relabelled so that its "i+A minus i" is (i)+A - A)
        wr = [i for i in ids if tb.ref.get(i) is not None]
        sel = [pos[i] for i in wr]
        iA = (EA.rankdata(prim["i"]) + EA.rankdata(prim["A"])) / 2
        err = np.array([tb.yhat[i] != tb.ref[i] for i in wr], float)
        rc = RA.risk_coverage({"i": np.asarray(prim["A"])[sel], "i+A": np.asarray(iA)[sel]}, err,
                              AN.unit_index([units[i] for i in wr]))
        res["S2_aurc_iA_minus_A"][cell] = {"aurc_A": rc["aurc"]["i"], "aurc_iA": rc["aurc"]["i+A"],
                                           "iA_minus_A": rc["aurc_i+A_minus_i"]}

        # S3 stable errors against correct items only
        correct = np.array([tb.ref.get(i) is not None and tb.yhat[i] == tb.ref[i] for i in ids])
        sti = np.array([pos[i] for i in st])
        mask = np.zeros(len(ids), bool)
        mask[sti] = True
        order_in_ids = np.sort(sti)
        back = {j: q for q, j in enumerate(order_in_ids)}
        pp = {}
        for m in ("i", "A"):
            v = per_positive_auc(prim[m], mask, correct)
            pp[m] = np.array([v[back[j]] for j in sti])
        n_u = int(u.max()) + 1
        s_i, nu = ph1.unit_sums(x["i"] - RANDOM, u, n_u)
        boot = (w @ s_i) / (w @ nu)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        res["S3_confidence_vs_correct"][cell] = {
            "n_correct": int(correct.sum()),
            "auroc_vs_correct": {m: float(v.mean()) for m, v in pp.items()},
            "auroc_i_minus_half": ci(AN.diff_ci(pp["i"], np.full(len(pp["i"]), 0.5), u, w)),
            "detection_i_minus_random": {"d": float(x["i"].mean() - RANDOM), "lo": float(lo), "hi": float(hi)}}

    # S2 for ChaosNLI (as posthoc_round3 R4)
    cfg1 = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cnt1 = json.loads((ph1.S1 / "counts.json").read_text())
    ref1 = json.loads((ph1.S1 / "references.json").read_text())["refs"]["chaos"]
    sys.path.insert(0, str(ROOT / "src"))
    import eval_data as D
    items = {it["id"]: it for it in ph1.jl(ph1.S1 / "transform/chaos_inputs.jsonl")}
    qids = None
    for llm in ("qwen", "olmo"):
        tb = EA.Table(cfg1, ph1.S1, "chaos", "nli", "roberta", llm, cnt1["chaos/roberta"], ref1,
                      restrict=None if llm == "qwen" else qids)
        if llm == "qwen":
            qids = set(tb.ids)
        ids = tb.ids
        wr = [i for i in ids if ref1.get(i) is not None]
        pos = {i: j for j, i in enumerate(ids)}
        sel = [pos[i] for i in wr]
        sA, si = tb.scores("A", ids), tb.scores("i", ids)
        iA = (EA.rankdata(si) + EA.rankdata(sA)) / 2
        err = np.array([tb.yhat[i] != ref1[i] for i in wr], float)
        rc = RA.risk_coverage({"i": np.asarray(sA)[sel], "i+A": np.asarray(iA)[sel]}, err,
                              AN.unit_index([D.nli_unit(items[i]) for i in wr]))
        res["S2_aurc_iA_minus_A"][f"chaos/roberta/{llm}"] = {"aurc_A": rc["aurc"]["i"], "aurc_iA": rc["aurc"]["i+A"],
                                                             "iA_minus_A": rc["aurc_i+A_minus_i"]}

    out = ROOT / "papers/bics/tables/posthoc_round4.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
