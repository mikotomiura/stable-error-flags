"""Post-hoc descriptive analyses requested in attack-me round 3 (NOT preregistered; reported in S8 with [P]).

Reuses posthoc_round1.cells() / posthoc_round2.debias() and checks the frozen detection rates again (1e-9).
  python -X utf8 papers/bics/scripts/posthoc_round3.py   (from the PJ root)
Analyses:
  R1 sensitivity of the debiased controls to the target share of "different": A's share (P6), 0.5, and the classifier's
     own share; for (ii) and (iii). Also stratified flags: the top q is flagged within each yhat stratum separately,
     which removes the allocation of flags between strata, so A - (ii) compares rankings within strata only.
  R2 union flags at the same budget: for each q, items are taken alternately from A's and (i)'s rankings (skipping
     duplicates) until qN items are flagged; detection on stable errors vs A alone and the mean of ranks (i)+A.
  R3 AUROC of each score for stable errors against all other ranked items (per-positive values, so the frozen unit
     bootstrap applies); A - (ii) and its between-study difference (independent bootstraps per study, as P1).
  R4 judges on ChaosNLI (accuracy, label distribution) and AURC of (i), A, (i)+A on ChaosNLI.
  R5 classifier saturation: share of ranked items with P(yhat) >= 1 - 1e-6 (fp32 outputs).
  R6 FLAN lemma overlap between studies; DeBERTa accuracy on MCL-WiC dev vs test.
  R7 adjudication: A - debiased (ii) on confirmed / contested errors (Qwen), confirmed - contested difference of
     A - (ii) with a joint unit bootstrap, and the share of yhat = different in each class.
  R8 per-item numeric outputs for the supplementary archive (ids, labels and probabilities only; no text)."""
import csv
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
QS = AN.QS


def ci(d):
    return {"d": float(d["d"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def debias_to(tb, ids, m, share):
    return ph2.debias(tb, ids, m, share)[0]


def stratified_flags(scores, strata):
    """q-mean flags with the top q taken within each stratum (fractional ties as in the frozen code)."""
    scores, strata = np.asarray(scores, float), np.asarray(strata)
    out = np.zeros(len(scores))
    for s in np.unique(strata):
        ix = np.where(strata == s)[0]
        out[ix] = AN.q_mean_flags(scores[ix])
    return out


def union_flags(s_a, s_b):
    """q-mean flags of the alternating union of two rankings (higher score = flagged first)."""
    n = len(s_a)
    oa, ob = np.argsort(-np.asarray(s_a), kind="stable"), np.argsort(-np.asarray(s_b), kind="stable")
    order, seen, ia, ib = [], set(), 0, 0
    while len(order) < n:
        for o, k in ((oa, "a"), (ob, "b")):
            idx = ia if k == "a" else ib
            while idx < n and o[idx] in seen:
                idx += 1
            if idx < n:
                order.append(o[idx]); seen.add(o[idx]); idx += 1
            if k == "a":
                ia = idx
            else:
                ib = idx
    rank = np.empty(n, int)
    rank[np.array(order)] = np.arange(n)
    return np.mean([(rank < int(round(q * n))).astype(float) for q in QS], axis=0)


def per_positive_auc(scores, pos_mask):
    s = np.asarray(scores, float)
    neg = np.sort(s[~pos_mask])
    p = s[pos_mask]
    lt = np.searchsorted(neg, p, side="left")
    le = np.searchsorted(neg, p, side="right")
    return (lt + 0.5 * (le - lt)) / len(neg)


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    res = {"note": "post hoc [P], not preregistered", "R1_debias_sensitivity": {}, "R1_stratified": {},
           "R2_union": {}, "R3_auroc": {}, "R5_saturation": {}, "R6": {}, "R7_adjudication": {}}
    keep, units_by = {}, {}
    item_rows = []
    for study, data, base, llm, tb, stable, units in ph1.cells():
        ids = tb.ids
        pos = {i: j for j, i in enumerate(ids)}
        prim = {m: tb.scores(m, ids) for m in PRIM}
        st = [i for i in stable if i in pos]
        u = AN.unit_index([units[i] for i in st])
        w = AN.boot_weights(int(u.max()) + 1)
        key = f"{data}/{base}/{llm}"
        flags = EA.flags_over(prim, ids)
        x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
        assert all(abs(x[m].mean() - frozen[study][key]["detection"][m]) < TOL for m in PRIM), key
        cell = f"study{study}:{key}"
        sti = np.array([pos[i] for i in st])
        units_by[(study, base)] = {units[i] for i in st}

        # R1 sensitivity of the debiasing target
        shares = {"A": float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids])), "0.5": 0.5,
                  "classifier": float(np.mean([tb.yhat[i] == "F" for i in ids]))}
        r1 = {"targets": shares}
        for m in ("ii", "iii"):
            for t, sh in shares.items():
                s = debias_to(tb, ids, m, sh)
                f = AN.q_mean_flags(s)[sti]
                r1[f"A_minus_{m}_deb_{t}"] = ci(AN.diff_ci(x["A"], f, u, w))
        res["R1_debias_sensitivity"][cell] = r1
        yh = [tb.yhat[i] for i in ids]
        sf = {m: stratified_flags(prim[m], yh)[sti] for m in ("A", "ii", "iii", "i")}
        res["R1_stratified"][cell] = {"detection": {m: float(v.mean()) for m, v in sf.items()},
                                      "A_minus_ii": ci(AN.diff_ci(sf["A"], sf["ii"], u, w)),
                                      "A_minus_iii": ci(AN.diff_ci(sf["A"], sf["iii"], u, w))}

        # R2 union of A and (i) at the same budget
        fu = union_flags(prim["A"], prim["i"])[sti]
        mean_rank = (EA.rankdata(prim["i"]) + EA.rankdata(prim["A"])) / 2
        fm = AN.q_mean_flags(mean_rank)[sti]
        res["R2_union"][cell] = {"detection": {"A": float(x["A"].mean()), "i": float(x["i"].mean()),
                                               "union_A_i": float(fu.mean()), "mean_rank_i_A": float(fm.mean())},
                                 "union_minus_A": ci(AN.diff_ci(fu, x["A"], u, w)),
                                 "union_minus_i": ci(AN.diff_ci(fu, x["i"], u, w))}

        # R3 AUROC (stable errors vs all other ranked items)
        mask = np.zeros(len(ids), bool)
        mask[sti] = True
        ppa = {m: per_positive_auc(prim[m], mask) for m in ("A", "i", "ii", "iii")}
        ppa_order = {m: v[np.argsort(np.argsort(sti))] for m, v in ppa.items()}  # positives in sti order
        # per_positive_auc returns values in the order of positives as they appear in ids; map to st order
        order_in_ids = np.sort(sti)
        back = {j: k for k, j in enumerate(order_in_ids)}
        pp = {m: np.array([ppa[m][back[j]] for j in sti]) for m in ppa}
        res["R3_auroc"][cell] = {"auroc": {m: float(v.mean()) for m, v in pp.items()},
                                 "A_minus_i": ci(AN.diff_ci(pp["A"], pp["i"], u, w)),
                                 "A_minus_ii": ci(AN.diff_ci(pp["A"], pp["ii"], u, w))}
        keep[(study, base, llm)] = dict(pp=pp, u=u, w=w)

        # R5 classifier saturation
        pb = np.array([tb.p_base(i) for i in ids])
        res["R5_saturation"][cell] = {"share_p_yhat_ge_1_minus_1e-6": float(np.mean(pb >= 1 - 1e-6)),
                                      "share_p_yhat_eq_1": float(np.mean(pb >= 1.0))}

        # R8 per-item outputs (qwen and olmo tables differ only in the judges)
        stset = set(st)
        for i in ids:
            item_rows.append({"study": study, "dataset": "WiC test" if data == "wic_test" else "MCL-WiC En-En",
                              "item_id": i.split("_")[-1] if data == "wic_test" else i, "classifier": base, "llm": llm,
                              "reference": tb.ref.get(i), "classifier_prediction": tb.yhat[i],
                              "p_classifier_prediction": round(float(tb.p_base(i)), 6),
                              **{f"p_same_{m}": round(float(tb.rows[m][i]["p_same"]), 6) for m in ("A", "ii", "iii", "iv")},
                              "stable_error": int(i in stset)})

    # R3 between-study AUROC difference
    res["R3_between_study"] = {}
    for base, llm in (("deberta", "qwen"), ("flan", "qwen"), ("deberta", "olmo")):
        a, b = keep[("1", base, llm)], keep[("2", base, llm)]
        wb = AN.boot_weights(int(b["u"].max()) + 1, seed=AN.SEED + 1)
        ests = []
        for v, w_ in ((a, a["w"]), (b, wb)):
            n_u = int(v["u"].max()) + 1
            s, nu = ph1.unit_sums(v["pp"]["A"] - v["pp"]["ii"], v["u"], n_u)
            ests.append((w_ @ s) / (w_ @ nu))
        point = float((a["pp"]["A"] - a["pp"]["ii"]).mean() - (b["pp"]["A"] - b["pp"]["ii"]).mean())
        lo, hi = np.quantile(ests[0] - ests[1], [0.025, 0.975])
        res["R3_between_study"][f"{base}/{llm}"] = {"study1_minus_study2": point, "lo": float(lo), "hi": float(hi)}

    # R4 ChaosNLI judges and AURC
    cfg1 = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cnt1 = json.loads((ph1.S1 / "counts.json").read_text())
    ref1 = json.loads((ph1.S1 / "references.json").read_text())["refs"]["chaos"]
    sys.path.insert(0, str(ROOT / "src"))
    import eval_data as D
    items = {it["id"]: it for it in ph1.jl(ph1.S1 / "transform/chaos_inputs.jsonl")}
    res["R4_nli"] = {}
    qids = None
    for llm in ("qwen", "olmo"):
        tb = EA.Table(cfg1, ph1.S1, "chaos", "nli", "roberta", llm, cnt1["chaos/roberta"], ref1,
                      restrict=None if llm == "qwen" else qids)
        if llm == "qwen":
            qids = set(tb.ids)
        ids = tb.ids
        wr = [i for i in ids if ref1.get(i) is not None]
        d = {}
        for m in ("A", "ii", "iii"):
            pred = {i: EA.method_pred(tb.rows[m][i], "nli") for i in ids}
            d[m] = {"accuracy": float(np.mean([pred[i] == ref1[i] for i in wr])),
                    "label_share": {lab: float(np.mean([pred[i] == lab for i in ids])) for lab in EA.LAB}}
        d["reference_label_share"] = {lab: float(np.mean([ref1[i] == lab for i in wr])) for lab in EA.LAB}
        prim = {m: tb.scores(m, ids) for m in ("i", "A")}
        prim["i+A"] = (EA.rankdata(prim["i"]) + EA.rankdata(prim["A"])) / 2
        pos = {i: j for j, i in enumerate(ids)}
        sel = [pos[i] for i in wr]
        err = np.array([tb.yhat[i] != ref1[i] for i in wr], float)
        rc = RA.risk_coverage({m: np.asarray(v)[sel] for m, v in prim.items()}, err,
                              AN.unit_index([D.nli_unit(items[i]) for i in wr]))
        d["aurc"] = rc["aurc"]
        d["aurc_A_minus_i"] = rc.get("aurc_A_minus_i")
        res["R4_nli"][f"chaos/roberta/{llm}"] = d

    # R6 lemma overlap (FLAN) and DeBERTa accuracy on MCL dev vs test
    norm = lambda s: str(s).lower()
    res["R6"]["flan_stable_error_units"] = {"study1": len(units_by[("1", "flan")]), "study2": len(units_by[("2", "flan")]),
                                            "shared_lemmas": len({norm(x) for x in units_by[("1", "flan")]} & {norm(x) for x in units_by[("2", "flan")]})}
    ref2 = json.loads((ph1.S2 / "references.json").read_text())["refs"]["mcl"]
    b2 = EA.base_rows(ph1.S2, "mcl", "deberta", "orig")
    for split in ("dev", "test"):
        ii = [i for i in b2 if i.startswith(split + ".") and ref2.get(i) is not None]
        res["R6"][f"deberta_accuracy_mcl_{split}"] = {"n": len(ii), "accuracy": float(np.mean([b2[i]["pred"] == ref2[i] for i in ii]))}

    # R7 adjudication with the debiased (ii)
    adj = ROOT / "experiments/20261110-adjudication"
    key = json.loads((adj / "private/key.json").read_text(encoding="utf-8"))
    R = {r: json.loads((adj / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
    J = {"S": "T", "D": "F", "U": "U"}
    cls = {}
    for k, v in key.items():
        if v["role"] != "error":
            continue
        j = [J[R[r][k]] for r in R]
        cls[v["id"]] = "confirmed" if all(x == v["ref"] for x in j) else "contested"
    for study, data, base, llm, tb, stable, units in ph1.cells():
        if base != "deberta" or llm != "qwen":
            continue
        ids = tb.ids
        pos = {i: j for j, i in enumerate(ids)}
        prim = {m: tb.scores(m, ids) for m in PRIM}
        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        sdeb = debias_to(tb, ids, "ii", shareA)
        fl = {m: AN.q_mean_flags(prim[m]) for m in ("A", "ii")}
        fl["ii_deb"] = AN.q_mean_flags(sdeb)
        out = {}
        groups = {}
        for c in ("confirmed", "contested"):
            sid = [i for i in stable if i in pos and cls.get(i) == c]
            groups[c] = sid
            uu = AN.unit_index([units[i] for i in sid])
            ww = AN.boot_weights(int(uu.max()) + 1)
            xa = np.array([fl["A"][pos[i]] for i in sid])
            out[c] = {"n": len(sid), "share_yhat_different": float(np.mean([tb.yhat[i] == "F" for i in sid])),
                      "A_minus_ii": ci(AN.diff_ci(xa, np.array([fl["ii"][pos[i]] for i in sid]), uu, ww)),
                      "A_minus_ii_deb": ci(AN.diff_ci(xa, np.array([fl["ii_deb"][pos[i]] for i in sid]), uu, ww))}
        # confirmed - contested difference of A - (ii), joint unit bootstrap over all stable errors
        allid = groups["confirmed"] + groups["contested"]
        uu = AN.unit_index([units[i] for i in allid])
        ww = AN.boot_weights(int(uu.max()) + 1)
        n_u = int(uu.max()) + 1
        for m in ("ii", "ii_deb"):
            d = np.array([fl["A"][pos[i]] - fl[m][pos[i]] for i in allid])
            isc = np.array([i in set(groups["confirmed"]) for i in allid], float)
            sc, nc = ph1.unit_sums(d * isc, uu, n_u)
            so, no = ph1.unit_sums(d * (1 - isc), uu, n_u)
            nc = np.bincount(uu, weights=isc, minlength=n_u)
            no = np.bincount(uu, weights=1 - isc, minlength=n_u)
            est = (ww @ sc) / (ww @ nc) - (ww @ so) / (ww @ no)
            point = float(d[isc == 1].mean() - d[isc == 0].mean())
            lo, hi = np.quantile(est, [0.025, 0.975])
            out[f"confirmed_minus_contested_A_minus_{m}"] = {"d": point, "lo": float(lo), "hi": float(hi)}
        res["R7_adjudication"][f"study{study}:{data}/deberta/qwen"] = out

    (ROOT / "papers/bics/tables/posthoc_round3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    outcsv = ROOT / "papers/bics/data/item_outputs.csv"
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(item_rows[0]))
        wri.writeheader()
        wri.writerows(item_rows)
    print("->", ROOT / "papers/bics/tables/posthoc_round3.json", "and", outcsv, len(item_rows))


if __name__ == "__main__":
    main()
