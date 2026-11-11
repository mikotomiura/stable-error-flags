"""Post-hoc descriptive analyses requested in attack-me round 2 (NOT preregistered; reported in S8 with [P]).

Reuses posthoc_round1.cells() (the frozen Table / flags of all seven WiC-type cells) and checks again that the frozen
detection rates and A - control intervals are reproduced to within 1e-9 before computing anything new.
  python -X utf8 papers/bics/scripts/posthoc_round2.py   (from the PJ root)
Analyses (all label-free on the method side; references are used only to evaluate, as in the frozen analysis):
  Q1 debiased controls: the unstructured judges (ii)/(iii) lean towards "different". Shift each judge's log-odds of
     "same" by one constant per cell so that the share of items it predicts as "different" equals A's share over the
     same ranked items (no labels used), rescore 1 - P'(yhat), and compare A with the debiased judges.
  Q2 ceiling-normalised detection: random ranking detects q on average (q-mean 0.30); a perfect ranking detects
     min(1, qN/n). Normalised detection = (det - 0.30) / (ceiling - 0.30); A - control on that scale, per cell and
     between studies (independent bootstraps per study as in P1).
  Q3 judge diagnostics: accuracy against the reference, share predicted "different", and score ties (share of ranked
     items whose score equals another item's score; ties at the flag boundaries) for A, (ii), (iii), (iv).
  Q4 combined ranks: (i)+(ii) - (i), (i)+A - (i)+(ii), (i)+A - A (mean of ranks, as the frozen (i)+A); AURC over
     q = 0, 0.05, ..., 0.95 for both studies (frozen risk_coverage).
  Q5 intervals for A - (ii) within yhat strata (P3), cluster bootstrap within the stratum.
  Q6 lemma overlap between the two studies' stable errors; adjudication baseline on stable-correct controls."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ph1", ROOT / "papers/bics/scripts/posthoc_round1.py")
ph1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph1)
AN, EA = ph1.AN, ph1.EA
sys.path.insert(0, str(ROOT / "src"))
import rep_analysis as RA  # noqa: E402

PRIM = ph1.PRIM
TOL = 1e-9
RANDOM = float(np.mean(AN.QS))


def ci(d):
    return {"d": float(d["d"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def debias(tb, ids, m, target_share_F):
    """Shift the log-odds of p_same so that share(p' < 0.5) == target; return scores 1 - P'(yhat)."""
    p = np.clip(np.array([tb.rows[m][i]["p_same"] for i in ids], float), 1e-12, 1 - 1e-12)
    lo = np.log(p / (1 - p))
    srt = np.sort(lo)
    k = int(round(target_share_F * len(lo)))
    t = srt[0] - 1 if k == 0 else (srt[-1] + 1 if k >= len(lo) else (srt[k - 1] + srt[k]) / 2)
    p2 = 1 / (1 + np.exp(-(lo - t)))
    y = np.array([tb.yhat[i] == "T" for i in ids])
    return 1 - np.where(y, p2, 1 - p2), float(np.mean(p2 < 0.5))


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    res = {"note": "post hoc [P], not preregistered", "random_detection": RANDOM,
           "Q1_debiased": {}, "Q2_normalised": {}, "Q3_judges": {}, "Q4_combined": {}, "Q5_yhat_strata": {},
           "Q6": {}}
    keep = {}
    for study, data, base, llm, tb, stable, units in ph1.cells():
        ids = tb.ids
        pos = {i: j for j, i in enumerate(ids)}
        prim = {m: tb.scores(m, ids) for m in PRIM}
        st = [i for i in stable if i in pos]
        u = AN.unit_index([units[i] for i in st])
        w = AN.boot_weights(int(u.max()) + 1)
        key = f"{data}/{base}/{llm}"
        fr = frozen[study][key]
        flags = EA.flags_over(prim, ids)
        x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
        ok = len(st) == fr["n_stable"] and all(abs(x[m].mean() - fr["detection"][m]) < TOL for m in PRIM)
        for c in ("i", "ii", "iii", "iv"):
            d = AN.diff_ci(x["A"], x[c], u, w)
            ok &= all(abs(d[k] - fr["diff_qmean"][c][k]) < TOL for k in ("d", "lo", "hi"))
        assert ok, f"does not reproduce the frozen analysis: {study} {key}"
        cell = f"study{study}:{key}"

        # Q1 debiased (ii)/(iii)
        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        ext = dict(prim)
        achieved = {}
        for m in ("ii", "iii"):
            ext[f"{m}_deb"], achieved[m] = debias(tb, ids, m, shareA)
        ext["i+A"] = (EA.rankdata(prim["i"]) + EA.rankdata(prim["A"])) / 2
        ext["i+ii"] = (EA.rankdata(prim["i"]) + EA.rankdata(prim["ii"])) / 2
        fl = EA.flags_over(ext, ids)
        xe = {m: np.array([fl[m][i] for i in st]) for m in ext}
        n_u = int(u.max()) + 1
        sA_i, nu = ph1.unit_sums(xe["A"] - xe["i"], u, n_u)
        q1 = {"target_share_different_(A)": shareA, "achieved_share_different": achieved,
              "detection": {m: float(xe[m].mean()) for m in ("A", "i", "ii", "iii", "ii_deb", "iii_deb")}}
        for m in ("ii_deb", "iii_deb"):
            q1[f"A_minus_{m}"] = ci(AN.diff_ci(xe["A"], xe[m], u, w))
            s_c, _ = ph1.unit_sums(xe[m] - xe["i"], u, n_u)
            num, den = (w @ s_c) / (w @ nu), (w @ sA_i) / (w @ nu)
            okd = den > 0
            lo, hi = np.quantile(num[okd] / den[okd], [0.025, 0.975])
            q1[f"share_{m}"] = {"share": float((xe[m] - xe["i"]).mean() / (xe["A"] - xe["i"]).mean()),
                                "lo": float(lo), "hi": float(hi), "denominator_le0_frac": float(1 - okd.mean())}
        res["Q1_debiased"][cell] = q1

        # Q2 ceiling-normalised
        N, n = len(ids), len(st)
        ceil = float(np.mean([min(1.0, q * N / n) for q in AN.QS]))
        scale = ceil - RANDOM
        q2 = {"ranked_items": N, "stable_errors": n, "prevalence": n / N, "ceiling": ceil,
              "normalised_detection": {m: (float(xe[m].mean()) - RANDOM) / scale for m in ("A", "i", "ii", "iii", "ii_deb")}}
        for c in ("i", "ii", "iii", "ii_deb"):
            d = AN.diff_ci(xe["A"], xe[c], u, w)
            q2[f"A_minus_{c}"] = {k: float(d[k]) / scale for k in ("d", "lo", "hi")}
        res["Q2_normalised"][cell] = q2
        keep[(study, base, llm)] = dict(xe=xe, u=u, w=w, scale=scale)

        # Q3 judge diagnostics
        wr = [i for i in ids if tb.ref.get(i) is not None]
        q3 = {}
        for m in ("A", "ii", "iii", "iv"):
            pred = {i: EA.method_pred(tb.rows[m][i], "wic") for i in ids}
            s = np.round(np.asarray(prim[m], float), 12)
            _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
            q3[m] = {"accuracy": float(np.mean([pred[i] == tb.ref[i] for i in wr])),
                     "share_predict_different": float(np.mean([pred[i] == "F" for i in ids])),
                     "share_items_with_tied_score": float(np.mean(cnt[inv] > 1)),
                     "ties_at_boundaries": RA.ties_at_boundaries(prim[m])}
        s = np.round(np.asarray(prim["i"], float), 12)
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        q3["i"] = {"share_items_with_tied_score": float(np.mean(cnt[inv] > 1)), "ties_at_boundaries": RA.ties_at_boundaries(prim["i"])}
        res["Q3_judges"][cell] = q3

        # Q4 combined ranks and AURC
        q4 = {"detection": {m: float(xe[m].mean()) for m in ("i", "A", "ii", "i+A", "i+ii")},
              "i+ii_minus_i": ci(AN.diff_ci(xe["i+ii"], xe["i"], u, w)),
              "i+A_minus_i+ii": ci(AN.diff_ci(xe["i+A"], xe["i+ii"], u, w)),
              "i+A_minus_A": ci(AN.diff_ci(xe["i+A"], xe["A"], u, w))}
        sel = [pos[i] for i in wr]
        err = np.array([tb.yhat[i] != tb.ref[i] for i in wr], float)
        rc = RA.risk_coverage({m: np.asarray(ext[m])[sel] for m in ("i", "A", "i+A", "ii", "i+ii")}, err,
                              AN.unit_index([units[i] for i in wr]))
        q4["aurc"] = rc["aurc"]
        q4["aurc_i+A_minus_i"] = rc.get("aurc_i+A_minus_i")
        q4["aurc_A_minus_i"] = rc.get("aurc_A_minus_i")
        res["Q4_combined"][cell] = q4

        # Q5 yhat strata intervals for A - (ii)
        q5 = {}
        for y in ("T", "F"):
            sid = [k for k, i in enumerate(st) if tb.yhat[i] == y]
            if len(sid) < 2:
                continue
            uu = AN.unit_index([units[st[k]] for k in sid])
            ww = AN.boot_weights(int(uu.max()) + 1)
            q5[f"yhat_{y}"] = {"n": len(sid), "units": int(uu.max()) + 1,
                               "A_minus_ii": ci(AN.diff_ci(xe["A"][sid], xe["ii"][sid], uu, ww)),
                               "A_minus_ii_deb": ci(AN.diff_ci(xe["A"][sid], xe["ii_deb"][sid], uu, ww))}
        res["Q5_yhat_strata"][cell] = q5
        if study == "1" and base == "deberta" and llm == "qwen":
            res["Q6"]["units_study1_stable"] = sorted({units[i] for i in st})
        if study == "2" and base == "deberta" and llm == "qwen":
            res["Q6"]["units_study2_stable"] = sorted({units[i] for i in st})

    # Q2 between-study differences on the normalised scale (as P1)
    res["Q2_between_study"] = {}
    for base, llm in (("deberta", "qwen"), ("flan", "qwen"), ("deberta", "olmo")):
        a, b = keep[("1", base, llm)], keep[("2", base, llm)]
        wb = AN.boot_weights(int(b["u"].max()) + 1, seed=AN.SEED + 1)
        out = {}
        for c in ("ii", "ii_deb"):
            ests = []
            for v, w_ in ((a, a["w"]), (b, wb)):
                n_u = int(v["u"].max()) + 1
                s, nu = ph1.unit_sums(v["xe"]["A"] - v["xe"][c], v["u"], n_u)
                ests.append((w_ @ s) / (w_ @ nu) / v["scale"])
            point = float((a["xe"]["A"] - a["xe"][c]).mean() / a["scale"] - (b["xe"]["A"] - b["xe"][c]).mean() / b["scale"])
            lo, hi = np.quantile(ests[0] - ests[1], [0.025, 0.975])
            out[c] = {"study1_minus_study2": point, "lo": float(lo), "hi": float(hi)}
        res["Q2_between_study"][f"{base}/{llm}"] = out

    # Q6 lemma overlap (unit labels) and adjudication baseline on controls
    u1, u2 = set(res["Q6"].pop("units_study1_stable")), set(res["Q6"].pop("units_study2_stable"))
    norm = lambda s: str(s).split("|")[0].split("#")[0].lower()
    res["Q6"]["stable_error_units"] = {"study1": len(u1), "study2": len(u2),
                                        "shared_lemmas": len({norm(x) for x in u1} & {norm(x) for x in u2})}
    adj = ROOT / "experiments/20261110-adjudication"
    key = json.loads((adj / "private/key.json").read_text(encoding="utf-8"))
    R = {r: json.loads((adj / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
    J = {"S": "T", "D": "F", "U": "U"}
    base_ctl = {}
    for st_ in ("1", "2"):
        for role in ("control", "error"):
            rows = [k for k, v in key.items() if v["study"] == st_ and v["role"] == role]
            both = sum(all(J[R[r][k]] == key[k]["ref"] for r in R) for k in rows)
            opp = sum(all(J[R[r][k]] not in (key[k]["ref"], "U") for r in R) for k in rows)
            base_ctl[f"study{st_}_{role}"] = {"n": len(rows), "both_agree_with_reference": both,
                                              "both_opposite": opp, "share_both_agree": both / len(rows)}
    res["Q6"]["adjudication_by_role"] = base_ctl
    out = ROOT / "papers/bics/tables/posthoc_round2.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
