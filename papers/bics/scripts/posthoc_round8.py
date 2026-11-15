"""Post-hoc descriptive analyses requested in attack-me round 7 (NOT preregistered; reported in S8 with [P]).

Reuses posthoc_round7 (restriction, frozen check) and checks the frozen detection rates and A - control intervals
again (1e-9) before computing anything new. No WordNet is read.
  .venv-mac/bin/python -X utf8 papers/bics/scripts/posthoc_round8.py   (from the PJ root)
Analyses:
  P25 adjudicated errors against the random expectation (Study 1, DeBERTa): on the stable errors where both raters
      chose the label opposite to the reference (68 with Qwen) and on those where both chose the reference
      (confirmed), each method's detection rate minus 0.30 (the q-mean of a random ranking), unit bootstrap.
  P26 the restricted comparisons of P22 at the same flag rates in every configuration: q = 0.1-0.3 (in Study 2 the
      budget exceeds the stably predicted items at q = 0.4 and 0.5, where every restricted difference is 0).
      A_r - random_r, (i)_r - random_r, A_r - (i)_r, A_r - (ii)_r, A_r - debiased (ii)_r, normalised detection,
      and the share ((ii)_r - (i)_r)/(A_r - (i)_r) (also with debiased (ii)) as in P2, for q = 0.1-0.5 and 0.1-0.3."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ph7 = load("ph7", ROOT / "papers/bics/scripts/posthoc_round7.py")
ph1, ph2, AN, EA = ph7.ph1, ph7.ph2, ph7.AN, ph7.EA
TOL = 1e-9
QS_ALL = AN.QS
QS_UNSAT = (0.10, 0.20, 0.30)


def ci(d, scale=1.0):
    return {"d": float(d["d"]) / scale, "lo": float(d["lo"]) / scale, "hi": float(d["hi"]) / scale}


def share_ci(xc, xi, xa, u, w):
    n_u = int(u.max()) + 1
    s_c, nu = ph1.unit_sums(xc - xi, u, n_u)
    s_a, _ = ph1.unit_sums(xa - xi, u, n_u)
    num, den = (w @ s_c) / (w @ nu), (w @ s_a) / (w @ nu)
    ok = den > 0
    lo, hi = np.quantile(num[ok] / den[ok], [0.025, 0.975])
    return {"share": float((xc - xi).mean() / (xa - xi).mean()), "lo": float(lo), "hi": float(hi),
            "denominator_le0_frac": float(1 - ok.mean())}


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    r7 = json.loads((ROOT / "papers/bics/tables/posthoc_round7.json").read_text())["P22_restricted_baselines"]
    adj = ROOT / "experiments/20261110-adjudication"
    key = json.loads((adj / "private/key.json").read_text(encoding="utf-8"))
    R = {r: json.loads((adj / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
    J = {"S": "T", "D": "F", "U": "U"}
    opposite = {"T": "F", "F": "T"}
    both_opp = {v["id"] for k, v in key.items() if v["role"] == "error" and all(J[R[r][k]] == opposite[v["ref"]] for r in R)}
    both_ref = {v["id"] for k, v in key.items() if v["role"] == "error" and all(J[R[r][k]] == v["ref"] for r in R)}
    res = {"note": "post hoc [P], not preregistered", "check_reproduces_frozen": {}, "P25_adjudicated_vs_random": {},
           "P26_restricted_same_rates": {}}
    for study, data, base, llm, tb, stable, units in ph1.cells():
        k = f"{data}/{base}/{llm}"
        ok, ids, pos, prim, st, u, w, x = ph7.check_frozen(tb, stable, units, frozen[study][k])
        assert ok, f"does not reproduce the frozen analysis: {study} {k}"
        cell = f"study{study}:{k}"
        res["check_reproduces_frozen"][cell] = True
        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        ii_deb = ph2.debias(tb, ids, "ii", shareA)[0]

        # P25
        if study == "1" and base == "deberta":
            fl = {m: AN.q_mean_flags(prim[m]) for m in ("A", "i", "ii")}
            fl["ii_deb"] = AN.q_mean_flags(ii_deb)
            out = {}
            for name, S in (("both_opposite", both_opp), ("confirmed", both_ref)):
                sid = [i for i in st if i in S]
                uu = AN.unit_index([units[i] for i in sid])
                ww = AN.boot_weights(int(uu.max()) + 1)
                xs = {m: np.array([fl[m][pos[i]] for i in sid]) for m in fl}
                rnd = np.full(len(sid), float(np.mean(QS_ALL)))
                out[name] = {"n": len(sid), "units": int(uu.max()) + 1,
                             "detection": {m: float(v.mean()) for m, v in xs.items()},
                             **{f"{m}_minus_random": ci(AN.diff_ci(xs[m], rnd, uu, ww)) for m in xs},
                             "A_minus_ii": ci(AN.diff_ci(xs["A"], xs["ii"], uu, ww)),
                             "A_minus_ii_deb": ci(AN.diff_ci(xs["A"], xs["ii_deb"], uu, ww))}
            if llm == "qwen":
                assert out["both_opposite"]["n"] == 68, out["both_opposite"]["n"]
            res["P25_adjudicated_vs_random"][cell] = out

        # P26
        out_dir = ph1.S1 if study == "1" else ph1.S2
        excl = set(tb.cnt["non_transformable"]) | set(tb.cnt["orig_skipped"])
        sp = ph7.stable_pred_mask(out_dir, data, base, ids, excl)
        assert int(sp.sum()) == r7[cell]["n_stable_predictions"]
        sti = [pos[i] for i in st]
        N, n, P = len(ids), len(st), int(sp.sum())
        scores = {"i": prim["i"], "A": prim["A"], "random": np.ones(len(ids))}
        if llm == "qwen":
            scores["ii"], scores["ii_deb"] = prim["ii"], ii_deb
        s_r = {m: ph7.restrict(s, sp) for m, s in scores.items()}
        blk = {}
        for tag, qs in (("q10_50", QS_ALL), ("q10_30", QS_UNSAT)):
            xr = {m: np.mean([AN.fractional_flags(s_r[m], q)[sti] for q in qs], axis=0) for m in s_r}
            if tag == "q10_50":
                for m in ("i", "A"):
                    assert abs(xr[m].mean() - r7[cell]["detection"][m]) < TOL
                assert abs(xr["random"].mean() - r7[cell]["random_r"]) < TOL
            ceil = float(np.mean([min(1.0, q * N / n) for q in qs]))
            rnd = float(xr["random"].mean())
            b = {"qs": list(qs), "random_r": rnd, "ceiling": ceil,
                 "detection": {m: float(v.mean()) for m, v in xr.items() if m != "random"},
                 "normalised_r": {m: (float(xr[m].mean()) - rnd) / (ceil - rnd) for m in xr if m != "random"},
                 "A_r_minus_random_r": ci(AN.diff_ci(xr["A"], xr["random"], u, w)),
                 "i_r_minus_random_r": ci(AN.diff_ci(xr["i"], xr["random"], u, w)),
                 "A_r_minus_i_r": ci(AN.diff_ci(xr["A"], xr["i"], u, w))}
            if "ii" in xr:
                b["A_r_minus_ii_r"] = ci(AN.diff_ci(xr["A"], xr["ii"], u, w))
                b["A_r_minus_ii_deb_r"] = ci(AN.diff_ci(xr["A"], xr["ii_deb"], u, w))
                b["share_ii_r"] = share_ci(xr["ii"], xr["i"], xr["A"], u, w)
                b["share_ii_deb_r"] = share_ci(xr["ii_deb"], xr["i"], xr["A"], u, w)
            blk[tag] = b
        blk["n_stable_predictions"], blk["n_ranked"], blk["budget_exceeds_at"] = P, N, [q for q in QS_ALL if q * N >= P]
        res["P26_restricted_same_rates"][cell] = blk

    path = ROOT / "papers/bics/tables/posthoc_round8.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["check_reproduces_frozen"]))
    print("->", path)


if __name__ == "__main__":
    main()
