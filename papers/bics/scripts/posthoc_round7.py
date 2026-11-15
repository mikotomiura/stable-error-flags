"""Post-hoc descriptive analyses requested in attack-me round 6 (NOT preregistered; reported in S8 with [P]).

Reuses posthoc_round1.cells() / posthoc_round2.debias() / posthoc_round6.related_fn() and checks the frozen detection
rates and A - control intervals again (1e-9) before computing anything new; for coarse WiC it rebuilds the frozen
secondary cell (grain/deberta/qwen) the same way and checks it too.
  .venv-mac/bin/python -X utf8 papers/bics/scripts/posthoc_round7.py   (from the PJ root)
Analyses:
  P22 baselines for the stability-restricted flags of P20. Restricted random: a constant score on the stably
      predicted items, so that the flag rule of the frozen code (the top q of all ranked items, ties at the boundary
      split equally) gives each of them min(1, qN/P) and spills the rest evenly over the other items. Ceiling: a
      perfect ranking, q-mean of min(1, qN/n) (as P7). Normalised detection (d - random_r)/(ceiling - random_r).
      Per-q detection. Restricted (ii) and restricted debiased (ii) (shifted to A's share as in P6, then restricted);
      A_r minus each. Restricted (i) and A on all eligible errors (stable or not) and AURC over items with a
      reference (frozen risk_coverage). Study 2: the same restriction with stability measured on the paraphrase as
      formed, before the human check (items the check removed included).
  P23 coarsened A against the share shift. A_c shifted back to A's share of "different" (one log-odds constant,
      as P6; no labels) minus A; debiased (ii) shifted to A_c's share, A_c minus it.
  P24 coarse WiC (DeBERTa x Qwen; reference from CWSD-20, whose senses are Wikipedia pages, not WordNet senses):
      A - debiased (ii), A_c, A_c - A, A_c - debiased (ii) at A_c's share, A_c shifted back - A. 20 lemmas, 20 units."""
import hashlib
import importlib.util
import json
import os
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
ph6 = load("ph6", ROOT / "papers/bics/scripts/posthoc_round6.py")
AN, EA, D = ph1.AN, ph1.EA, ph1.D
import rep_analysis as RA  # noqa: E402  (src/ is on sys.path via posthoc_round1)
from eval_run import GRAIN  # noqa: E402

PRIM = ph1.PRIM
TOL = 1e-9
QS = AN.QS


def ci(d, scale=1.0):
    return {"d": float(d["d"]) / scale, "lo": float(d["lo"]) / scale, "hi": float(d["hi"]) / scale}


def shift_to_share(p, target_share_F):
    """Shift the log-odds of P(same) by one constant so that share(p' < 0.5) == target (as posthoc_round2.debias)."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    lo = np.log(p / (1 - p))
    srt = np.sort(lo)
    k = int(round(target_share_F * len(lo)))
    t = srt[0] - 1 if k == 0 else (srt[-1] + 1 if k >= len(lo) else (srt[k - 1] + srt[k]) / 2)
    return 1 / (1 + np.exp(-(lo - t)))


def score_from_psame(tb, ids, p):
    y = np.array([tb.yhat[i] == "T" for i in ids])
    return 1 - np.where(y, p, 1 - p)


def restrict(s, mask):
    s = np.asarray(s, float)
    return np.where(mask, s, s - 10.0)


def check_frozen(tb, stable, units, fr):
    ids = tb.ids
    pos = {i: j for j, i in enumerate(ids)}
    prim = {m: tb.scores(m, ids) for m in PRIM}
    st = [i for i in stable if i in pos]
    u = AN.unit_index([units[i] for i in st])
    w = AN.boot_weights(int(u.max()) + 1)
    flags = EA.flags_over(prim, ids)
    x = {m: np.array([flags[m][i] for i in st]) for m in PRIM}
    ok = len(st) == fr["n_stable"] and all(abs(x[m].mean() - fr["detection"][m]) < TOL for m in PRIM)
    for c in ("i", "ii", "iii", "iv"):
        d = AN.diff_ci(x["A"], x[c], u, w)
        ok &= all(abs(d[q] - fr["diff_qmean"][c][q]) < TOL for q in ("d", "lo", "hi"))
    return ok, ids, pos, prim, st, u, w, x


def stable_pred_mask(out, data, base, ids, excl):
    preds = {f: {i: r["pred"] for i, r in EA.base_rows(out, data, base, f).items()} for f in D.WIC_S}
    return np.array([i not in excl and all(i in preds[f] for f in D.WIC_S)
                     and all(preds[f][i] == preds["orig"][i] for f in D.WIC_S) for i in ids])


def restricted_block(tb, ids, pos, st, un, u, w, scores, mask, units, n_ranked):
    """Detection of stable errors (q-mean and per q), baselines, normalised values, all-error detection and AURC."""
    N, n, P = n_ranked, len(st), int(mask.sum())
    sti = [pos[i] for i in st]
    s_r = {m: restrict(scores[m], mask) for m in scores}
    s_r["random_r"] = np.where(mask, 1.0, 0.0)
    fl = {m: AN.q_mean_flags(s) for m, s in s_r.items()}
    xr = {m: fl[m][sti] for m in fl}
    by_q = {m: {str(q): float(AN.fractional_flags(s_r[m], q)[sti].mean()) for q in QS} for m in s_r}
    ceil_q = {str(q): float(min(1.0, q * N / n)) for q in QS}
    ceil = float(np.mean(list(ceil_q.values())))
    rand = float(xr["random_r"].mean())
    scale = ceil - rand
    blk = {"n_ranked": N, "n_stable_errors": n, "n_stable_predictions": P,
           "random_unrestricted": float(np.mean(QS)), "random_r": rand, "ceiling": ceil,
           "random_r_analytic": float(np.mean([min(1.0, q * N / P) for q in QS])),
           "detection": {m: float(v.mean()) for m, v in xr.items()},
           "detection_by_q": by_q, "ceiling_by_q": ceil_q,
           "normalised_r": {m: (float(xr[m].mean()) - rand) / scale for m in xr if m != "random_r"},
           "diff": {}}
    for a, b in (("A", "random_r"), ("i", "random_r"), ("A", "i"), ("A", "ii"), ("A", "ii_deb")):
        if a in xr and b in xr:
            d = AN.diff_ci(xr[a], xr[b], u, w)
            blk["diff"][f"{a}_r_minus_{b if b == 'random_r' else b + '_r'}"] = ci(d)
            blk["diff"][f"{a}_r_minus_{b if b == 'random_r' else b + '_r'}_normalised"] = ci(d, scale)
    # per-q differences A_r - i_r on the unsaturated rates (budget below the stably predicted items)
    unsat = [q for q in QS if q * N < P]
    blk["unsaturated_qs"] = unsat
    if unsat:
        xa = np.mean([AN.fractional_flags(s_r["A"], q)[sti] for q in unsat], axis=0)
        xi = np.mean([AN.fractional_flags(s_r["i"], q)[sti] for q in unsat], axis=0)
        xz = np.mean([AN.fractional_flags(s_r["random_r"], q)[sti] for q in unsat], axis=0)
        blk["unsaturated_mean"] = {"A_r": float(xa.mean()), "i_r": float(xi.mean()), "random_r": float(xz.mean()),
                                   "A_r_minus_i_r": ci(AN.diff_ci(xa, xi, u, w)),
                                   "A_r_minus_random_r": ci(AN.diff_ci(xa, xz, u, w)),
                                   "i_r_minus_random_r": ci(AN.diff_ci(xi, xz, u, w))}
    # all eligible errors (stable or unstable), as the frozen all_errors
    allerr = st + un
    if allerr:
        ai = [pos[i] for i in allerr]
        ua = AN.unit_index([units[i] for i in allerr])
        fl_all = {m: AN.q_mean_flags(s) for m, s in (("i", scores["i"]), ("A", scores["A"]),
                                                       ("i_r", s_r["i"]), ("A_r", s_r["A"]), ("random_r", s_r["random_r"]))}
        xa = {m: v[ai] for m, v in fl_all.items()}
        blk["all_errors"] = {"n": len(allerr), "detection": {m: float(v.mean()) for m, v in xa.items()},
                             "A_r_minus_i_r": ci(AN.diff_ci(xa["A_r"], xa["i_r"], ua)),
                             "A_r_minus_A": ci(AN.diff_ci(xa["A_r"], xa["A"], ua)),
                             "i_r_minus_i": ci(AN.diff_ci(xa["i_r"], xa["i"], ua))}
    # AURC over items with a reference (frozen risk_coverage: lemma resampling, flags fixed)
    wr = [i for i in ids if tb.ref.get(i) is not None]
    sel = [pos[i] for i in wr]
    err = np.array([tb.yhat[i] != tb.ref[i] for i in wr], float)
    uw = AN.unit_index([units[i] for i in wr])
    rc1 = RA.risk_coverage({"i": s_r["i"][sel], "A": s_r["A"][sel]}, err, uw)
    rc2 = RA.risk_coverage({"i": np.asarray(scores["A"], float)[sel], "A": s_r["A"][sel]}, err, uw)
    rc3 = RA.risk_coverage({"i": np.asarray(scores["i"], float)[sel], "A": s_r["i"][sel]}, err, uw)
    blk["aurc"] = {"i": rc3["aurc"]["i"], "A": rc2["aurc"]["i"], "i_r": rc1["aurc"]["i"], "A_r": rc1["aurc"]["A"],
                   "A_r_minus_i_r": rc1["aurc_A_minus_i"], "A_r_minus_A": rc2["aurc_A_minus_i"],
                   "i_r_minus_i": rc3["aurc_A_minus_i"], "note": "lower is better"}
    return blk


def p21_block(tb, ids, pos, st, u, w, x, related, prim):
    pc = {}
    maxdev = 0.0
    for i in ids:
        r = tb.rows["A"][i]
        K = len(r["synsets"])
        p1, p2 = r["p1"], r["p2"]
        maxdev = max(maxdev, abs(sum(p1[j] * p2[j] for j in range(K)) - r["p_same"]))
        pc[i] = sum(p1[a] * p2[b] for a in range(K) for b in range(K) if related(r["synsets"][a], r["synsets"][b]))
    assert maxdev < TOL, f"stored P_A(same) not reproduced: {maxdev}"
    pcv = np.array([pc[i] for i in ids])
    shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
    shareAc = float(np.mean(pcv < 0.5))
    pc_back = shift_to_share(pcv, shareA)
    sc = {"A_c": score_from_psame(tb, ids, pcv), "A_c_back": score_from_psame(tb, ids, pc_back),
          "ii_deb_A": ph2.debias(tb, ids, "ii", shareA)[0], "ii_deb_Ac": ph2.debias(tb, ids, "ii", shareAc)[0]}
    fl = {m: AN.q_mean_flags(s) for m, s in sc.items()}
    sti = [pos[i] for i in st]
    xc = {m: fl[m][sti] for m in fl}
    blk = {"check_p_same_reproduced": maxdev, "share_different_A": shareA, "share_different_A_c": shareAc,
           "share_different_A_c_back": float(np.mean(pc_back < 0.5)),
           "share_different_ii_deb_Ac": float(ph2.debias(tb, ids, "ii", shareAc)[1]),
           "detection": {"A": float(x["A"].mean()), "i": float(x["i"].mean()), "ii": float(x["ii"].mean()),
                         **{m: float(v.mean()) for m, v in xc.items()}},
           "A_minus_ii_deb_A": ci(AN.diff_ci(x["A"], xc["ii_deb_A"], u, w)),
           "A_c_minus_A": ci(AN.diff_ci(xc["A_c"], x["A"], u, w)),
           "A_c_back_minus_A": ci(AN.diff_ci(xc["A_c_back"], x["A"], u, w)),
           "A_c_minus_ii_deb_Ac": ci(AN.diff_ci(xc["A_c"], xc["ii_deb_Ac"], u, w)),
           "A_c_back_minus_ii_deb_A": ci(AN.diff_ci(xc["A_c_back"], xc["ii_deb_A"], u, w)),
           "A_c_minus_i": ci(AN.diff_ci(xc["A_c"], x["i"], u, w))}
    return blk


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    # BICS_NLTK_DIR: a local copy of .nltk-win (nltk's zip reads from Google Drive failed with a CRC error on
    # 2026-11-15); the copy must have the frozen SHA-256.
    nd = Path(os.environ.get("BICS_NLTK_DIR", ROOT / ".nltk-win"))
    frozen_sha = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["wordnet_zip_sha256"]
    assert hashlib.sha256((nd / "corpora/wordnet.zip").read_bytes()).hexdigest() == frozen_sha, "WordNet zip differs"
    wn = ph1.inventory.get_wn(str(nd))
    related = ph6.related_fn(wn)
    res = {"note": "post hoc [P], not preregistered", "check_reproduces_frozen": {},
           "P22_restricted_baselines": {}, "P22_study2_before_check": {}, "P23_share_shift": {}, "P24_coarse_wic": {}}
    for study, data, base, llm, tb, stable, units in ph1.cells():
        k = f"{data}/{base}/{llm}"
        ok, ids, pos, prim, st, u, w, x = check_frozen(tb, stable, units, frozen[study][k])
        assert ok, f"does not reproduce the frozen analysis: {study} {k}"
        cell = f"study{study}:{k}"
        res["check_reproduces_frozen"][cell] = True
        out = ph1.S1 if study == "1" else ph1.S2
        cnt = tb.cnt
        un = [i for i in cnt["unstable"] if i in pos]
        excl = set(cnt["non_transformable"]) | set(cnt["orig_skipped"])
        sp = stable_pred_mask(out, data, base, ids, excl)
        assert all(sp[pos[i]] for i in st), "a stable error is not a stable prediction"
        # same restriction as posthoc_round6 (P20): check it reproduces the stored restricted detection
        r6 = json.loads((ROOT / "papers/bics/tables/posthoc_round6.json").read_text())["P20_stability_restricted"][cell]
        assert int(sp.sum()) == r6["n_stable_predictions"]
        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        scores = {"i": prim["i"], "A": prim["A"]}
        if llm == "qwen":  # OLMo's (ii) is not a working judge (main text, Section 4)
            scores["ii"] = prim["ii"]
            scores["ii_deb"] = ph2.debias(tb, ids, "ii", shareA)[0]
        blk = restricted_block(tb, ids, pos, st, un, u, w, scores, sp, units, len(ids))
        assert abs(blk["detection"]["i"] - r6["detection"]["i_r"]) < TOL
        assert abs(blk["detection"]["A"] - r6["detection"]["A_r"]) < TOL
        res["P22_restricted_baselines"][cell] = blk
        if study == "2":
            excl_pre = set(cnt["orig_skipped"])
            sp_pre = stable_pred_mask(out, data, base, ids, excl_pre)
            assert all(sp_pre[pos[i]] for i in st)
            assert np.all(sp_pre >= sp), "restriction after the check is not a subset of the one before"
            res["P22_study2_before_check"][cell] = restricted_block(
                tb, ids, pos, st, un, u, w, {"i": prim["i"], "A": prim["A"]}, sp_pre, units, len(ids))
        res["P23_share_shift"][cell] = p21_block(tb, ids, pos, st, u, w, x, related, prim)

    # P24: coarse WiC, the frozen secondary cell grain/deberta/qwen
    cfg1 = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cnt1 = json.loads((ph1.S1 / "counts.json").read_text())
    ref1 = json.loads((ph1.S1 / "references.json").read_text())["refs"]
    units_g = {it["id"]: D.wic_unit(it) for it in ph1.jl(ph1.S1 / f"transform/{GRAIN}_inputs.jsonl")}
    c = cnt1[f"{GRAIN}/deberta"]
    tb = EA.Table(cfg1, ph1.S1, GRAIN, "wic", "deberta", "qwen", c, ref1[GRAIN], restrict=None)
    ok, ids, pos, prim, st, u, w, x = check_frozen(tb, c["stable"], units_g, frozen["1"][f"{GRAIN}/deberta/qwen"])
    assert ok, "does not reproduce the frozen coarse WiC cell"
    res["check_reproduces_frozen"][f"study1:{GRAIN}/deberta/qwen"] = True
    blk = p21_block(tb, ids, pos, st, u, w, x, related, prim)
    blk["A_minus_ii"] = ci(AN.diff_ci(x["A"], x["ii"], u, w))
    blk["n_stable"], blk["units"] = len(st), int(u.max()) + 1
    blk["share_different_ii"] = float(np.mean([EA.method_pred(tb.rows["ii"][i], "wic") == "F" for i in ids]))
    blk["share_gold_different"] = float(np.mean([tb.ref[i] == "F" for i in ids]))
    res["P24_coarse_wic"][f"study1:{GRAIN}/deberta/qwen"] = blk

    path = ROOT / "papers/bics/tables/posthoc_round7.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["check_reproduces_frozen"]))
    print("->", path)


if __name__ == "__main__":
    main()
