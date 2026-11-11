"""Counting and analysis on the evaluation data (preregistration v4, Sec. 0, Steps 4 and 6; Secs. 1, 2, 5, 7-9).

  python eval_analysis.py count   --config ../config.json --out O --wic-zip W --chaos-zip C --grain G
  python eval_analysis.py analyze --config ../config.json --out O

count reads the references for the first time, finds the stable errors and their units, and applies the
abort rules of Sec. 9. It must run before the structured methods (Step 5). analyze builds the scores, the
flags and the decisions, and every report listed in Sec. 8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analysis as AN
import eval_data as D
import methods as M
from common import read_jsonl
from eval_run import GRAIN, WIC_ALL, WIC_DATA, build_forms, check_data_or_smoke, load_config

LAB = M.NLI_LABELS


# ---------- small helpers ----------

def rankdata(x):
    """Average ranks (1..n), ties sharing the mean rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def quantiles(x):
    x = np.asarray(x, float)
    return {k: float(np.quantile(x, q)) for k, q in (("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9))} if len(x) else None


def base_rows(out, data, name, form):
    p = Path(out) / "base" / f"{data}_{name}_{form}.jsonl"
    return {r["id"]: r for r in read_jsonl(p)} if p.exists() else {}


def enough(n_err, n_units, cfg):
    return n_err >= cfg["abort"]["min_errors"] and n_units >= cfg["abort"]["min_units"]


# ---------- Step 4: count ----------

def count_task(out, data, base, forms_S, ref, units, wn_excl, forms):
    """Stable and unstable errors of one base model on one data set (Secs. 1, 2, 4).

    forms: {id: {form name: ...}} from the transform outputs. An item is not transformable if a form of set S
    could not be built, or if the base model skipped it as too long (no truncation, R3). Items whose original
    form was skipped have no base prediction and leave every comparison (reported as orig_skipped)."""
    preds = {f: {i: r["pred"] for i, r in base_rows(out, data, base, f).items()} for f in forms_S}
    assert set(preds["orig"]) <= set(forms), "base outputs contain unknown ids"
    orig_skipped = sorted(set(forms) - set(preds["orig"]))
    ids = sorted(preds["orig"])
    non_transformable = {i for i in ids if any(f not in forms[i] or i not in preds[f] for f in forms_S)}
    for f in forms_S:  # missing forms are filled with the original prediction; such ids are excluded anyway
        for i in ids:
            preds[f].setdefault(i, preds["orig"][i])
    ref_all = {i: ref[i] for i in ids}
    ref_in = {i: ref[i] for i in ids if i not in wn_excl}
    stable, unstable = AN.classify_errors(preds, ref_in, excluded=non_transformable | set(wn_excl))
    st0, _ = AN.classify_errors(preds, ref_all, excluded=non_transformable)  # before the WordNet exclusion (Sec. 8)
    acc = np.mean([preds["orig"][i] == ref_in[i] for i in ref_in if ref_in[i] is not None])
    return {"stable": stable, "unstable": unstable, "n_stable": len(stable), "n_unstable": len(unstable),
            "units_stable": len({units[i] for i in stable}),
            "n_stable_before_wordnet": len(st0), "units_stable_before_wordnet": len({units[i] for i in st0}),
            "non_transformable": sorted(non_transformable), "orig_skipped": orig_skipped,
            "wordnet_excluded": sorted(wn_excl), "no_reference": sorted(i for i in ref_in if ref_in[i] is None),
            "accuracy_report_only": float(acc), "stable_share_of_errors": len(stable) / max(1, len(stable) + len(unstable))}


def count(cfg, a):
    out = Path(a.out)
    t = out / "transform"
    for task in ("wic", "nli"):  # Sec. 0: the paraphrase check (Step 2) comes before the references are read
        assert (out / "check" / f"{task}_result.json").exists(), "score the paraphrase check first (Sec. 0, Step 2)"
    check_data_or_smoke(cfg, a)
    wn_excl = json.loads((t / "wordnet_exclusions.json").read_text())
    nli_units = json.loads((t / "nli_units.json").read_text())
    forms = build_forms(t)
    res, refs = {}, {}
    for d in WIC_ALL:
        if d not in forms:
            continue
        gold = D.load_grain_gold(a.grain) if d == GRAIN else D.load_wic_gold(a.wic_zip, d)
        refs[d] = gold
        items = read_jsonl(t / f"{d}_inputs.jsonl")
        units = {it["id"]: D.wic_unit(it) for it in items}
        for base in (("deberta",) if d == GRAIN else ("deberta", "flan")):
            res[f"{d}/{base}"] = count_task(out, d, base, D.WIC_S, gold, units, wn_excl[d], forms[d])
        nt = set(res[f"{d}/deberta"]["non_transformable"])
        res[f"{d}/excluded_by_pos"] = {p: {"wordnet": sum(1 for it in items if it["pos"] == p and it["id"] in wn_excl[d]),
                                           "not_transformable": sum(1 for it in items if it["pos"] == p and it["id"] in nt)}
                                       for p in ("N", "V")}
    lab = D.load_chaos_labels(a.chaos_zip)
    refs["chaos"] = {i: v["ref"] for i, v in lab.items()}
    c = count_task(out, "chaos", "roberta", D.NLI_S, refs["chaos"], nli_units, [], forms["chaos"])
    med = float(np.median([v["max_frac"] for v in lab.values()]))
    high = {i for i, v in lab.items() if v["max_frac"] >= med}
    c["h4_median_max_frac"] = med
    c["h4_high_n"] = len(high)
    c["h4_high_share"] = len(high) / len(lab)
    c["h4_high_at_median"] = sum(1 for v in lab.values() if v["max_frac"] == med)
    c["h4_stable"] = [i for i in c["stable"] if i in high]
    c["h4_units_stable"] = len({nli_units[i] for i in c["h4_stable"]})
    res["chaos/roberta"] = c
    # Abort rules (Sec. 9), on counts only.
    T, Dv = res["wic_test/deberta"], res["wic_dev/deberta"]
    abort = {"aborted": [], "dev_dropped": False, "exploratory": []}
    if not enough(T["n_stable"], T["units_stable"], cfg):
        abort["aborted"] += ["H1", "H2"]
    if not enough(Dv["n_stable"], Dv["units_stable"], cfg):
        abort["dev_dropped"] = True
    if not enough(c["n_stable"], c["units_stable"], cfg):
        abort["aborted"] += ["H3", "H4"]
    elif not enough(len(c["h4_stable"]), c["h4_units_stable"], cfg):
        abort["exploratory"].append("H4")
    res["abort"] = abort
    (out / "counts.json").write_text(json.dumps(res, indent=1))
    (out / "references.json").write_text(json.dumps({"refs": refs, "chaos_labels": lab}))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_stable", "units_stable")} for k, v in res.items() if "n_stable" in v} | {"abort": abort}, indent=1))


# ---------- Step 6: scores ----------

def p_of(method_row, yhat, task):
    """P_X(yhat) from a method's output row."""
    if task == "wic":
        return method_row["p_same"] if yhat == "T" else 1 - method_row["p_same"]
    return method_row["probs"][LAB.index(yhat)]


def method_pred(row, task):
    if task == "wic":
        return "T" if row["p_same"] >= 0.5 else "F"
    return LAB[int(np.argmax(row["probs"]))]


class Table:
    """Scores of every method on one data set x base model, over the items kept for the comparison."""

    def __init__(self, cfg, out, data, task, base, model, cnt, ref, restrict=None):
        """restrict: the item ids of the primary table. A secondary structured model (OLMo) only drops its own
        parse failures from the primary item set (Sec. 5); it never adds items back."""
        self.task, self.data = task, data
        s = Path(out) / f"structured_{model}"
        sdata = "chaos" if task == "nli" else data
        self.rows = {m: {r["id"]: r for r in read_jsonl(s / f"{sdata}_{m}.jsonl")} for m in ("A", "ii", "iii", "iv")}
        b = base_rows(out, sdata, base, "orig")
        wn = set(cnt["wordnet_excluded"])
        self.ids_all = sorted(i for i in b if i not in wn and i in self.rows["A"] and (restrict is None or i in restrict))
        self.base = b
        self.yhat = {i: b[i]["pred"] for i in self.ids_all}
        self.ref = ref
        self.failed = {m: {i for i in self.ids_all if not self.rows[m][i]["parsed"]} for m in self.rows}
        self.fail_rate = {m: len(v) / len(self.ids_all) for m, v in self.failed.items()}
        self.ids = [i for i in self.ids_all if not any(i in v for v in self.failed.values())]
        self.cnt = cnt

    def p_base(self, i):
        r = self.base[i]
        return (r["p_same"] if self.yhat[i] == "T" else 1 - r["p_same"]) if self.task == "wic" else r["probs"][self.yhat[i]]

    def score(self, m, i):
        if m == "i":
            return 1 - self.p_base(i)
        return 1 - p_of(self.rows[m][i], self.yhat[i], self.task)

    def scores(self, m, ids, fail_value=None):
        out = []
        for i in ids:
            if fail_value is not None and m in self.failed and i in self.failed[m]:
                out.append(fail_value)
            else:
                out.append(self.score(m, i))
        return np.array(out)


def flags_over(scores_by_m, ids, qs=AN.QS):
    return {m: dict(zip(ids, AN.q_mean_flags(s, qs))) for m, s in scores_by_m.items()}


def secondary_scores(tb, cfg, out, ids):
    """Secondary controls of Sec. 5 (not used for decisions)."""
    s = {}
    A = tb.scores("A", ids)
    s["b7"] = np.array([1 - max(tb.rows["A"][i]["p_same"], 1 - tb.rows["A"][i]["p_same"]) if tb.task == "wic"
                        else 1 - max(tb.rows["A"][i]["probs"]) for i in ids])
    s["i+A"] = (rankdata(tb.scores("i", ids)) + rankdata(A)) / 2
    if tb.task == "wic":
        s["b-yhat"] = np.array([1.0 if tb.yhat[i] == "T" else 0.0 for i in ids])
        K = {i: len(tb.rows["A"][i]["p1"]) - 1 for i in ids}
        s["b-K"] = np.array([1 - 1 / K[i] if tb.yhat[i] == "T" else 1 / K[i] for i in ids])
    else:
        cbar, ebar = cfg["b_atoms"]["c_mean"], cfg["b_atoms"]["e_mean"]
        ntok = json.loads((Path(out) / "base" / "chaos_premise_tokens.json").read_text())
        mx = max(ntok[i] for i in ids) + 1
        for d in cfg["b_yhat_nli_directions"]:
            s[f"b-yhat[{d}]"] = np.array([1.0 if tb.yhat[i] == d else 0.0 for i in ids])
            s[f"b-len[{d}]"] = np.array([(1.0 if tb.yhat[i] == d else 0.0) + ntok[i] / mx for i in ids])
        s["b-atoms"] = np.array([1 - M.combine_atoms([cbar] * len(tb.rows["A"][i]["c"]), [ebar] * len(tb.rows["A"][i]["c"]))[LAB.index(tb.yhat[i])]
                                 for i in ids])
    return s


def b2_scores(tb, out):
    """(b2): share of the set-B forms on which the base prediction differs from yhat; None if no B form."""
    sdata = "chaos" if tb.task == "nli" else tb.data
    forms = D.WIC_B if tb.task == "wic" else D.NLI_B
    preds = [base_rows(out, sdata, tb.base_name, f) for f in forms]
    v = {}
    for i in tb.ids_all:
        got = [p[i]["pred"] for p in preds if i in p]
        v[i] = (sum(g != tb.yhat[i] for g in got) / len(got)) if len(got) == len(forms) else None
    return v


def compare(x, unit, a="A", controls=("i", "ii", "iii", "iv")):
    w = AN.boot_weights(int(np.max(unit)) + 1)
    return {c: AN.diff_ci(x[a], x[c], unit, w) for c in controls if c in x}


def build_x(tb, ids_err, flags):
    return {m: np.array([f[i] for i in ids_err]) for m, f in flags.items()}


def analyze_table(cfg, out, tb, stable, unstable, units, extra=None):
    """Primary flags and every reported comparison for one data set x base model x structured model."""
    ids = tb.ids
    pos = {i: j for j, i in enumerate(ids)}
    prim = {m: tb.scores(m, ids) for m in ("i", "A", "ii", "iii", "iv")}
    flags = flags_over(prim, ids)
    st = [i for i in stable if i in pos]
    un = [i for i in unstable if i in pos]
    u = AN.unit_index([units[i] for i in st]) if st else np.array([], int)
    x = build_x(tb, st, flags)
    rep = {"n_items": len(ids), "n_items_before_parse_exclusion": len(tb.ids_all), "fail_rate": tb.fail_rate,
           "n_stable": len(st), "units_stable": int(u.max()) + 1 if len(st) else 0, "n_unstable": len(un)}
    if not st:
        return rep, None, None
    rep["detection"] = {m: float(v.mean()) for m, v in x.items()}
    rep["diff_qmean"] = compare(x, u)
    rep["lo_above_0.05"] = {c: ci["lo"] > 0.05 for c, ci in rep["diff_qmean"].items()}
    # q = 25% and the per-q curve
    f25 = flags_over(prim, ids, (0.25,))
    rep["diff_q25"] = compare(build_x(tb, st, f25), u)
    sti = [pos[i] for i in st]
    rep["detection_by_q"] = {m: {str(q): float(AN.fractional_flags(prim[m], q)[sti].mean()) for q in AN.QS + (0.25,)} for m in prim}
    # Secondary controls
    sec = secondary_scores(tb, cfg, out, ids)
    fsec = flags_over(sec, ids)
    xs = dict(x, **build_x(tb, st, fsec))
    rep["diff_secondary"] = compare(xs, u, controls=tuple(sec))
    rep["diff_i_plus_A_vs_i"] = AN.diff_ci(xs["i+A"], x["i"], u)
    b2 = b2_scores(tb, out)
    valid = [i for i in ids if b2[i] is not None]
    fb2 = flags_over({"b2": np.array([b2[i] for i in valid])}, valid)["b2"]
    st_b2 = [i for i in st if i in fb2]
    if st_b2:
        ub = AN.unit_index([units[i] for i in st_b2])
        rep["diff_b2"] = AN.diff_ci(np.array([flags["A"][i] for i in st_b2]), np.array([fb2[i] for i in st_b2]), ub)
        rep["b2_n_stable"] = len(st_b2)
    # All errors (stable or not): detection over stable + unstable errors (non-transformable items and items
    # without a reference are not errors, Sec. 2). The precision counts, among the flags over all ranked items,
    # those on items whose base prediction differs from the reference (including non-transformable items).
    allerr = st + un
    ref = {i: tb.ref[i] for i in ids if tb.ref.get(i) is not None}
    err = set(i for i in ids if i in ref and tb.yhat[i] != ref[i])
    rep["all_errors"] = {}
    alli = [pos[i] for i in allerr]
    erri = [pos[i] for i in err]
    for m in prim:
        fl = {q: AN.fractional_flags(prim[m], q) for q in AN.QS + (0.25,)}
        rep["all_errors"][m] = {
            "detection_qmean": float(np.mean([flags[m][i] for i in allerr])) if allerr else None,
            "detection_by_q": {str(q): float(fl[q][alli].mean()) for q in fl} if allerr else None,
            "precision_by_q": {str(q): float(fl[q][erri].sum() / max(1e-12, fl[q].sum())) for q in fl}}
    if allerr:
        ua = AN.unit_index([units[i] for i in allerr])
        xa = build_x(tb, allerr, flags)
        rep["all_errors_A_minus_i"] = AN.diff_ci(xa["A"], xa["i"], ua)
    rep["p_yhat_stable"] = quantiles([tb.p_base(i) for i in st])
    rep["p_yhat_unstable"] = quantiles([tb.p_base(i) for i in un])
    # Strata by yhat
    rep["by_yhat"] = {}
    for y in sorted(set(tb.yhat[i] for i in ids)):
        sid = [i for i in st if tb.yhat[i] == y]
        lay = [i for i in ids if tb.yhat[i] == y]
        r = {"n_items": len(lay), "n_stable": len(sid)}
        if sid:
            r["detection"] = {m: float(np.mean([flags[m][i] for i in sid])) for m in prim}
            fl = flags_over({m: tb.scores(m, lay) for m in prim}, lay)
            ul = AN.unit_index([units[i] for i in sid])
            r["within_stratum_diff"] = compare(build_x(tb, sid, fl), ul)
        rep["by_yhat"][str(y)] = r
    # Discrepancy breakdown and error direction (base vs structured A), over items with a reference
    br = {"base_only": 0, "A_only": 0, "both": 0, "both_same_label": 0}
    for i in ref:
        bw, aw = tb.yhat[i] != ref[i], method_pred(tb.rows["A"][i], tb.task) != ref[i]
        br["base_only"] += bw and not aw
        br["A_only"] += aw and not bw
        br["both"] += bw and aw
        br["both_same_label"] += bw and aw and method_pred(tb.rows["A"][i], tb.task) == tb.yhat[i]
    rep["discrepancy"] = br
    # Overlap (Sec. 1, secondary): share of base errors on which the method predicts the same wrong label
    rep["overlap"] = {m: {k: (float(np.mean([method_pred(tb.rows[m][i], tb.task) == tb.yhat[i] for i in s])) if s else None)
                          for k, s in (("stable", st), ("unstable", un))} for m in ("ii", "A")}
    if extra:
        extra(rep, tb, ids, prim, flags, st, u, x)
    # Sensitivity analyses (Sec. 5): when a primary method fails on more than 5% of items
    # The failed items are kept with the failed method's score set to 1 or 0; the arrays are returned so that
    # the decision function can be applied to each version (analyze). Sensitivity never changes the decision.
    sens = None
    if any(v > cfg["sensitivity_fail_rate"] for v in tb.fail_rate.values()):
        rep["sensitivity"], sens = {}, {}
        all_pos = set(tb.ids_all)
        sst = [i for i in stable if i in all_pos]
        us = AN.unit_index([units[i] for i in sst])
        for val in (1.0, 0.0):
            sc = {m: tb.scores(m, tb.ids_all, fail_value=val) for m in prim}
            xv = build_x(tb, sst, flags_over(sc, tb.ids_all))
            rep["sensitivity"][f"fail_score_{int(val)}"] = compare(xv, us)
            sens[val] = (xv, us, sst)
    return rep, (x, u), sens


def wic_extra(rep, tb, ids, prim, flags, st, u, x):
    """Degeneracy diagnostics of (iv) and A (Sec. 8)."""
    for m in ("A", "iv"):
        rows = [tb.rows[m][i] for i in ids]
        K = [len(r["p1"]) - 1 for r in rows]
        rep[f"diag_{m}"] = {"p_same": quantiles([r["p_same"] for r in rows]),
                            "none_greedy_rate": float(np.mean([c for r in rows for c in r["choice_correct_none"]])),
                            "spearman_score_K": spearman(prim[m], K)}
    rep["iv_none_over_half"] = rep["diag_iv"]["none_greedy_rate"] > 0.5
    tol = [t for i in ids for t in (tb.rows["iv"][i].get("tolerance") or [])]
    rep["iv_replacement_tolerance"] = {str(k): tol.count(k) for k in sorted(set(map(str, tol)))} if tol else None


def grain_extra(rep, tb, ids, prim, flags, st, u, x):
    """GrainWiC coarse: the breakdown over the 20 words (Sec. 2) and the diagnostics of WiC."""
    wic_extra(rep, tb, ids, prim, flags, st, u, x)
    lem = {i: tb.lemma[i] for i in ids}
    rep["by_word"] = {w: {"n_stable": sum(lem[i] == w for i in st),
                          "detection": {m: (float(np.mean([flags[m][i] for i in st if lem[i] == w])) if any(lem[i] == w for i in st) else None)
                                        for m in prim}} for w in sorted(set(lem.values()))}


def nli_extra_factory(lab, units):
    def extra(rep, tb, ids, prim, flags, st, u, x):
        ent = [lab[i]["entropy"] for i in ids]
        rep["spearman_score_entropy"] = {m: spearman(prim[m], ent) for m in prim}
        mf = np.array([lab[i]["max_frac"] for i in st])
        if len(st) >= 4:
            cuts = np.quantile([lab[i]["max_frac"] for i in ids], [.25, .5, .75])
            qd = {}
            for k in range(4):
                lo = -np.inf if k == 0 else cuts[k - 1]
                hi = np.inf if k == 3 else cuts[k]
                sel = [j for j, v in enumerate(mf) if lo < v <= hi]
                qd[f"Q{k + 1}"] = {"n": len(sel), "A_minus": {c: float(np.mean(x["A"][sel] - x[c][sel])) if sel else None for c in ("i", "ii", "iii")}}
            rep["max_frac_quartiles"] = qd
        rep["by_part"] = {}
        for part in ("chaos_s", "chaos_m"):
            lay = [i for i in ids if i.startswith(part + "_")]
            sid = [i for i in st if i.startswith(part + "_")]
            if sid:
                fl = flags_over({m: tb.scores(m, lay) for m in prim}, lay)
                rep["by_part"][part] = {"n_stable": len(sid), "diff": compare(build_x(tb, sid, fl), AN.unit_index([units[i] for i in sid]))}
        rep["diff_A_minus_iv_report"] = rep["diff_qmean"].get("iv")
    return extra


def analyze(cfg, a):
    out = Path(a.out)
    cnt = json.loads((out / "counts.json").read_text())
    R = json.loads((out / "references.json").read_text())
    t = out / "transform"
    units_w = {d: {it["id"]: D.wic_unit(it) for it in read_jsonl(t / f"{d}_inputs.jsonl")} for d in WIC_ALL
               if (t / f"{d}_inputs.jsonl").exists()}
    units_n = json.loads((t / "nli_units.json").read_text())
    lab = R["chaos_labels"]
    check = {task: json.loads((out / "check" / f"{task}_result.json").read_text()) for task in ("wic", "nli")}
    report = {"counts": {k: {kk: v[kk] for kk in v if kk not in ("stable", "unstable", "h4_stable")} if isinstance(v, dict) else v
                         for k, v in cnt.items()}, "check": check}
    ab = cnt["abort"]
    aborted, exploratory = set(ab["aborted"]), set(ab["exploratory"])
    if check["wic"]["exploratory"]:
        exploratory |= {"H1", "H2"}
    if check["nli"]["exploratory"]:
        exploratory |= {"H3", "H4"}

    primary_ids = {}

    def table(data, task, base, model):
        key = f"{data}/{base}"
        tb = Table(cfg, out, data, task, base, model, cnt[key], R["refs"][data],
                   restrict=None if model == "qwen" else primary_ids.get((data, base)))
        tb.base_name = base
        if model == "qwen":
            primary_ids[(data, base)] = set(tb.ids)
        if task == "wic":
            tb.lemma = units_w[data]
        return tb, cnt[key]

    res = {}
    data_args = {}
    sens_args = {1.0: {}, 0.0: {}}

    def keep_sens(name, sens):
        for val in sens_args:
            if sens is not None:
                sens_args[val][name] = sens[val][:2]  # (x, unit); the id list is only needed for H4
    for model in ("qwen", "olmo"):
        if not (out / f"structured_{model}").exists():
            continue
        for d in WIC_DATA:
            for base in ("deberta", "flan"):
                if model == "olmo" and base == "flan":
                    continue
                tb, c = table(d, "wic", base, model)
                rep, xu, sens = analyze_table(cfg, out, tb, c["stable"], c["unstable"], units_w[d], wic_extra)
                res[f"{d}/{base}/{model}"] = rep
                if model == "qwen" and base == "deberta":
                    data_args[d] = xu
                    keep_sens(d, sens)
                    if not enough(rep["n_stable"], rep["units_stable"], cfg):  # Sec. 9: re-check after parse exclusions
                        if d == "wic_test":
                            exploratory |= {"H1", "H2"}
                        else:
                            ab["dev_dropped"] = True
        tb, c = table("chaos", "nli", "roberta", model)
        rep, xu, sens = analyze_table(cfg, out, tb, c["stable"], c["unstable"], units_n, nli_extra_factory(lab, units_n))
        res[f"chaos/roberta/{model}"] = rep
        if model == "qwen":
            data_args["chaos"] = xu
            if sens is not None:
                high_all = set(c["h4_stable"])
                for val, (xv, us, sst) in sens.items():
                    sens_args[val]["chaos"] = (xv, us)
                    sel = [j for j, i in enumerate(sst) if i in high_all]
                    if sel:
                        sens_args[val]["chaos_high"] = ({m: v[sel] for m, v in xv.items()}, AN.unit_index([units_n[sst[j]] for j in sel]))
            if not enough(rep["n_stable"], rep["units_stable"], cfg):
                exploratory |= {"H3", "H4"}
            high = set(c["h4_stable"])
            st = [i for i in c["stable"] if i in set(tb.ids)]
            sel = [j for j, i in enumerate(st) if i in high]
            if sel and xu is not None:
                xh = {m: v[sel] for m, v in xu[0].items()}
                uh = AN.unit_index([units_n[st[j]] for j in sel])
                data_args["chaos_high"] = (xh, uh)
                if not enough(len(sel), int(uh.max()) + 1, cfg):
                    exploratory.add("H4")
            elif "H4" not in aborted:
                exploratory.add("H4")  # Sec. 9: below the lower bound after the parse exclusions -> exploratory
            res["chaos_high_n_stable_after_parse"] = len(sel)
    # GrainWiC coarse (secondary): DeBERTa x Qwen only
    if "grain/deberta" in cnt and (out / "structured_qwen" / f"{GRAIN}_A.jsonl").exists():
        tb, c = table(GRAIN, "wic", "deberta", "qwen")
        res["grain/deberta/qwen"] = analyze_table(cfg, out, tb, c["stable"], c["unstable"], units_w[GRAIN], grain_extra)[0]
    test = None if "H1" in aborted or data_args.get("wic_test") is None else data_args["wic_test"]
    dev = None if ab["dev_dropped"] else data_args.get("wic_dev")
    nli = None if "H3" in aborted else data_args.get("chaos")
    nli_high = None if "H4" in aborted else data_args.get("chaos_high")
    for h, v in (("H1", test), ("H2", test), ("H3", nli)):
        if v is None:
            aborted.add(h)
    if nli_high is None and "H4" not in aborted:
        if nli is None:
            aborted.add("H4")
        else:  # H4 subset empty after exclusions: exploratory with nothing to compute (Sec. 9)
            exploratory.add("H4")
    decisions = AN.decide(test=test, dev=dev, nli=nli, nli_high=nli_high, aborted=aborted, exploratory=exploratory - aborted)
    report["decisions"] = decisions
    # Sensitivity (Sec. 5): the same decision function on each version; reported only, never changes the decision
    if any(sens_args[1.0].values()):
        report["sensitivity_decisions"] = {}
        for val, sa in sens_args.items():
            g = lambda k, fallback: sa.get(k, fallback)
            dsens = AN.decide(test=None if test is None else g("wic_test", test), dev=None if dev is None else g("wic_dev", dev),
                              nli=None if nli is None else g("chaos", nli), nli_high=None if nli_high is None else g("chaos_high", nli_high),
                              aborted=aborted, exploratory=exploratory - aborted)
            report["sensitivity_decisions"][f"fail_score_{int(val)}"] = {h: v["status"] for h, v in dsens.items()}
        report["sensitivity_changes_conclusion"] = any(
            report["sensitivity_decisions"][k][h] != decisions[h]["status"] for k in report["sensitivity_decisions"] for h in decisions)
    report["results"] = res
    report["reading_rules"] = reading_rules(res)
    (out / "analysis.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({h: v["status"] for h, v in decisions.items()}, indent=1))


def reading_rules(res):
    """Mechanical parts of the Sec. 8 reporting rules, so the wording does not depend on the researcher."""
    notes = []
    t = res.get("wic_test/deberta/qwen")
    if t and "all_errors_A_minus_i" in t and t["all_errors_A_minus_i"]["d"] <= 0:
        notes.append("H1 vs (i): includes the effect of restricting to stable errors (all-error A - (i) <= 0).")
    if t and "diff_secondary" in t and t["diff_secondary"].get("b-K", {}).get("lo", 1) <= 0:
        notes.append("H1: includes a part explained by the predicted label and the number of senses (A - (b-K) lower end <= 0).")
    if t and t.get("iv_none_over_half"):
        notes.append("H2: (iv) chose 'none of the above' in more than 50% of answers; H2 compares against a fake inventory that did not work as a judgement.")
    n = res.get("chaos/roberta/qwen")
    if n and "diff_secondary" in n:
        if n["diff_secondary"].get("b-atoms", {}).get("lo", 1) <= 0:
            notes.append("H3: includes a part explained by the predicted label and the number of atoms (A - (b-atoms) lower end <= 0).")
        if any(v["lo"] <= 0 for k, v in n["diff_secondary"].items() if k.startswith("b-len")):
            notes.append("H3: includes a part explained by the predicted label and the premise length (A - (b-len) lower end <= 0 for some direction).")
    if n and "by_part" in n and len(n["by_part"]) == 2:
        sgn = {p: tuple(np.sign(v["diff"][c]["d"]) for c in ("i", "ii", "iii")) for p, v in n["by_part"].items()}
        if sgn["chaos_s"] != sgn["chaos_m"]:
            notes.append("H3: the direction differs between ChaosNLI-S and -M; the claim needs a qualification.")
    for k, r in res.items():
        if not isinstance(r, dict) or "diff_qmean" not in r:
            continue
        for y, s in (r.get("by_yhat") or {}).items():
            for c, ci in (s.get("within_stratum_diff") or {}).items():
                main = r["diff_qmean"].get(c, {}).get("d")
                if c in ("i", "ii", "iii") and main is not None and np.sign(ci["d"]) != np.sign(main) and ci["d"] != 0:
                    notes.append(f"{k}: within the stratum yhat={y}, A - ({c}) has the opposite sign to the main result; report in the limitations.")
    g = res.get("grain/deberta/qwen")
    if g and "diff_qmean" in g and g["diff_qmean"]["ii"]["d"] < 0:
        notes.append("Limitations: on GrainWiC coarse, whose reference is not WordNet, no gain of A was seen (20 words).")
    return notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["count", "analyze"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wic-zip")
    ap.add_argument("--chaos-zip")
    ap.add_argument("--grain")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"count": count, "analyze": analyze}[a.stage](cfg, a)
