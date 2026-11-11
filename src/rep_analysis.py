"""Counting and analysis of the replication study (rep_preregistration.md, Sec. 0, Steps 5 and 7).

  python rep_analysis.py count   --config ../rep_config.json --raw R --out O
  python rep_analysis.py analyze --config ../rep_config.json --out O [--v4-out <main evaluation results/eval>]

count reads the references for the first time (after the scored check), finds the stable and unstable errors
of DeBERTa and FLAN, and applies the abort rules. Items excluded by the paraphrase check lose their
paraphrase-1 form and so count as not transformable (kept in the flag ranking and in the precision denominator,
removed from the stable and unstable errors). analyze reuses the tables and comparisons of the main
evaluation (eval_analysis.Table / analyze_table) and applies the fixed sequence H1 -> H2 -> H1b:

- H1  (DeBERTa x Qwen): A beats (i), (ii) and (iii) (intersection-union, lower end of the 95% interval > 0).
- H2  (DeBERTa x Qwen): A beats (iv); confirmatory only if H1 is supported.
- H1b (FLAN x Qwen): A beats (i), (ii) and (iii); confirmatory only if H1 and H2 are supported.
OLMo (secondary) gives the same tables, reported only. The descriptive additions of Sec. 8 (main evaluation's
A as A_v4, per split, ties at the flag boundaries, risk-coverage, calibration, profile of the excluded items)
never enter a decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analysis as AN
import eval_data as D
import rep_data as R
from common import read_jsonl
from eval_analysis import (Table, analyze_table, base_rows, count_task, enough, flags_over, method_pred, p_of, quantiles,
                           reading_rules, rankdata, wic_extra)
from eval_run import load_config
from rep_run import DATA, check_raw, forms_of

BASES = ("deberta", "flan")
SEQUENCE = ("H1", "H2", "H1b")
COVERAGE_QS = tuple(round(0.05 * k, 2) for k in range(0, 20))  # abstention rates 0, 0.05, ..., 0.95 for AURC


def count(cfg, a):
    out = Path(a.out)
    t = out / "transform"
    chk = out / "check" / "result.json"
    assert chk.exists(), "score the paraphrase check first (Sec. 0, Step 3)"
    check_raw(cfg, a.raw)
    excluded = set(json.loads(chk.read_text())["excluded"])
    forms = forms_of(t)
    for i in excluded:
        forms[i].pop("para1", None)
    items = read_jsonl(t / f"{DATA}_inputs.jsonl")
    gold = R.load_gold(a.raw)
    ref = {it["id"]: gold[it["id"]] for it in items}
    units = {it["id"]: R.unit(it) for it in items}
    wn_excl = list(json.loads((t / "wordnet_exclusions.json").read_text())[DATA])
    res = {f"{DATA}/{b}": count_task(out, DATA, b, D.WIC_S, ref, units, wn_excl, forms) for b in BASES}
    for b in BASES:
        res[f"{DATA}/{b}"]["check_excluded"] = len(excluded)
    split_lemmas = {}
    for it in items:
        split_lemmas.setdefault(it["id"].split(".")[0], set()).add(R.unit(it))
    res["units_in_both_splits"] = len(split_lemmas.get("dev", set()) & split_lemmas.get("test", set()))
    Dn, Fl = res[f"{DATA}/deberta"], res[f"{DATA}/flan"]
    abort = {"aborted": [], "exploratory": []}
    if not enough(Dn["n_stable"], Dn["units_stable"], cfg):
        abort["aborted"] += list(SEQUENCE)
    elif not enough(Fl["n_stable"], Fl["units_stable"], cfg):
        abort["aborted"].append("H1b")
    res["abort"] = abort
    (out / "counts.json").write_text(json.dumps(res, indent=1))
    (out / "references.json").write_text(json.dumps({"refs": {DATA: ref}}))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_stable", "units_stable", "n_unstable")} for k, v in res.items() if isinstance(v, dict) and "n_stable" in v}
                     | {"abort": abort}, indent=1))


def rep_reading_rules(res):
    """The WiC rules of eval_analysis.reading_rules (Sec. 8), applied to H1/H2 (DeBERTa) and H1b (FLAN) on mcl,
    plus its generic within-stratum rule. The ChaosNLI and GrainWiC rules do not apply."""
    notes = []
    for b in BASES:
        t = res.get(f"{DATA}/{b}/qwen")
        if not t:
            continue
        for line in reading_rules({"wic_test/deberta/qwen": t}):
            if b == "deberta" and line.startswith(("H1", "H2")):
                notes.append(line)
            elif b == "flan" and line.startswith("H1"):
                notes.append(line.replace("H1", "H1b", 1))
    notes += [line for line in reading_rules(res) if line.startswith(f"{DATA}/")]
    return notes


def decide(h1, h2, h1b, aborted, exploratory):
    """Fixed sequence H1 -> H2 -> H1b. h1, h1b: (x, unit) for the IUT; h2: (x, unit) for A vs (iv); or None."""
    out, confirm = {}, True

    def put(h, res, passed):
        nonlocal confirm
        if h in aborted:
            res.update(status=AN.ABORTED, confirmatory=False)
            confirm = False
        elif h in exploratory or not confirm:
            res.update(status=AN.EXPLORATORY, confirmatory=False)
            confirm = False
        else:
            res.update(status=AN.SUPPORTED if passed else AN.REJECTED, confirmatory=True)
            if not passed:
                for c in res.get("failed", []):
                    res.setdefault("negative_reading", {})[c] = AN.negative_reading(res["ci"][c])
                confirm = False
        out[h] = res

    def iut_step(h, arg):
        if arg is None:
            put(h, {}, False)
        else:
            r = AN.iut(*arg)
            put(h, r, r["pass"])

    iut_step("H1", h1)
    if h2 is None:
        put("H2", {}, False)
    else:
        x, u = h2
        ci = AN.diff_ci(x["A"], x["iv"], u)
        r = {"ci": {"iv": ci}, "pass": ci["lo"] > 0, "failed": [] if ci["lo"] > 0 else ["iv"],
             "report_iv_minus_ii": AN.diff_ci(x["iv"], x["ii"], u)}
        if AN.diff_ci(x["A"], x["ii"], u)["lo"] > 0:
            r["content_share"] = AN.content_share(x, u)
        put("H2", r, r["pass"])
    iut_step("H1b", h1b)
    return out


# ---------- descriptive additions (Sec. 8; never used for a decision) ----------

def ties_at_boundaries(s, qs=AN.QS):
    """Share of the ranked items whose score equals the score at the flag boundary of each q."""
    s = np.round(np.asarray(s, float), 12)
    srt = np.sort(s)[::-1]
    out = {}
    for q in qs:
        k = int(round(q * len(s)))
        out[str(q)] = float(np.mean(s == srt[k - 1])) if k else 0.0
    return out


def risk_coverage(scores_by_m, err, unit, n_boot=AN.N_BOOT):
    """Selective risk at q = 10..50% and AURC (mean risk over q = 0, 0.05, ..., 0.95) per method, and the
    differences (i)+A - (i) and A - (i) with a lemma-resampling 95% interval (flags fixed, as in diff_ci)."""
    unit = np.asarray(unit)
    W = AN.boot_weights(int(unit.max()) + 1, n_boot=n_boot)[:, unit]
    res = {"risk_at_q": {}, "aurc": {}}
    curves = {}
    for m, s in scores_by_m.items():
        keep = np.array([1 - AN.fractional_flags(s, q) for q in COVERAGE_QS])  # (n_q, n)
        point = (keep @ err) / keep.sum(axis=1)
        boot = (W @ (keep * err).T) / (W @ keep.T)  # (n_boot, n_q)
        curves[m] = (point, boot)
        res["risk_at_q"][m] = {str(q): float(curves[m][0][COVERAGE_QS.index(q)]) for q in AN.QS}
        res["aurc"][m] = float(curves[m][0].mean())
    for a, b in (("i+A", "i"), ("A", "i")):
        if a in curves and b in curves:
            d = curves[a][1].mean(axis=1) - curves[b][1].mean(axis=1)
            lo, hi = np.quantile(d, [0.025, 0.975])
            res[f"aurc_{a}_minus_{b}"] = {"d": res["aurc"][a] - res["aurc"][b], "lo": float(lo), "hi": float(hi)}
            ix = [COVERAGE_QS.index(q) for q in AN.QS]
            dq = curves[a][1][:, ix].mean(axis=1) - curves[b][1][:, ix].mean(axis=1)
            lo, hi = np.quantile(dq, [0.025, 0.975])
            res[f"risk_q10_50_{a}_minus_{b}"] = {"d": float(np.mean(curves[a][0][ix] - curves[b][0][ix])), "lo": float(lo), "hi": float(hi)}
    return res


def ece(p_pred, correct, bins=10):
    """Expected calibration error of the base model's probability of its own prediction (equal-width bins)."""
    p, c = np.asarray(p_pred, float), np.asarray(correct, float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == k].mean() - c[idx == k].mean()) * np.mean(idx == k) for k in range(bins) if np.any(idx == k)))


def base_profile(rows, ref, ids):
    ids = [i for i in ids if i in rows and ref.get(i) is not None]
    p = [rows[i]["p_same"] if rows[i]["pred"] == "T" else 1 - rows[i]["p_same"] for i in ids]
    ok = [rows[i]["pred"] == ref[i] for i in ids]
    return {"n": len(ids), "accuracy": float(np.mean(ok)), "ece": ece(p, ok), "share_pred_T": float(np.mean([rows[i]["pred"] == "T" for i in ids])),
            "p_pred": quantiles(p)}


def describe(cfg, out, tb, b, prim, flags, st, units, ref, excluded, on_sheet, v4_out=None):
    ids = tb.ids
    rep = {"scope": {"ranked_items": len(ids), "note": "items with >= 2 WordNet senses on which every primary method parsed"}}
    wr0 = [i for i in ids if ref.get(i) is not None]
    rep["accuracy_on_mcl"] = {"base": float(np.mean([tb.yhat[i] == ref[i] for i in wr0])),
                              **{m: float(np.mean([method_pred(tb.rows[m][i], "wic") == ref[i] for i in wr0])) for m in ("A", "ii", "iii", "iv")}}
    s = dict(prim)
    s["i+A"] = (rankdata(prim["i"]) + rankdata(prim["A"])) / 2
    v4 = Path(out) / "structured_qwen" / f"{DATA}_A_v4.jsonl"
    if tb.model_name == "qwen" and v4.exists():
        rows = {r["id"]: r for r in read_jsonl(v4)}
        ok = [i for i in ids if i in rows and rows[i]["parsed"]]
        rep["A_v4"] = {"n_items": len(ok), "parse_rate": len(ok) / max(1, len(ids))}
        pos = {i: j for j, i in enumerate(ids)}
        sv = {m: np.asarray(v)[[pos[i] for i in ok]] for m, v in prim.items()}
        sv["A_v4"] = np.array([1 - p_of(rows[i], tb.yhat[i], "wic") for i in ok])
        fv = flags_over(sv, ok)
        sto = [i for i in st if i in set(ok)]
        if sto:
            u = AN.unit_index([units[i] for i in sto])
            x = {m: np.array([f[i] for i in sto]) for m, f in fv.items()}
            w = AN.boot_weights(int(u.max()) + 1)
            rep["A_v4"]["n_stable"] = len(sto)
            rep["A_v4"]["detection"] = {m: float(v.mean()) for m, v in x.items()}
            rep["A_v4"]["A_v4_minus"] = {c: AN.diff_ci(x["A_v4"], x[c], u, w) for c in ("i", "ii", "iii")}
            rep["A_v4"]["A_minus_A_v4"] = AN.diff_ci(x["A"], x["A_v4"], u, w)
        rep["A_v4"]["p_same"] = quantiles([rows[i]["p_same"] for i in ok])
        rep["ties_A_v4"] = ties_at_boundaries(sv["A_v4"]) if ok else None
    rep["ties_at_boundaries"] = {m: ties_at_boundaries(s[m]) for m in ("i", "A", "iv", "i+A")}
    # Per split (dev / test): the flags are computed over all items; only the stable errors are split.
    rep["by_split"] = {}
    for sp in ("dev", "test"):
        sid = [i for i in st if i.split(".")[0] == sp]
        if sid:
            u = AN.unit_index([units[i] for i in sid])
            x = {m: np.array([flags[m][i] for i in sid]) for m in flags}
            w = AN.boot_weights(int(u.max()) + 1)
            rep["by_split"][sp] = {"n_stable": len(sid), "detection": {m: float(v.mean()) for m, v in x.items()},
                                   "A_minus": {c: AN.diff_ci(x["A"], x[c], u, w) for c in ("i", "ii", "iii", "iv")}}
    # Risk-coverage over every ranked item with a reference
    wr = [i for i in ids if ref.get(i) is not None]
    pos = {i: j for j, i in enumerate(ids)}
    sel = [pos[i] for i in wr]
    err = np.array([tb.yhat[i] != ref[i] for i in wr], float)
    rep["risk_coverage"] = risk_coverage({m: np.asarray(s[m])[sel] for m in ("i", "A", "i+A")}, err, AN.unit_index([units[i] for i in wr]))
    # Base model on mcl (and on the main evaluation's WiC test, when given)
    rep["base_on_mcl"] = base_profile(tb.base, ref, ids)
    if v4_out:
        r4 = json.loads((Path(v4_out) / "references.json").read_text())["refs"]["wic_test"]
        b4 = base_rows(v4_out, "wic_test", b, "orig")
        rep["base_on_v4_wic_test"] = base_profile(b4, r4, sorted(b4))
    # Items excluded by the check vs kept (among the ranked items)
    K = {i: len(tb.rows["A"][i]["p1"]) - 1 for i in ids}
    prof = {}
    for name, grp in (("excluded_by_check", [i for i in ids if i in excluded]),
                      ("passed_check", [i for i in ids if i in on_sheet and i not in excluded]),
                      ("not_on_sheet", [i for i in ids if i not in on_sheet])):
        g = [i for i in grp if ref.get(i) is not None]
        prof[name] = {"n": len(grp), "base_error_rate": float(np.mean([tb.yhat[i] != ref[i] for i in g])) if g else None,
                      "share_noun": float(np.mean([i in tb.pos_noun for i in grp])) if grp else None,
                      "K": quantiles([K[i] for i in grp]) if grp else None}
    rep["check_exclusion_profile"] = prof
    return rep


def analyze(cfg, a):
    out = Path(a.out)
    cnt = json.loads((out / "counts.json").read_text())
    ref = json.loads((out / "references.json").read_text())["refs"][DATA]
    chk = json.loads((out / "check" / "result.json").read_text())
    excluded = set(chk["excluded"])
    on_sheet = {i for i, fs in forms_of(out / "transform").items() if "para1" in fs}
    items = read_jsonl(out / "transform" / f"{DATA}_inputs.jsonl")
    units = {it["id"]: R.unit(it) for it in items}
    nouns = {it["id"] for it in items if it["pos"] == "N"}
    aborted, exploratory = set(cnt["abort"]["aborted"]), set(cnt["abort"]["exploratory"])
    if chk["exploratory"]:
        exploratory |= set(SEQUENCE)
    res, args, sens_args, primary_ids, desc = {}, {}, {1.0: {}, 0.0: {}}, {}, {}
    for model in ("qwen", "olmo"):
        if not (out / f"structured_{model}").exists():
            continue
        for b in BASES:
            c = cnt[f"{DATA}/{b}"]
            tb = Table(cfg, out, DATA, "wic", b, model, c, ref, restrict=None if model == "qwen" else primary_ids.get(b))
            tb.base_name, tb.model_name, tb.pos_noun = b, model, nouns
            if model == "qwen":
                primary_ids[b] = set(tb.ids)
            captured = {}

            def extra(rep, tb_, ids, prim, flags, st, u, x):
                wic_extra(rep, tb_, ids, prim, flags, st, u, x)
                captured.update(prim=prim, flags=flags, st=st)

            rep, xu, sens = analyze_table(cfg, out, tb, c["stable"], c["unstable"], units, extra)
            res[f"{DATA}/{b}/{model}"] = rep
            if captured:
                desc[f"{DATA}/{b}/{model}"] = describe(cfg, out, tb, b, captured["prim"], captured["flags"], captured["st"], units, ref,
                                                       excluded, on_sheet, getattr(a, "v4_out", None))
            if model == "qwen":
                args[b] = xu
                if sens is not None:
                    for val in sens:
                        sens_args[val][b] = sens[val][:2]
                if not enough(rep["n_stable"], rep["units_stable"], cfg):  # Sec. 9: re-check after parse exclusions
                    exploratory |= set(SEQUENCE) if b == "deberta" else {"H1b"}
    for h, v in (("H1", args.get("deberta")), ("H2", args.get("deberta")), ("H1b", args.get("flan"))):
        if v is None and h not in exploratory:
            aborted.add(h)
    dec = decide(args.get("deberta"), args.get("deberta"), args.get("flan"), aborted, exploratory - aborted)
    report = {"counts": {k: {kk: v[kk] for kk in v if kk not in ("stable", "unstable")} if isinstance(v, dict) else v for k, v in cnt.items()},
              "check": {k: v for k, v in chk.items() if k != "excluded"}, "decisions": dec}
    if any(sens_args[1.0].values()):
        report["sensitivity_decisions"] = {}
        for val, sa in sens_args.items():
            g = lambda b: sa.get(b, args.get(b))
            ds = decide(g("deberta"), g("deberta"), g("flan"), aborted, exploratory - aborted)
            report["sensitivity_decisions"][f"fail_score_{int(val)}"] = {h: v["status"] for h, v in ds.items()}
        report["sensitivity_changes_conclusion"] = any(report["sensitivity_decisions"][k][h] != dec[h]["status"]
                                                       for k in report["sensitivity_decisions"] for h in dec)
    report["results"] = res
    report["descriptive"] = desc
    report["reading_rules"] = rep_reading_rules(res)
    (out / "analysis.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({h: v["status"] for h, v in dec.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["count", "analyze"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw")
    ap.add_argument("--v4-out", help="analyze only: the main evaluation's results/eval, for the base-model comparison")
    a = ap.parse_args()
    {"count": count, "analyze": analyze}[a.stage](load_config(a.config), a)
