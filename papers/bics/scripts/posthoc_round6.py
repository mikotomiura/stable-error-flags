"""Post-hoc descriptive analyses requested in attack-me round 5 (NOT preregistered; reported in S8 with [P]).

Reuses posthoc_round1.cells() / posthoc_round2.debias() and checks the frozen detection rates and A - control
intervals again (1e-9) before computing anything new.
  .venv-mac/bin/python -X utf8 papers/bics/scripts/posthoc_round6.py   (from the PJ root)
Analyses:
  P20 stability-restricted flags. Whether a prediction is stable (the same label on orig, swap and para1, the forms of
      set S) is observable without labels; stable errors can only be among stably predicted, transformable items.
      Each method keeps its flag budget (the top q of all ranked items) but spends it first on the stably predicted
      items, ranked by its own score; flags spill to the other items only if the budget exceeds them. Detection of
      stable errors for restricted (i) and restricted A; A_r - (i)_r and A - (i)_r (unit bootstrap).
  P21 coarsened A. P_A(same) recomputed from the stored sense distributions with the coarsening that built WiC test's
      reference [19]: two senses also count as the same if they are first-degree connections in the WordNet graph
      (any synset pointer, including sister senses, i.e. a shared direct hypernym) or share a supersense
      (lexicographer file). No labels are used. Detection of stable errors and A_c - (i), A_c - A, A_c - (ii),
      A_c - debiased (ii); the share of ranked items A_c predicts "different". The stored P_A(same) is first
      reproduced from p1, p2 (1e-9)."""
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


ph1 = load("ph1", ROOT / "papers/bics/scripts/posthoc_round1.py")
ph2 = load("ph2", ROOT / "papers/bics/scripts/posthoc_round2.py")
AN, EA, D = ph1.AN, ph1.EA, ph1.D
PRIM = ph1.PRIM
TOL = 1e-9
POINTERS = ("hypernyms", "hyponyms", "instance_hypernyms", "instance_hyponyms", "member_holonyms", "member_meronyms",
            "part_holonyms", "part_meronyms", "substance_holonyms", "substance_meronyms", "also_sees", "verb_groups",
            "similar_tos", "entailments", "causes", "attributes")


def ci(d):
    return {"d": float(d["d"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def related_fn(wn):
    cache = {}

    def neigh(s):
        if s.name() not in cache:
            cache[s.name()] = {x.name() for p in POINTERS for x in getattr(s, p)()}
        return cache[s.name()]

    def related(a, b):
        if a == b:
            return True
        sa, sb = wn.synset(a), wn.synset(b)
        if sa.lexname() == sb.lexname():
            return True
        if b in neigh(sa) or a in neigh(sb):
            return True
        return bool(set(sa.hypernyms() + sa.instance_hypernyms()) & set(sb.hypernyms() + sb.instance_hypernyms()))
    return related


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    wn = ph1.inventory.get_wn(str(ROOT / ".nltk-win"))
    related = related_fn(wn)
    res = {"note": "post hoc [P], not preregistered", "check_reproduces_frozen": {}, "check_p_same_reproduced": {},
           "P20_stability_restricted": {}, "P21_coarsened_A": {}}
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
        res["check_reproduces_frozen"][cell] = True
        out = ph1.S1 if study == "1" else ph1.S2
        cnt = tb.cnt

        # P20: label-free stable predictions among transformable items
        preds = {f: {i: r["pred"] for i, r in EA.base_rows(out, data, base, f).items()} for f in D.WIC_S}
        excl = set(cnt["non_transformable"]) | set(cnt["orig_skipped"])
        sp = np.array([i not in excl and all(i in preds[f] for f in D.WIC_S)
                       and all(preds[f][i] == preds["orig"][i] for f in D.WIC_S) for i in ids])
        assert all(sp[pos[i]] for i in st), "a stable error is not a stable prediction"
        fl_r = {}
        for m in ("i", "A"):
            s = np.asarray(prim[m], float)
            fl_r[m] = AN.q_mean_flags(np.where(sp, s, s - 10.0))
        xr = {m: np.array([fl_r[m][pos[i]] for i in st]) for m in fl_r}
        res["P20_stability_restricted"][cell] = {
            "n_ranked": len(ids), "n_stable_predictions": int(sp.sum()), "share_stable_predictions": float(sp.mean()),
            "detection": {"i": float(x["i"].mean()), "A": float(x["A"].mean()), "i_r": float(xr["i"].mean()),
                          "A_r": float(xr["A"].mean())},
            "A_r_minus_i_r": ci(AN.diff_ci(xr["A"], xr["i"], u, w)),
            "A_minus_i_r": ci(AN.diff_ci(x["A"], xr["i"], u, w)),
            "i_r_minus_i": ci(AN.diff_ci(xr["i"], x["i"], u, w))}

        # P21: coarsened A
        maxdev = 0.0
        pc = {}
        for i in ids:
            r = tb.rows["A"][i]
            K = len(r["synsets"])
            p1, p2 = r["p1"], r["p2"]
            maxdev = max(maxdev, abs(sum(p1[j] * p2[j] for j in range(K)) - r["p_same"]))
            pc[i] = sum(p1[a] * p2[b] for a in range(K) for b in range(K) if related(r["synsets"][a], r["synsets"][b]))
        assert maxdev < TOL, f"stored P_A(same) not reproduced: {maxdev}"
        res["check_p_same_reproduced"][cell] = maxdev
        sc = np.array([1 - (pc[i] if tb.yhat[i] == "T" else 1 - pc[i]) for i in ids])
        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        fl = {"A_c": AN.q_mean_flags(sc), "ii_deb": AN.q_mean_flags(ph2.debias(tb, ids, "ii", shareA)[0])}
        xc = {m: np.array([fl[m][pos[i]] for i in st]) for m in fl}
        res["P21_coarsened_A"][cell] = {
            "share_different_A": shareA, "share_different_A_c": float(np.mean([pc[i] < 0.5 for i in ids])),
            "share_items_changed_side": float(np.mean([(pc[i] >= 0.5) != (tb.rows["A"][i]["p_same"] >= 0.5) for i in ids])),
            "detection": {"A": float(x["A"].mean()), "A_c": float(xc["A_c"].mean()), "i": float(x["i"].mean()),
                          "ii": float(x["ii"].mean()), "ii_deb": float(xc["ii_deb"].mean())},
            "A_c_minus_i": ci(AN.diff_ci(xc["A_c"], x["i"], u, w)),
            "A_c_minus_A": ci(AN.diff_ci(xc["A_c"], x["A"], u, w)),
            "A_c_minus_ii": ci(AN.diff_ci(xc["A_c"], x["ii"], u, w)),
            "A_c_minus_ii_deb": ci(AN.diff_ci(xc["A_c"], xc["ii_deb"], u, w))}

    path = ROOT / "papers/bics/tables/posthoc_round6.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["check_reproduces_frozen"]), json.dumps(res["check_p_same_reproduced"]))
    print("->", path)


if __name__ == "__main__":
    main()
