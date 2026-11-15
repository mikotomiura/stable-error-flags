"""Post-hoc descriptive analyses on items deferred in attack-me rounds 3-4 (NOT preregistered; reported in S8 with [P]).

Two analyses that the internal review had left optional:
  H1 hidden errors (round 2-4, R1 #6): among the stable-correct controls of the adjudication sheet (DeBERTa), the items
     on which both raters chose the label opposite to the reference. If the reference is wrong there, the classifier is
     wrong too and a flag is a hit. Flag rates of A, (i), (ii), (iii) and debiased (ii) on these items and on the
     controls where both raters chose the reference; A - (ii) and A - debiased (ii) on the hidden errors (unit bootstrap).
     The shared-inventory explanation predicts A - (ii) < 0 here (A siding with the WordNet-derived reference), the
     mirror image of P17 (A - (ii) > 0 on stable errors that both raters reversed).
  H2 contested errors by reference label x WordNet origin (round 3, R2 N5; Study 1 only, where P4's origin is defined):
     n and flag rates of A, (ii), debiased (ii); A - (ii) and A - debiased (ii) per cell (unit bootstrap). Cells are small.
Reuses posthoc_round1.cells() / posthoc_round2.debias() and checks the frozen detection rates and A - control
intervals again (1e-9) before computing anything new.
  .venv-mac/bin/python -X utf8 papers/bics/scripts/posthoc_round5.py   (from the PJ root)"""
import importlib.util
import json
import re
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
AN, EA = ph1.AN, ph1.EA
PRIM = ph1.PRIM
TOL = 1e-9
J = {"S": "T", "D": "F", "U": "U"}
OPP = {"T": "F", "F": "T"}


def ci(d):
    return {"d": float(d["d"]), "lo": float(d["lo"]), "hi": float(d["hi"])}


def summarise(sid, fl, pos, units):
    """Flag rates on the items sid and A minus (ii) / debiased (ii) with the unit bootstrap."""
    out = {"n": len(sid)}
    if not sid:
        return out
    uu = AN.unit_index([units[i] for i in sid])
    out["units"] = int(uu.max()) + 1
    xs = {m: np.array([fl[m][pos[i]] for i in sid]) for m in fl}
    out["flag_rate"] = {m: float(v.mean()) for m, v in xs.items()}
    if len(sid) >= 2 and out["units"] >= 2:
        ww = AN.boot_weights(out["units"])
        out["A_minus_ii"] = ci(AN.diff_ci(xs["A"], xs["ii"], uu, ww))
        out["A_minus_ii_deb"] = ci(AN.diff_ci(xs["A"], xs["ii_deb"], uu, ww))
    return out


def main():
    frozen = {"1": json.loads((ph1.S1 / "analysis.json").read_text())["results"],
              "2": json.loads((ph1.S2 / "analysis.json").read_text())["results"]}
    adj = ROOT / "experiments/20261110-adjudication"
    key = json.loads((adj / "private/key.json").read_text(encoding="utf-8"))
    R = {r: json.loads((adj / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
    rows = {}  # (study, id) -> (role, ref, [rater labels])
    for k, v in key.items():
        rows[(str(v["study"]), v["id"])] = (v["role"], v["ref"], [J[R[r][k]] for r in R])

    # P4's WordNet origin (Study 1 items), recomputed exactly as posthoc_round1.py does
    wn = ph1.inventory.get_wn(str(ROOT / ".nltk-win"))
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    pmap = {"N": "n", "V": "v"}
    items1 = {it["id"]: it for it in ph1.jl(ph1.S1 / "transform/wic_test_inputs.jsonl")}
    in_wn = {}
    for i, it in items1.items():
        ex = {norm(e) for s in wn.synsets(it["lemma"], pmap[it["pos"]]) for e in s.examples()}
        in_wn[i] = all(norm(it[f"sentence{j}"]) in ex for j in (1, 2))
    p4 = json.loads((ROOT / "papers/bics/tables/posthoc_round1.json").read_text())["P4_wordnet_examples"]
    assert sum(in_wn.values()) == p4["items_both_sentences_in_wordnet_examples"] and len(in_wn) == p4["items_total"], \
        "WordNet origin does not reproduce P4"

    res = {"note": "post hoc [P], not preregistered", "check_reproduces_frozen": {},
           "H1_hidden_errors": {}, "H2_contested_by_label_origin": {}}
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
        res["check_reproduces_frozen"][cell] = bool(ok)
        if base != "deberta":  # the adjudication sheet holds DeBERTa's items only
            continue

        shareA = float(np.mean([EA.method_pred(tb.rows["A"][i], "wic") == "F" for i in ids]))
        fl = {m: AN.q_mean_flags(prim[m]) for m in ("A", "i", "ii", "iii")}
        fl["ii_deb"] = AN.q_mean_flags(ph2.debias(tb, ids, "ii", shareA)[0])

        ctrl = [(i, ref, lab) for (s, i), (role, ref, lab) in rows.items() if s == study and role == "control"]
        missing = [i for i, _, _ in ctrl if i not in pos]
        hidden = [i for i, ref, lab in ctrl if i in pos and all(x_ == OPP[ref] for x_ in lab)]
        conf_ok = [i for i, ref, lab in ctrl if i in pos and all(x_ == ref for x_ in lab)]
        res["H1_hidden_errors"][cell] = {
            "n_controls": len(ctrl), "controls_not_ranked": len(missing),
            "hidden_errors_both_raters_opposite": summarise(hidden, fl, pos, units),
            "correct_both_raters_reference": summarise(conf_ok, fl, pos, units)}

        if study == "1":
            out = {}
            contested = [i for (s, i), (role, ref, lab) in rows.items()
                         if s == "1" and role == "error" and i in pos and i in set(st) and not all(x_ == ref for x_ in lab)]
            out["n_contested"] = len(contested)
            for ref_lab in ("T", "F"):
                for origin, keep in (("wordnet_examples", True), ("other", False)):
                    sid = [i for i in contested if tb.ref[i] == ref_lab and in_wn[i] == keep]
                    out[f"ref_{'same' if ref_lab == 'T' else 'different'}/{origin}"] = summarise(sid, fl, pos, units)
            res["H2_contested_by_label_origin"][cell] = out

    path = ROOT / "papers/bics/tables/posthoc_round5.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["check_reproduces_frozen"]))
    print("->", path)


if __name__ == "__main__":
    main()
