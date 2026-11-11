"""Post-hoc P5 (attack-me round 1, reviewer R1; NOT preregistered, reported in S8 with [P]): flag leakage.

Every method flags the top q of ALL ranked items, but stable errors can only be found among items whose set-S forms
exist (paraphrase formed and, in Study 2, kept by the human check). A method that puts more of its flags on items that
can never be stable errors has fewer flags left for the target. This script reports, for DeBERTa x Qwen / OLMo in both
studies, the share of each method's flag mass (q-mean) that falls on such ineligible items, next to the share of
ineligible items among the ranked items (what a random ranking would give).

Reuses the frozen code through experiments/20261110-adjudication/scripts/adjudication.py (studies(), which reproduces
the frozen stable-error lists) and the frozen Table / flags_over; the flags are checked against analysis.json there.
  python -X utf8 papers/bics/scripts/posthoc_leakage.py   (from the PJ root)
Output: papers/bics/tables/posthoc_leakage.json (aggregates only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("adj", ROOT / "experiments/20261110-adjudication/scripts/adjudication.py")
adj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adj)


def main():
    AN, EA, D, R, read_jsonl, build_forms, forms_of = adj._frozen()
    _, _, S = adj.studies()
    s1, s2 = S["1"]["out"], S["2"]["out"]
    forms = {"1": build_forms(s1 / "transform", data=("wic_test",))["wic_test"], "2": forms_of(s2 / "transform")}
    for i in json.loads((s2 / "check/result.json").read_text())["excluded"]:
        forms["2"][i].pop("para1", None)
    res = {"note": "post hoc [P], not preregistered; DeBERTa cells; flag mass = sum over ranked items of the q-mean flag"}
    for st, s in S.items():
        cell_res, _ = adj.cell_flags(AN, EA, s)  # asserts reproduction of the frozen analysis
        preds = {f: {i: r["pred"] for i, r in EA.base_rows(s["out"], s["data"], "deberta", f).items()} for f in D.WIC_S}
        qwen_ids = None
        for llm in ("qwen", "olmo"):
            tb = EA.Table(s["cfg"], s["out"], s["data"], "wic", "deberta", llm, s["counts"], s["ref"],
                          restrict=None if llm == "qwen" else qwen_ids)
            if llm == "qwen":
                qwen_ids = set(tb.ids)
            ids = tb.ids
            flags = EA.flags_over({m: tb.scores(m, ids) for m in adj.PRIM}, ids)
            inelig = {i for i in ids if any(f not in forms[st][i] or i not in preds[f] for f in D.WIC_S)}
            key = f"{s['data']}/deberta/{llm}"
            # consistency with the frozen detection rate (same flags as cell_flags)
            stab = [i for i in s["stable"] if i in set(ids)]
            for m in adj.PRIM:
                assert abs(np.mean([flags[m][i] for i in stab]) - np.mean([v["flags"][m] for v in cell_res[key]["stable"].values()])) < 1e-12
            tot = {m: float(sum(flags[m][i] for i in ids)) for m in adj.PRIM}
            res[f"study{st}:{key}"] = {
                "ranked_items": len(ids), "ineligible_items": len(inelig),
                "ineligible_share_of_items": len(inelig) / len(ids),
                "flag_mass_share_on_ineligible": {m: float(sum(flags[m][i] for i in inelig)) / tot[m] for m in adj.PRIM}}
    out = ROOT / "papers/bics/tables/posthoc_leakage.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
