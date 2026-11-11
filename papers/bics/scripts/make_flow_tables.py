"""Table 2 (item flow from the input data to stable errors, both studies; projection gap of study 2) and
Table S3 (human paraphrase check, both studies). Reads only frozen-run outputs; writes papers/bics/tables/*.csv.
  python -X utf8 papers/bics/scripts/make_flow_tables.py   (from the PJ root)
Raters are reported as "rater 1" / "rater 2" in the order in which the analysis files list them (rater 1 = the
researcher in both studies). No rater name is written by this script; sentences are never read into the outputs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S1 = ROOT / "experiments/20261103-evaluation/results/eval"
S2 = ROOT / "experiments/20261109-rep-main"
SRC = {"study1_analysis": S1 / "analysis.json",
       "study2_analysis": S2 / "rep/analysis.json",
       "study2_check": S2 / "rep/check/result.json",
       "study2_projection": S2 / "results/projection_gap.json"}
OUT = ROOT / "papers/bics/tables"
N_ITEMS = {"wic_test": 1400, "chaos": 3113, "mcl": 1654}  # input sizes ([19] Table 1, [26] Table 1, rep prereg Sec. 2)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe)


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "rows ->", OUT / name)


def flow():
    a1, a2 = load(SRC["study1_analysis"]), load(SRC["study2_analysis"])
    chk2 = load(SRC["study2_check"])
    rows = []
    cells = [("study1", a1, "wic_test", "deberta", "WiC test", "wic_test/deberta/qwen"),
             ("study1", a1, "wic_test", "flan", "WiC test", "wic_test/flan/qwen"),
             ("study1", a1, "chaos", "roberta", "ChaosNLI", "chaos/roberta/qwen"),
             ("study2", a2, "mcl", "deberta", "MCL-WiC dev+test", "mcl/deberta/qwen"),
             ("study2", a2, "mcl", "flan", "MCL-WiC dev+test", "mcl/flan/qwen")]
    for study, A, data, base, label, key in cells:
        c = A["counts"][f"{data}/{base}"]
        wn = set(c["wordnet_excluded"])
        nt = set(c["non_transformable"])
        check_excl = c.get("check_excluded", 0)
        rows.append({
            "study": study, "data": label, "base": base, "items": N_ITEMS[data],
            "wordnet_lt2_senses": len(wn),
            # items whose paraphrase-1 was not formed (both sentences / both sides), excluding WordNet-excluded items
            "paraphrase_not_formed": len(nt - wn) - check_excl,
            "excluded_by_human_check": check_excl,
            "reference_tie": len(c.get("no_reference", [])),
            "errors_classified": c["n_stable"] + c["n_unstable"],
            "unstable_errors": c["n_unstable"],
            "stable_errors": c["n_stable"],
            "units": c["units_stable"],
            "stable_after_parse_exclusion": A["results"][key]["n_stable"],
            "units_after_parse_exclusion": A["results"][key]["units_stable"],
            "stable_share_of_errors": round(c["stable_share_of_errors"], 3),
            "base_accuracy_report_only": round(c["accuracy_report_only"], 3),
            "stop_rule": "stable >= 100 and units >= 50 (base DeBERTa / RoBERTa; FLAN only for H1b in study 2)",
        })
    assert chk2["items_excluded"] == rows[3]["excluded_by_human_check"] == rows[4]["excluded_by_human_check"]
    write("table2_flow.csv", rows)

    pg = load(SRC["study2_projection"])
    prow = []
    names = {"error_rate": "base error rate", "para1_rate": "paraphrase 1 formed", "check_pass": "passed human check",
             "error_x_exclusion": "errors vs. check exclusion (correlation term)", "stable_share": "stable share of errors"}
    for base in ("deberta", "flan"):
        g = pg[base]
        for f, v in g["factors"].items():
            prow.append({"base": base, "factor": names[f], "projected": round(v["projected"], 3),
                         "actual": round(v["actual"], 3), "ratio": round(v["ratio"], 3)})
        prow.append({"base": base, "factor": "stable errors (product x 1,654)", "projected": round(g["projected"], 1),
                     "actual": g["n_stable"], "ratio": round(g["n_stable"] / g["projected"], 3)})
    write("table2_projection.csv", prow)


def check_table():
    a1 = load(SRC["study1_analysis"])
    chk2 = load(SRC["study2_check"])
    rows = []
    for task, label in (("wic", "WiC (study 1)"), ("nli", "ChaosNLI (study 1)")):
        raters = list(a1["check"][task]["raters"])  # order as listed in the analysis file
        kept = []
        for i, r in enumerate(raters, 1):
            v = a1["check"][task]["raters"][r]
            sheet = list(csv.DictReader(open(S1 / "check_raters" / r / "check" / f"{task}_sheet.csv", encoding="utf-8")))
            kept.append([int(x["kept"]) for x in sheet])
            rows.append({"study": "study1", "task": label, "rater": f"rater {i}", "design": "audit sample",
                         "paraphrase_rows": v["S_n"], "paraphrase_rows_kept": v["S_kept"],
                         "kept_rate": round(v["S_rate"], 3),
                         "wilson95": f"[{v['S_wilson95'][0]:.2f}, {v['S_wilson95'][1]:.2f}]",
                         "controls_used": v["controls_used"], "controls_kept": v["controls_kept"],
                         "rule": "kept rate >= 0.90 and controls kept <= 1, for both raters",
                         "verdict": "fail" if (v["S_fail"] or v["controls_fail"]) else "pass",
                         "agreement_pct": "", "kappa": ""})
        po, k = kappa(*kept)
        for r in rows[-2:]:
            r["agreement_pct"], r["kappa"] = round(po, 3), round(k, 3)
            r["agreement_scope"] = f"all {len(kept[0])} sheet rows"
    for i, (r, v) in enumerate(chk2["raters"].items(), 1):
        ctrl_kept = v["other_kept"] + v["hardneg_kept"] + v["meaning_kept"]
        rows.append({"study": "study2", "task": "MCL-WiC (study 2)", "rater": f"rater {i}", "design": "every item (filter)",
                     "paraphrase_rows": v["S_rows"], "paraphrase_rows_kept": v["S_rows_kept"],
                     "kept_rate": round(v["S_rows_kept"] / v["S_rows"], 3), "wilson95": "",
                     "controls_used": chk2["controls_other"] + chk2["controls_hardneg"] + chk2["controls_meaning"],
                     "controls_kept": ctrl_kept,
                     "rule": "item excluded if either rater rejects either sentence; attention checks on 60 controls",
                     "verdict": "attention checks passed" if not chk2["pair"]["attention_fail"] else "fail",
                     "agreement_pct": round(chk2["agreement_S_rows"]["percent"], 3),
                     "kappa": round(chk2["agreement_S_rows"]["kappa"], 3),
                     "agreement_scope": f"{v['S_rows']} paraphrase rows"})
    for r in rows:
        r.setdefault("agreement_scope", "")
    write("tableS3_check.csv", rows)
    halves = []
    for h in ("first", "second"):
        for i, (r, v) in enumerate(chk2["halves"][h].items(), 1):
            halves.append({"half": h, "rater": f"rater {i}", "paraphrase_rows": v["S_rows"],
                           "kept": v["S_rows_kept"], "kept_rate": round(v["S_rows_kept"] / v["S_rows"], 3),
                           "controls": v["controls"], "controls_kept": v["controls_kept"]})
    write("tableS3_halves.csv", halves)
    items = {"items_on_sheet": chk2["items_checked"], "items_excluded": chk2["items_excluded"],
             "items_passed": chk2["items_checked"] - chk2["items_excluded"],
             "pass_rate": round(1 - chk2["items_excluded"] / chk2["items_checked"], 3)}
    (OUT / "tableS3_items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flow()
    check_table()
    (OUT / "sources_flow.json").write_text(json.dumps(
        {k: {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
         for k, p in SRC.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
