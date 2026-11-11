"""Main tables of the paper (study 1 = main evaluation v4, freeze 2; study 2 = replication on MCL-WiC, freeze R).
Reads only the frozen-run outputs (analysis.json of each study); writes papers/bics/tables/*.csv.
  python -X utf8 papers/bics/scripts/make_tables.py   (from the PJ root)"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = {"study1": ROOT / "experiments/20261103-evaluation/results/eval/analysis.json",
       "study2": ROOT / "experiments/20261109-rep-main/rep/analysis.json"}
OUT = ROOT / "papers/bics/tables"
# (study, key in results, data label, status of the confirmatory hypothesis that used this cell)
CELLS = [("study1", "wic_test/deberta/qwen", "WiC test", "H1/H2 (exploratory: check failed)"),
         ("study1", "wic_test/flan/qwen", "WiC test", "secondary"),
         ("study1", "wic_test/deberta/olmo", "WiC test", "secondary"),
         ("study1", "chaos/roberta/qwen", "ChaosNLI", "H3 (exploratory: check failed)"),
         ("study1", "chaos/roberta/olmo", "ChaosNLI", "secondary"),
         ("study2", "mcl/deberta/qwen", "MCL-WiC dev+test", "H1/H2 (aborted: 91 < 100)"),
         ("study2", "mcl/flan/qwen", "MCL-WiC dev+test", "H1b (aborted)"),
         ("study2", "mcl/deberta/olmo", "MCL-WiC dev+test", "secondary"),
         ("study2", "mcl/flan/olmo", "MCL-WiC dev+test", "secondary")]
CTRL = ("i", "ii", "iii", "iv")


def main():
    A = {s: json.loads(p.read_text()) for s, p in SRC.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, k, data, status in CELLS:
        r = A[s]["results"][k]
        _, base, llm = k.split("/")
        row = {"study": s, "data": data, "base": base, "llm": llm, "status": status,
               "n_stable": r["n_stable"], "units": r["units_stable"], "det_A": round(r["detection"]["A"], 3)}
        for c in CTRL:
            d = r["diff_qmean"][c]
            row[f"det_{c}"] = round(r["detection"][c], 3)
            row[f"A-{c}"] = round(d["d"], 3)
            row[f"A-{c}_lo"] = round(d["lo"], 3)
            row[f"A-{c}_hi"] = round(d["hi"], 3)
        rows.append(row)
    with open(OUT / "table_main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (OUT / "sources.json").write_text(json.dumps({s: {"path": str(p.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for s, p in SRC.items()}, indent=1))
    print(len(rows), "rows ->", OUT / "table_main.csv")


if __name__ == "__main__":
    main()
