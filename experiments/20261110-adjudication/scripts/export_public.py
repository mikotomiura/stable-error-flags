"""Public release of the adjudication (no sentences): item id, dataset, role, reference, both judgements, class.
WiC test ids are the 0-based line index in the official test file (test.data.txt); MCL-WiC ids are the official ids.
Contains no text from the datasets (CC BY-NC 4.0 restricts the sentences; ids and labels carry no sentence text).
  python -X utf8 experiments/20261110-adjudication/scripts/export_public.py   (from the PJ root)
Output: papers/bics/data/adjudication_items.csv"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/20261110-adjudication"
key = json.loads((EXP / "private/key.json").read_text(encoding="utf-8"))
R = {r: json.loads((EXP / f"results/ratings_{r}.json").read_text(encoding="utf-8")) for r in ("rater1", "rater2")}
J = {"S": "same", "D": "different", "U": "undecided"}
REF = {"T": "same", "F": "different"}
out = ROOT / "papers/bics/data/adjudication_items.csv"
out.parent.mkdir(exist_ok=True)
rows = []
for k, v in key.items():
    ref = REF[v["ref"]]
    j = [J[R[r][k]] for r in ("rater1", "rater2")]
    cls = "certain" if all(x == ref for x in j) else ("both_raters_opposite" if all(x not in (ref, "undecided") for x in j) else "contested_other")
    data, iid = ("WiC test", v["id"].split("_")[-1]) if v["study"] == "1" else ("MCL-WiC En-En", v["id"])
    rows.append({"dataset": data, "item_id": iid, "role": "stable_error" if v["role"] == "error" else "stable_correct",
                 "reference": ref, "rater1": j[0], "rater2": j[1], "class": cls})
rows.sort(key=lambda r: (r["dataset"], r["role"], r["item_id"]))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("->", out, len(rows))
