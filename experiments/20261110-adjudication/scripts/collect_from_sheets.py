"""Collect the returned judgements read from Google Sheets copies of the returned csv files (2026-11-11).

Why: the returned csv files are on Drive, and the controlled Chrome does not download to this PC. Each csv was opened
as a Google Sheets copy, and only the judgement column and checksums computed in Sheets were read (the sentence columns
were never read). This replaces `adjudication.py collect` (a full text comparison) and is a deviation from freeze A
(notes.md).

Checks (Sheets SUMPRODUCT values; weight = sheet row = sheet.csv row index + 2):
  - row column equals 1..598; weighted length of `word` and weighted `pos == noun` equal the issued sheet;
  - weighted lengths of sentence1 / sentence2 and the weighted count of the letter e equal the issued sheet;
  - weighted row sums of S, D and U in the transcribed judgement string equal the Sheets values.
  python -X utf8 experiments/20261110-adjudication/scripts/collect_from_sheets.py
"""
import csv
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = json.loads((EXP / "private/returns/from_sheets.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(EXP / "private/sheet.csv", encoding="utf-8-sig")))
W = lambda i: i + 2
ref = {"word_len": sum(len(r["word"]) * W(i) for i, r in enumerate(rows)),
       "noun_rows": sum(W(i) for i, r in enumerate(rows) if r["pos"] == "noun"),
       "len_s1": sum(len(r["sentence1"]) * W(i) for i, r in enumerate(rows)),
       "len_s2": sum(len(r["sentence2"]) * W(i) for i, r in enumerate(rows)),
       "e_count": sum((r["sentence1"].lower().count("e") + r["sentence2"].lower().count("e")) * W(i) for i, r in enumerate(rows))}
report = {}
for rater, s in SRC["raters"].items():
    j, cs = s["judgements"], s["sheets_checksums"]
    assert len(j) == len(rows) and set(j) <= {"S", "D", "U"}, rater
    for c in "SDU":
        assert sum(W(i) for i, x in enumerate(j) if x == c) == cs[f"weighted_{c}"], (rater, c)
    assert cs["row_sq_dev"] == 0, rater
    for k, v in ref.items():
        assert cs[k] == v, (rater, k)
    p = EXP / f"results/ratings_{rater}.json"
    assert not p.exists(), p
    p.write_text(json.dumps({i + 1: x for i, x in enumerate(j)}, indent=0), encoding="utf-8")
    report[rater] = {"source": s["source"], "rows": len(j), "counts": {c: j.count(c) for c in "SDU"},
                     "sheet_match": True, "declaration_in_file": s["declaration_in_file"]}
(EXP / "results/collected.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
