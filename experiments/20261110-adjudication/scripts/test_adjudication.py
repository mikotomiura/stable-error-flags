"""Tests of the adjudication analysis with synthetic judgements (no rater data exists yet).

  .venv-win/Scripts/python.exe -X utf8 -m unittest experiments/20261110-adjudication/scripts/test_adjudication.py -v
(from the PJ root, after `adjudication.py build`). Reads private/key.json and results/flags.json; prints numbers only.
"""
import csv
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adjudication as ADJ  # noqa: E402

sys.path.insert(0, str(ADJ.ROOT / "src"))
import analysis as AN  # noqa: E402

KEY = {int(n): e for n, e in json.loads(ADJ.KEY.read_text(encoding="utf-8")).items()}
FLAGS = json.loads(ADJ.FLAGS.read_text(encoding="utf-8"))
INV = {"T": "S", "F": "D"}
OPP = {"T": "D", "F": "S"}


def ratings(fn):
    return {n: fn(n, e) for n, e in KEY.items()}


class PureFunctions(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(ADJ.classify("T", ["T", "T"]), {"class": "certain", "suspect": False})
        self.assertEqual(ADJ.classify("T", ["T", "U"]), {"class": "contested", "suspect": False})
        self.assertEqual(ADJ.classify("T", ["F", "T"]), {"class": "contested", "suspect": False})
        self.assertEqual(ADJ.classify("T", ["F", "F"]), {"class": "contested", "suspect": True})
        self.assertEqual(ADJ.classify("F", ["U", "U"]), {"class": "contested", "suspect": False})

    def test_kappa(self):
        self.assertAlmostEqual(ADJ.cohen_kappa("TTFF", "TFFF"), 0.5)  # po .75, pe .5
        self.assertAlmostEqual(ADJ.cohen_kappa("TFU", "TFU"), 1.0)
        self.assertIsNone(ADJ.cohen_kappa("TT", "TT"))  # one category only: undefined
        self.assertIsNone(ADJ.cohen_kappa([], []))

    def test_agreement(self):
        a = ADJ.agreement(["T", "F", "U", "T"], ["T", "T", "T", "T"])
        self.assertEqual((a["agree"], a["opposite"], a["undecided"]), (0.5, 0.25, 0.25))

    def test_reading(self):
        mk = lambda lo: {"n": 10, "A_minus": {"ii": {"lo": lo}}}
        self.assertEqual(ADJ.reading(mk(0.05), mk(-0.1)), "the structured gain remained on the errors the reference supports")
        self.assertEqual(ADJ.reading(mk(-0.1), mk(0.05)), "the structured gain depends on the contested items")
        self.assertEqual(ADJ.reading(mk(0.05), mk(0.10)), "the structured gain depends on the contested items")
        self.assertEqual(ADJ.reading(mk(-0.01), mk(-0.02)), "indistinguishable")
        self.assertEqual(ADJ.reading(mk(0.05), {"n": 0}), "not computed (an empty class)")

    def test_validate(self):
        built = {1: {"row": "1", "word": "w", "pos": "noun", "sentence1": "a [[w]]", "sentence2": "b [[w]]", "judgement": ""}}
        ok = [dict(built[1], judgement=" s ")]
        self.assertEqual(ADJ.validate(built, ok), {1: "S"})
        for bad in ([dict(built[1], judgement="X")], [dict(built[1], sentence1="a w", judgement="S")],
                    [dict(built[1], judgement="S"), dict(built[1], row="2", judgement="S")], []):
            with self.assertRaises(ValueError):
                ADJ.validate(built, bad)


class EndToEndSynthetic(unittest.TestCase):
    def test_all_agree_reproduces_frozen(self):
        """Raters agree with the reference everywhere: every error is certain, and the 'certain' stats equal the
        frozen analysis (same items, same units, same bootstrap draws)."""
        r = {"rater1": ratings(lambda n, e: INV[e["ref"]]), "rater2": ratings(lambda n, e: INV[e["ref"]])}
        res = ADJ.analyze_ratings(AN, KEY, r, FLAGS)
        self.assertEqual(res["classes"]["1"], {"errors": 208, "certain": 208, "contested": 0, "suspected_reference_error": 0})
        self.assertEqual(res["classes"]["2"]["certain"], 91)
        for cell, c in res["cells"].items():
            st = "experiments/20261103-evaluation/results/eval" if cell.startswith("wic_test") else "experiments/20261109-rep-main/rep"
            fr = json.loads((ADJ.ROOT / st / "analysis.json").read_text())["results"][cell]
            for m in ADJ.CONTROLS:
                for k in ("d", "lo", "hi"):
                    self.assertAlmostEqual(c["certain"]["A_minus"][m][k], fr["diff_qmean"][m][k], places=9)
            self.assertEqual(c["reading"], "not computed (an empty class)")
        self.assertEqual(res["agreement_with_reference"]["rater1"]["study1_control"]["agree"], 1.0)
        self.assertAlmostEqual(res["rater_agreement"]["all"]["kappa3"], 1.0)

    def test_opposite_on_errors(self):
        """Raters give the opposite of the reference on every error: all errors contested and suspected."""
        f = lambda n, e: OPP[e["ref"]] if e["role"] == "error" else INV[e["ref"]]
        res = ADJ.analyze_ratings(AN, KEY, {"rater1": ratings(f), "rater2": ratings(f)}, FLAGS)
        self.assertEqual(res["classes"]["1"]["suspected_reference_error"], 208)
        self.assertEqual(res["classes"]["2"]["contested"], 91)
        self.assertEqual(res["agreement_with_reference"]["rater2"]["study2_error"]["opposite"], 1.0)

    def test_split_and_kappa(self):
        """rater2 is undecided on every other error: those become contested; classes partition the errors; kappa < 1."""
        r1 = ratings(lambda n, e: INV[e["ref"]])
        r2 = ratings(lambda n, e: "U" if e["role"] == "error" and n % 2 == 0 else INV[e["ref"]])
        res = ADJ.analyze_ratings(AN, KEY, {"rater1": r1, "rater2": r2}, FLAGS)
        for st, n_err in (("1", 208), ("2", 91)):
            c = res["classes"][st]
            self.assertEqual(c["certain"] + c["contested"], n_err)
            exp = sum(1 for n, e in KEY.items() if e["study"] == st and e["role"] == "error" and n % 2 == 0)
            self.assertEqual(c["contested"], exp)
        self.assertLess(res["rater_agreement"]["errors"]["kappa3"], 1.0)
        self.assertAlmostEqual(res["rater_agreement"]["controls"]["percent"], 1.0)
        for c in res["cells"].values():
            self.assertIn(c["reading"], ("the structured gain remained on the errors the reference supports",
                                         "the structured gain depends on the contested items", "indistinguishable"))

    def test_cells_partition(self):
        r1 = ratings(lambda n, e: "U" if n % 3 == 0 else INV[e["ref"]])
        res = ADJ.analyze_ratings(AN, KEY, {"rater1": r1, "rater2": r1}, FLAGS)
        for name, c in res["cells"].items():
            self.assertEqual(c["certain"]["n"] + c["contested"]["n"], len(FLAGS[name]["stable"]))

    def test_single_rater(self):
        res = ADJ.analyze_ratings(AN, KEY, {"rater1": ratings(lambda n, e: INV[e["ref"]])}, FLAGS)
        self.assertNotIn("rater_agreement", res)
        self.assertEqual(res["n_raters"], 1)


class SheetBlinding(unittest.TestCase):
    def test_sheet_hides_key(self):
        with open(ADJ.SHEET, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), ADJ.COLS)
        self.assertEqual(len(rows), len(KEY))
        self.assertTrue(all(r["judgement"] == "" for r in rows))
        self.assertTrue(all(r["sentence1"].count("[[") == 1 and r["sentence2"].count("[[") == 1 for r in rows))
        # controls match errors on reference-label counts, per study
        for st in ("1", "2"):
            for y in ("T", "F"):
                ne = sum(1 for e in KEY.values() if e["study"] == st and e["role"] == "error" and e["ref"] == y)
                nc = sum(1 for e in KEY.values() if e["study"] == st and e["role"] == "control" and e["ref"] == y)
                self.assertEqual(ne, nc)


if __name__ == "__main__":
    unittest.main()
