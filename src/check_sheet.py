"""Paraphrase quality check (preregistration v4, Sec. 4; Sec. 0, Step 2).

  python check_sheet.py build  --config ../config.json --out O      -> O/check/<task>_sheet.csv, <task>_key.json
  python check_sheet.py reveal --out O --task wic|nli                 -> lists the hard negatives (after the check)
  python check_sheet.py score  --config ../config.json --out O --task wic|nli [--hardneg-valid V.json]

Per task: 50 pairs from paraphrase 1 of set S (NLI: 25 premise side, 25 hypothesis side), 25 pairs from
paraphrase 2 of set B, and 10 controls (5 sentences of another item, 5 hard negatives). The 85 pairs are
shuffled with a fixed seed. The sheet does not show which pairs are controls; the key file holds that and
must not be opened before the sheet is filled. The rater fills the column `kept` with 1 (meaning kept) or 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import eval_data as D
from common import item_seed, read_jsonl
from eval_run import WIC_DATA, build_forms, load_config


def _rng(task, what):
    return np.random.default_rng(item_seed(task, "check", what))


def _draw(task, what, pool, n):
    assert len(pool) >= n, (task, what, len(pool), n)
    return [pool[int(j)] for j in _rng(task, what).choice(len(pool), size=n, replace=False)]


def wic_pairs(cfg, out):
    t = Path(out) / "transform"
    forms = build_forms(t)
    items = {it["id"]: it for d in WIC_DATA for it in read_jsonl(t / f"{d}_inputs.jsonl")}
    allf = {i: fs for d in WIC_DATA for i, fs in forms[d].items()}
    c = cfg["check"]
    S = sorted((i, k) for i, fs in allf.items() if "para1" in fs for k in (1, 2))
    B = sorted((i, k) for i, fs in allf.items() if "para2" in fs for k in (1, 2))
    row = lambda kind, i, k, new: {"kind": kind, "id": i, "k": k, "word": D.wic_word(items[i], k),
                                   "original": items[i][f"sentence{k}"], "rewritten": new}
    drawn_S = _draw("wic", "S", S, c["n_S"])
    pairs = [row("S_para1", i, k, allf[i]["para1"][f"sentence{k}"]) for i, k in drawn_S]
    pairs += [row("B_para2", i, k, allf[i]["para2"][f"sentence{k}"]) for i, k in _draw("wic", "B", B, c["n_B"])]
    # Controls: a sentence of another item (with a different lemma), and hard negatives. The control sources
    # are drawn from sentences not already on the sheet as S pairs.
    rest = [x for x in S if x not in set(drawn_S)]
    for i, k in _draw("wic", "other", rest, c["n_other"]):
        others = sorted(j for j in allf if items[j]["lemma"] != items[i]["lemma"])
        j = others[int(_rng("wic", "other_" + i).integers(len(others)))]
        pairs.append(row("C_other", i, k, items[j]["sentence1"]))
    hn = [r for r in read_jsonl(t / "wic_hardneg.jsonl") if r["parsed"]][: c["n_hardneg"]]
    pairs += [row("C_hardneg", r["id"], 1, r["text"]) for r in hn]  # may be fewer than n_hardneg; reported
    return pairs


def nli_pairs(cfg, out):
    t = Path(out) / "transform"
    forms = build_forms(t)["chaos"]
    items = {it["id"]: it for it in read_jsonl(t / "chaos_inputs.jsonl")}
    c = cfg["check"]
    row = lambda kind, i, side, f: {"kind": kind, "id": i, "side": side,
                                    "premise": items[i]["premise"], "hypothesis": items[i]["hypothesis"],
                                    "new_premise": f["premise"], "new_hypothesis": f["hypothesis"]}
    half = c["n_S"] // 2
    pairs = [row("S_prem_para1", i, "premise", forms[i]["prem_para1"])
             for i in _draw("nli", "S_prem", sorted(i for i, fs in forms.items() if "prem_para1" in fs), half)]
    pairs += [row("S_hyp_para1", i, "hypothesis", forms[i]["hyp_para1"])
              for i in _draw("nli", "S_hyp", sorted(i for i, fs in forms.items() if "hyp_para1" in fs), c["n_S"] - half)]
    pairs += [row("B_both_para2", i, "both", forms[i]["both_para2"])
              for i in _draw("nli", "B", sorted(i for i, fs in forms.items() if "both_para2" in fs), c["n_B"])]
    ids = sorted(forms)
    on_sheet = {p["id"] for p in pairs}
    for i in _draw("nli", "other", [x for x in ids if x not in on_sheet], c["n_other"]):
        others = [j for j in ids if D.nli_unit(items[j]) != D.nli_unit(items[i])]
        j = others[int(_rng("nli", "other_" + i).integers(len(others)))]
        pairs.append(row("C_other", i, "premise", {"premise": items[j]["premise"], "hypothesis": items[i]["hypothesis"]}))
    hn = [r for r in read_jsonl(t / "nli_hardneg.jsonl") if r["parsed"]][: c["n_hardneg"]]
    pairs += [row("C_hardneg", r["id"], "hypothesis", {"premise": items[r["id"]]["premise"], "hypothesis": r["text"]}) for r in hn]
    return pairs


SHEET_COLS = {"wic": ["row", "word", "original", "rewritten", "kept"],
              "nli": ["row", "premise", "hypothesis", "new_premise", "new_hypothesis", "kept"]}


def build(cfg, out):
    d = Path(out) / "check"
    d.mkdir(parents=True, exist_ok=True)
    for task, fn in (("wic", wic_pairs), ("nli", nli_pairs)):
        for f in (d / f"{task}_sheet.csv", d / f"{task}_key.json"):
            assert not f.exists(), f"{f} exists; the sheet is built once and never overwritten"
        pairs = fn(cfg, out)
        c = cfg["check"]
        n_hn = sum(p["kind"] == "C_hardneg" for p in pairs)
        assert len(pairs) == c["n_S"] + c["n_B"] + c["n_other"] + n_hn
        (d / f"{task}_build.json").write_text(json.dumps({"pairs": len(pairs), "hardneg": n_hn,
                                                          "hardneg_shortfall": c["n_hardneg"] - n_hn}))
        order = _rng(task, "shuffle").permutation(len(pairs))
        key = {}
        with open(d / f"{task}_sheet.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLS[task], extrasaction="ignore")
            w.writeheader()
            for r, j in enumerate(order, 1):
                p = dict(pairs[int(j)], row=r, kept="")
                key[r] = p
                w.writerow(p)
        (d / f"{task}_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
        print("SHEET", task, len(pairs), flush=True)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def score_rows(key, kept, hardneg_valid=None, s_min=0.90, max_controls_kept=1):
    """key: {row: pair}; kept: {row: 0/1}. hardneg_valid: {row: bool}, the researcher's after-check judgement
    of whether each hard negative really changed the sense (WiC) or the label (NLI); invalid ones are dropped
    from the control judgement. Returns the summary and whether the task's hypotheses go exploratory."""
    assert set(kept) == set(key), "every row must be rated"
    hardneg_valid = hardneg_valid or {}
    by = lambda pre: [int(r) for r, p in key.items() if p["kind"].startswith(pre)]
    S, B, C = by("S_"), by("B_"), by("C_")
    dropped = [r for r in C if key[r]["kind"] == "C_hardneg" and hardneg_valid.get(r) is False]
    C_used = [r for r in C if r not in dropped]
    kS = sum(kept[r] for r in S)
    lo, hi = wilson(kS, len(S))
    kC = sum(kept[r] for r in C_used)
    res = {"S_n": len(S), "S_kept": kS, "S_rate": kS / len(S), "S_wilson95": [lo, hi],
           "B_n": len(B), "B_kept": sum(kept[r] for r in B),
           "controls_used": len(C_used), "controls_kept": kC, "hardneg_dropped": len(dropped),
           "S_fail": kS / len(S) < s_min, "controls_fail": kC > max_controls_kept}
    res["exploratory"] = res["S_fail"] or res["controls_fail"]
    return res


def read_sheet(p):
    with open(p, encoding="utf-8-sig") as f:  # a spreadsheet program may add a BOM
        rows = list(csv.DictReader(f))
    kept = {}
    for r in rows:
        v = r["kept"].strip()
        assert v in ("0", "1"), f"row {r['row']}: kept must be 0 or 1, got {v!r}"
        kept[int(r["row"])] = int(v)
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "reveal", "score"])
    ap.add_argument("--config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["wic", "nli"])
    ap.add_argument("--hardneg-valid")
    a = ap.parse_args()
    d = Path(a.out) / "check"
    if a.cmd == "build":
        build(load_config(a.config), a.out)
    elif a.cmd == "reveal":
        key = json.loads((d / f"{a.task}_key.json").read_text())
        print(json.dumps({r: p for r, p in key.items() if p["kind"] == "C_hardneg"}, ensure_ascii=False, indent=1))
    else:
        cfg = load_config(a.config)
        key = {int(r): p for r, p in json.loads((d / f"{a.task}_key.json").read_text()).items()}
        hv = {int(r): bool(v) for r, v in json.loads(Path(a.hardneg_valid).read_text()).items()} if a.hardneg_valid else None
        res = score_rows(key, read_sheet(d / f"{a.task}_sheet.csv"), hv, cfg["check"]["s_min"], cfg["check"]["max_controls_kept"])
        (d / f"{a.task}_result.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
