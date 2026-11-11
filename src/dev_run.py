"""Prompt development on the training side (preregistration v4, Sec. 6).

For a given version of each prompt, report the parse rate, the task accuracy (reported only), and speed.
Usage: python dev_run.py --model <dir> --model-name qwen --trials <prompts_trials> --versions '{"wic_A":"v1",...}'
       --wic <wic_train200.jsonl> --mnli <mnli_train200.jsonl> --nltk <dir> --out <dir> [--only wic_A,wic_ii]
Versions are never selected by accuracy or detection rate (Sec. 6); accuracy is recorded for reporting only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import methods as M
from common import load_template, read_jsonl, write_jsonl
from inventory import DonorPool, get_wn, inventory_A, inventory_iv

ALL = ["wic_para1", "wic_para2", "wic_hardneg", "wic_A", "wic_iv", "wic_ii", "wic_iii",
       "nli_para1", "nli_para2", "nli_hardneg", "nli_atoms", "nli_A", "nli_iv", "nli_ii", "nli_iii"]
TEMPLATE_OF = {"wic_iv": "wic_A", "nli_iv": "nli_A"}  # (iv) uses the procedure and template of A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--versions", required=True)
    ap.add_argument("--wic", required=True)
    ap.add_argument("--mnli", required=True)
    ap.add_argument("--nltk", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(ALL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--m", type=int, default=0, help="m for NLI (0: derive it from the atom decomposition)")
    a = ap.parse_args()
    versions = json.loads(a.versions)
    only = a.only.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def tpl(method):
        key = TEMPLATE_OF.get(method, method)
        return load_template(Path(a.trials) / key / f"{versions[key]}.txt")

    from lm import ChatLM
    t0 = time.monotonic()
    lm = ChatLM(a.model)
    load_s = time.monotonic() - t0
    wic = read_jsonl(a.wic)
    nli = read_jsonl(a.mnli)
    if a.limit:
        wic, nli = wic[: a.limit], nli[: a.limit]
    summary = {"model": a.model_name, "versions": versions, "load_seconds": load_s, "methods": {}}

    def record(method, rows, extra=None):
        n = len(rows)
        parsed = sum(r["parsed"] for r in rows)
        logs = [l for l in lm.log if l.get("method") == method]
        s = {"n": n, "parsed": parsed, "parse_rate": parsed / n if n else None,
             "prompts": len(logs), "prompts_per_item": len(logs) / n if n else None,
             "in_tokens_mean": statistics.mean(l["in_tokens"] for l in logs) if logs else None,
             "out_tokens_mean": statistics.mean(l["out_tokens"] for l in logs) if logs else None,
             "seconds": sum(l["seconds"] for l in logs), "seconds_per_prompt": (sum(l["seconds"] for l in logs) / len(logs)) if logs else None}
        if extra:
            s.update(extra)
        summary["methods"][method] = s
        write_jsonl(out / f"{method}.jsonl", rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(method, json.dumps(s, ensure_ascii=False), flush=True)

    def run(method, fn, items):
        start = len(lm.log)
        rows = []
        for it in items:
            r = fn(it)
            r["id"] = it["id"]
            rows.append(r)
        for l in lm.log[start:]:
            l["method"] = method
        return rows

    def acc_wic(rows, key="p_same"):
        lab = {x["id"]: x["label"] for x in wic}
        ok = [((r[key] >= 0.5) == (lab[r["id"]] == "T")) for r in rows if r.get(key) is not None]
        return {"accuracy_report_only": sum(ok) / len(ok) if ok else None}

    def acc_nli(rows):
        lab = {x["id"]: x["label"] for x in nli}
        ok = [M.NLI_LABELS[max(range(3), key=lambda j: r["probs"][j])] == lab[r["id"]] for r in rows]
        return {"accuracy_report_only": sum(ok) / len(ok) if ok else None}

    # ---- WiC ----
    if any(m.startswith("wic") for m in only):
        wn = get_wn(a.nltk)
        invA, keep = {}, []
        for x in wic:
            inv = inventory_A(wn, "wic_train", x["id"], x["lemma"], x["pos"])
            if inv is not None:
                invA[x["id"]] = inv
                keep.append(x)
        summary["wic_excluded_lt2_senses"] = len(wic) - len(keep)
        for k in ("wic_para1", "wic_para2"):
            if k in only:
                rows = run(k, lambda it, k=k: {"s1": M.wic_paraphrase(lm, tpl(k), it, 1, k), "s2": M.wic_paraphrase(lm, tpl(k), it, 2, k)}, wic)
                for r in rows:
                    r["parsed"] = r["s1"]["parsed"] and r["s2"]["parsed"]
                record(k, rows, {"sentence_parse_rate": sum(r["s1"]["parsed"] + r["s2"]["parsed"] for r in rows) / (2 * len(rows))})
        if "wic_hardneg" in only:
            record("wic_hardneg", run("wic_hardneg", lambda it: M.wic_hardneg(lm, tpl("wic_hardneg"), it), wic))
        if "wic_A" in only:
            rows = run("wic_A", lambda it: M.wic_structured(lm, tpl("wic_A"), it, invA[it["id"]], "A"), keep)
            record("wic_A", rows, acc_wic(rows) | {"none_greedy_rate": sum(sum(r["choice_correct_none"]) for r in rows) / (2 * len(rows))})
        if "wic_iv" in only:
            pool = DonorPool(wn)
            inv_iv = {x["id"]: inventory_iv(wn, pool, "wic_train", x["id"], x["lemma"], x["pos"], invA[x["id"]]) for x in keep}
            rows = run("wic_iv", lambda it: M.wic_structured(lm, tpl("wic_iv"), it, inv_iv[it["id"]], "iv"), keep)
            record("wic_iv", rows, acc_wic(rows) | {"none_greedy_rate": sum(sum(r["choice_correct_none"]) for r in rows) / (2 * len(rows))})
        if "wic_ii" in only:
            rows = run("wic_ii", lambda it: M.wic_direct(lm, tpl("wic_ii"), it), keep)
            record("wic_ii", rows, acc_wic(rows))
        if "wic_iii" in only:
            ts = M.split_templates(tpl("wic_iii"))
            rows = run("wic_iii", lambda it: M.wic_two_step(lm, ts, it), keep)
            record("wic_iii", rows, acc_wic(rows) | {"step1_parse_rate": sum(r["step1_parsed"] for r in rows) / len(rows)})

    # ---- NLI ----
    if any(m.startswith("nli") for m in only):
        for k in ("nli_para1", "nli_para2"):
            if k in only:
                rows = run(k, lambda it, k=k: {"premise": M.nli_paraphrase(lm, tpl(k), it["premise"], k),
                                               "hypothesis": M.nli_paraphrase(lm, tpl(k), it["hypothesis"], k)}, nli)
                for r in rows:
                    r["parsed"] = r["premise"]["parsed"] and r["hypothesis"]["parsed"]
                record(k, rows, {"side_parse_rate": sum(r["premise"]["parsed"] + r["hypothesis"]["parsed"] for r in rows) / (2 * len(rows))})
        if "nli_hardneg" in only:
            record("nli_hardneg", run("nli_hardneg", lambda it: M.nli_hardneg(lm, tpl("nli_hardneg"), it), nli))
        atoms = {}
        if {"nli_atoms", "nli_A", "nli_iv"} & set(only):
            rows = run("nli_atoms", lambda it: M.nli_atoms(lm, tpl("nli_atoms"), it), nli)
            atoms = {r["id"]: r["atoms"] for r in rows}
            n_at = [len(r["atoms"]) for r in rows]
            record("nli_atoms", rows, {"atoms_mean": statistics.mean(n_at), "atoms_max": max(n_at),
                                       "truncated_gt8": sum(r["n_raw"] > M.MAX_ATOMS for r in rows),
                                       "m_candidate": round(1 + statistics.mean(n_at))})
        m = a.m or (summary["methods"]["nli_atoms"]["m_candidate"] if "nli_atoms" in summary["methods"] else 0)
        summary["nli_m_used"] = m
        if "nli_A" in only:
            rows = run("nli_A", lambda it: M.nli_structured(lm, tpl("nli_A"), it, atoms[it["id"]] or [it["premise"]], "A"), nli)
            for r in rows:
                r["parsed"] = r["parsed"] and bool(atoms[r["id"]])
            cs = [c for r in rows for c in r["c"]]
            es = [e for r in rows for e in r["e"]]
            record("nli_A", rows, acc_nli(rows) | {"c_mean": statistics.mean(cs), "e_mean": statistics.mean(es),
                                                   "prompts_A_mean_incl_atoms": 1 + statistics.mean(len(atoms[r["id"]]) for r in rows)})
        if "nli_iv" in only:
            def iv(it):
                d = M.nli_donor("mnli_train", nli, it)
                r = M.nli_structured(lm, tpl("nli_iv"), it, atoms[d["id"]] or [d["premise"]], "iv")
                r["donor"] = d["id"]
                return r
            rows = run("nli_iv", iv, nli)
            record("nli_iv", rows, acc_nli(rows))
        if "nli_ii" in only and m:
            ts = M.split_templates(tpl("nli_ii"))
            assert m <= len(ts), (m, len(ts))
            rows = run("nli_ii", lambda it: M.nli_direct(lm, ts, it, m), nli)
            record("nli_ii", rows, acc_nli(rows) | {"per_template_parse_rate": [sum(r["per_template_parsed"][j] for r in rows) / len(rows) for j in range(m)]})
        if "nli_iii" in only and m:
            ts = M.split_templates(tpl("nli_iii"))
            n_pairs = math.ceil(m / 2)
            assert 2 * n_pairs <= len(ts), (n_pairs, len(ts))
            rows = run("nli_iii", lambda it: M.nli_two_step(lm, ts, it, n_pairs), nli)
            record("nli_iii", rows, acc_nli(rows))
    summary["total_seconds"] = time.monotonic() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    write_jsonl(out / "prompt_log.jsonl", lm.log)


if __name__ == "__main__":
    main()
