"""GPU stages of the replication study on MCL-WiC (rep_preregistration.md, Sec. 0).

  python rep_run.py dev        --config ../rep_config.json --raw R --model <qwen dir> --models <base dir> --nltk N --out O
  python rep_run.py transform  --config ../rep_config.json --raw R --model <qwen dir> --nltk N --out O
  python rep_run.py base       --config ../rep_config.json --models <dir with deberta, flan> --out O
  python rep_run.py structured --config ../rep_config.json --model <qwen or olmo dir> --model-name qwen|olmo --nltk N --out O

- dev runs on a seeded sample of TRAIN only (prompt development, before the freeze). It never reads dev/test.
- transform reads the evaluation inputs only (no references): paraphrases 1 and 2 and the hard-negative
  candidates for the rater controls (judged before the sheets are built, rep_check.py). base needs the scored
  check; structured needs counts.json.
- structured also runs, for Qwen only, the main evaluation's A (frozen wic_A: "None of the above" option and its
  own wording) as "A_v4": reported only, never used for a decision (Sec. 5; decisions #11).
- The helpers of eval_run (environment record, weight and WordNet checks, per-file outputs that are skipped
  when present) are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import eval_data as D
import methods as M
import rep_data as R
import rep_methods as RM
from common import item_seed, read_jsonl, sha_file, write_jsonl
from eval_run import check_weights, check_wordnet, done, load_config, record_env, run_file, tpl

DATA = "mcl"


def check_raw(cfg, raw):
    got = {n: sha_file(Path(raw) / n) for n in (R.ALL_ZIP, R.GOLD_ZIP)}
    bad = sorted(n for n, v in got.items() if v != cfg["data_sha256"][n])
    assert not bad, f"MCL-WiC files differ from the frozen digests: {bad}"
    print("DATA_SHA_MATCH", flush=True)


def forms_of(t):
    """WiC forms (eval_data.wic_forms) of every evaluation item, from the transform outputs."""
    items = read_jsonl(t / f"{DATA}_inputs.jsonl")
    p1 = {r["id"]: r for r in read_jsonl(t / f"{DATA}_wic_para1.jsonl")}
    p2 = {r["id"]: r for r in read_jsonl(t / f"{DATA}_wic_para2.jsonl")}
    pick = lambda r: (r["s1"]["text"], r["s2"]["text"]) if r["s1"]["parsed"] and r["s2"]["parsed"] else None
    return {it["id"]: D.wic_forms(it, pick(p1[it["id"]]), pick(p2[it["id"]])) for it in items}


def structured_jobs(cfg, lm, wn, pool, data, with_v4=False):
    from inventory import inventory_A, inventory_iv
    tA = tpl(cfg, "rep_wic_A")

    def inv(it):
        return inventory_A(wn, data, it["id"], it["lemma"], it["pos"])
    jobs = {"A": lambda it: RM.wic_structured_forced(lm, tA, it, inv(it), "A"),
            "iv": lambda it: RM.wic_structured_forced(lm, tA, it, inventory_iv(wn, pool, data, it["id"], it["lemma"], it["pos"], inv(it)), "iv"),
            "ii": lambda it: M.wic_direct(lm, tpl(cfg, "wic_ii"), it),
            "iii": lambda it: M.wic_two_step(lm, M.split_templates(tpl(cfg, "wic_iii")), it)}
    if with_v4:  # the main evaluation's A, unchanged (reported only)
        jobs["A_v4"] = lambda it: M.wic_structured(lm, tpl(cfg, "wic_A"), it, inv(it), "A_v4")
    return jobs


# ---------- development on train (before the freeze) ----------

def dev(cfg, a):
    from base_models import run_deberta, run_flan
    from inventory import DonorPool, inventory_A
    from lm import ChatLM
    out = Path(a.out) / "dev"
    out.mkdir(parents=True, exist_ok=True)
    record_env(cfg, a, "dev")
    wn = check_wordnet(cfg, a.nltk)
    train = R.load_inputs(a.raw, ("train",))
    gold = R.load_gold(a.raw, ("train",))
    order = np.random.default_rng(item_seed("mcl_train", "dev", "sample")).permutation(len(train))
    items = [train[int(j)] for j in order[: a.n]]
    items = [it for it in items if inventory_A(wn, "mcl_train", it["id"], it["lemma"], it["pos"]) is not None]
    write_jsonl(out / "train_sample.jsonl", items)
    (out / "train_sample_gold.json").write_text(json.dumps({it["id"]: gold[it["id"]] for it in items}))
    for name, fn in (("deberta", run_deberta), ("flan", run_flan)):
        p = out / f"base_{name}.jsonl"
        if not done(p):
            check_weights(cfg, Path(a.models) / name, name)
            rows, skipped = fn(Path(a.models) / name, items, cfg["_root"] / "prompts")
            write_jsonl(p, rows)
            print("BASE", name, len(rows), "skipped", len(skipped), flush=True)
    check_weights(cfg, a.model, "qwen")
    lm = ChatLM(a.model)
    for k in ("wic_para1", "wic_para2"):
        p = out / f"{k}.jsonl"
        if not done(p):
            t = tpl(cfg, k)
            run_file(p, lambda it: {"s1": M.wic_paraphrase(lm, t, it, 1, k), "s2": M.wic_paraphrase(lm, t, it, 2, k)}, items, lm)
    pool = DonorPool(wn)
    for name, fn in structured_jobs(cfg, lm, wn, pool, "mcl_train").items():
        p = out / f"{name}.jsonl"
        if not done(p):
            run_file(p, fn, items, lm)
    p = out / "hardneg.jsonl"
    if not done(p):
        run_file(p, lambda it: M.wic_hardneg(lm, tpl(cfg, "wic_hardneg"), it), items[:40], lm)
    print("DEV_DONE", flush=True)


# ---------- Step 1: transform (inputs only; hard-negative candidates for Step 2) ----------

def transform(cfg, a):
    from inventory import senses
    from lm import ChatLM
    out = Path(a.out) / "transform"
    out.mkdir(parents=True, exist_ok=True)
    check_raw(cfg, a.raw)
    record_env(cfg, a, "transform")
    items = R.load_inputs(a.raw)
    if not done(out / f"{DATA}_inputs.jsonl"):
        write_jsonl(out / f"{DATA}_inputs.jsonl", items)
    wn = check_wordnet(cfg, a.nltk)
    few = [it["id"] for it in items if len(senses(wn, it["lemma"], it["pos"])) < 2]
    check_weights(cfg, a.model, "qwen")
    lm = ChatLM(a.model)
    for k in ("wic_para1", "wic_para2"):
        p = out / f"{DATA}_{k}.jsonl"
        if not done(p):
            t = tpl(cfg, k)
            run_file(p, lambda it: {"s1": M.wic_paraphrase(lm, t, it, 1, k), "s2": M.wic_paraphrase(lm, t, it, 2, k)}, items, lm)
    # Hard-negative candidates for the rater controls (Sec. 4): hardneg_candidates items in a seeded order, all
    # generated. Their validity is judged before the sheets are built (rep_check.py judge-sheet / build).
    c = cfg["check"]
    p = out / "wic_hardneg.jsonl"
    if not done(p):
        order = np.random.default_rng(item_seed("mcl", "check", "hardneg")).permutation(len(items))[: c["hardneg_candidates"]]
        rows = []
        for j in order:
            r = M.wic_hardneg(lm, tpl(cfg, "wic_hardneg"), items[int(j)])
            r["id"] = items[int(j)]["id"]
            rows.append(r)
        write_jsonl(p, rows)
    forms = forms_of(out)
    rep = {"items": len(items), "by_pos": {p_: sum(it["pos"] == p_ for it in items) for p_ in ("N", "V")},
           "fewer_than_two_senses": len(few), "para1_formed": sum("para1" in f for f in forms.values()),
           "para2_formed": sum("para2" in f for f in forms.values()),
           "hardneg_parsed": sum(r["parsed"] for r in read_jsonl(p))}
    (out / "wordnet_exclusions.json").write_text(json.dumps({DATA: {i: 1 for i in few}}))
    (out / "transform_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


# ---------- Step 4: base models (after the scored check, Step 3) ----------

def base(cfg, a):
    from base_models import run_deberta, run_flan
    o = Path(a.out)
    out = o / "base"
    out.mkdir(parents=True, exist_ok=True)
    assert done(o / "transform" / "transform_summary.json"), "run transform first (Sec. 0, Step 1)"
    assert done(o / "check" / "result.json"), "score the paraphrase check first (Sec. 0, Step 3)"
    record_env(cfg, a, "base")
    forms = forms_of(o / "transform")
    for name in ("deberta", "flan"):
        check_weights(cfg, Path(a.models) / name, name)
    for name, fn in (("deberta", run_deberta), ("flan", run_flan)):
        for f in D.WIC_S + D.WIC_B:
            p = out / f"{DATA}_{name}_{f}.jsonl"
            if done(p):
                continue
            items = [fs[f] for fs in forms.values() if f in fs]
            t = time.monotonic()
            rows, skipped = fn(Path(a.models) / name, items, cfg["_root"] / "prompts")
            write_jsonl(p, rows)
            p.with_suffix(".meta.json").write_text(json.dumps({"n": len(rows), "skipped_too_long": skipped, "seconds": time.monotonic() - t}))
            print("BASE", name, f, len(rows), "skipped", len(skipped), flush=True)


# ---------- Step 6: structured methods and controls (after count, Step 5) ----------

def structured(cfg, a):
    from inventory import DonorPool, inventory_A
    from lm import ChatLM
    o = Path(a.out)
    assert done(o / "counts.json"), "run rep_analysis.py count first (Sec. 0, Step 5)"
    out = o / f"structured_{a.model_name}"
    out.mkdir(parents=True, exist_ok=True)
    record_env(cfg, a, f"structured_{a.model_name}")
    wn = check_wordnet(cfg, a.nltk)
    check_weights(cfg, a.model, a.model_name)
    lm = ChatLM(a.model)
    items = read_jsonl(o / "transform" / f"{DATA}_inputs.jsonl")
    keep = [it for it in items if inventory_A(wn, DATA, it["id"], it["lemma"], it["pos"]) is not None]
    for name, fn in structured_jobs(cfg, lm, wn, DonorPool(wn), DATA, with_v4=a.model_name == "qwen").items():
        p = out / f"{DATA}_{name}.jsonl"
        if not done(p):
            run_file(p, fn, keep, lm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "transform", "base", "structured"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw")
    ap.add_argument("--model")
    ap.add_argument("--model-name", default="qwen")
    ap.add_argument("--models")
    ap.add_argument("--nltk")
    ap.add_argument("--n", type=int, default=200, help="dev only: size of the train sample")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"dev": dev, "transform": transform, "base": base, "structured": structured}[a.stage](cfg, a)
