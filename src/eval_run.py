"""GPU stages on the evaluation data, run once after freeze 2 (preregistration v4, Sec. 0, Steps 1, 3 and 5).

  python eval_run.py transform  --config ../config.json --wic-zip W --chaos-zip C --grain G --model <qwen dir> --nltk N --out O
  python eval_run.py base       --config ../config.json --models <dir with deberta, flan, roberta> --out O
  python eval_run.py structured --config ../config.json --model <qwen or olmo dir> --model-name qwen|olmo --nltk N --out O

- transform reads the inputs only (no references): paraphrases 1 and 2, the hard negatives for the check
  sheet, the WordNet exclusions and the NLI units. It needs nothing else.
- base needs the transform outputs; structured needs counts.json (Step 4), so the order of Sec. 0 is
  enforced by the code.
- Model weights are checked against the digests recorded at the first fetch before loading, and the WordNet
  corpus against the digest in config.json.
- Every stage writes one file per (data, method), with its timing and its prompt log, and skips files that
  already exist. A run stopped by an exception or a lack of resources can be restarted without recomputing
  (the only allowed re-run, Sec. 0).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import eval_data as D
import methods as M
from common import item_seed, load_template, read_jsonl, sha_file, write_jsonl

WIC_DATA = ("wic_dev", "wic_test")
GRAIN = "grain"  # GrainWiC coarse: secondary, base DeBERTa only, no paraphrase check (Sec. 2)
WIC_ALL = WIC_DATA + (GRAIN,)


def load_config(p):
    cfg = json.loads(Path(p).read_text())
    cfg["_root"] = Path(p).resolve().parent
    return cfg


def tpl(cfg, method):
    return load_template(cfg["_root"] / cfg["prompts"][method])


def done(p):
    return Path(p).exists()


def check_weights(cfg, model_dir, name):
    from base_models import verify_weights
    verify_weights(model_dir, name, cfg["_root"] / cfg["weights_sha256"])


def check_wordnet(cfg, nltk_dir):
    from inventory import get_wn
    p = Path(nltk_dir) / "corpora" / "wordnet.zip"
    assert sha_file(p) == cfg["wordnet_zip_sha256"], "WordNet corpus differs from the frozen one"
    return get_wn(nltk_dir)


SMOKE_MARK = "SMOKE"  # written into --out by a --limit run; a full run refuses an output folder that has it


def check_data(cfg, a):
    """Every evaluation file must match the digest recorded before freeze 2 (config.json, data_sha256)."""
    import hashlib
    want = cfg["data_sha256"]
    got = {"WiC_dataset.zip": sha_file(a.wic_zip), "grainwic_coarse_test.jsonl": sha_file(a.grain)}
    for data, name in D.CHAOS_FILES.items():
        got[f"chaosnli/{name}"] = hashlib.sha256(D.chaos_bytes(a.chaos_zip, data)).hexdigest()
    bad = sorted(k for k, v in got.items() if v != want[k])
    assert not bad, f"data files differ from the frozen digests: {bad}"
    print("DATA_SHA_MATCH", flush=True)


def check_data_or_smoke(cfg, a):
    if (Path(a.out) / SMOKE_MARK).exists():
        print("DATA_SHA_SKIPPED (smoke test on stand-ins)", flush=True)
    else:
        check_data(cfg, a)


def record_env(cfg, a, stage):
    """Record the library versions and the GPU of this stage next to the outputs. A difference from the frozen
    environment does not stop the run (Sec. 3: it is reported in Sec. 13); it is printed and written down."""
    import torch
    import transformers
    got = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    want = cfg["env"]
    diff = [k for k in ("torch", "transformers") if got[k] != want[k]]
    if "L4" not in got["gpu"]:
        diff.append("gpu")
    p = Path(a.out) / f"env_{stage}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stage": stage, "env": got, "differs_from_frozen": diff}, indent=1))
    print("ENV_MISMATCH" if diff else "ENV_MATCH", json.dumps(got), diff, flush=True)


def run_file(path, fn, items, lm=None):
    """Run fn over items and write the rows, the timing, and (for LLM stages) the prompt log of this file."""
    path = Path(path)
    start = len(lm.log) if lm is not None else 0
    t = time.monotonic()
    rows = []
    for it in items:
        r = fn(it)
        r["id"] = it["id"]
        rows.append(r)
    sec = time.monotonic() - t
    write_jsonl(path, rows)
    meta = {"n": len(rows), "seconds": sec, "parsed": sum(bool(r.get("parsed", True)) for r in rows)}
    if lm is not None:
        log = lm.log[start:]
        for l in log:
            l["file"] = path.stem
        write_jsonl(path.with_suffix(".log.jsonl"), log)
        meta.update(prompts=len(log), in_tokens=sum(l["in_tokens"] for l in log), out_tokens=sum(l["out_tokens"] for l in log))
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    print("DONE", path.name, json.dumps(meta), flush=True)
    return rows


def wic_inputs(a, d):
    return D.load_grain_inputs(a.grain) if d == GRAIN else D.load_wic_inputs(a.wic_zip, d)


# ---------- Step 1: transform ----------

def transform(cfg, a):
    from lm import ChatLM
    out = Path(a.out) / "transform"
    out.mkdir(parents=True, exist_ok=True)
    if a.limit:
        (Path(a.out) / SMOKE_MARK).write_text("stand-in smoke test; not the evaluation run\n")
    else:
        assert not (Path(a.out) / SMOKE_MARK).exists(), "this output folder holds a smoke test; use a fresh one"
    check_data_or_smoke(cfg, a)
    record_env(cfg, a, "transform")
    wic = {d: wic_inputs(a, d) for d in WIC_ALL}
    nli = D.load_chaos_inputs(a.chaos_zip)
    if a.limit:  # smoke test on train-side stand-ins only; never used on the evaluation data
        wic = {d: v[: a.limit] for d, v in wic.items()}
        nli = nli[: a.limit]
    for d, items in wic.items():
        if not done(out / f"{d}_inputs.jsonl"):
            write_jsonl(out / f"{d}_inputs.jsonl", items)
    if not done(out / "chaos_inputs.jsonl"):
        write_jsonl(out / "chaos_inputs.jsonl", nli)
    wn = check_wordnet(cfg, a.nltk)
    check_weights(cfg, a.model, "qwen")
    lm = ChatLM(a.model)
    for d, items in wic.items():
        for k in ("wic_para1", "wic_para2"):
            p = out / f"{d}_{k}.jsonl"
            if not done(p):
                t = tpl(cfg, k)
                run_file(p, lambda it: {"s1": M.wic_paraphrase(lm, t, it, 1, k), "s2": M.wic_paraphrase(lm, t, it, 2, k)}, items, lm)
    for k in ("nli_para1", "nli_para2"):
        p = out / f"chaos_{k}.jsonl"
        if not done(p):
            t = tpl(cfg, k)
            run_file(p, lambda it: {"premise": M.nli_paraphrase(lm, t, it["premise"], k),
                                    "hypothesis": M.nli_paraphrase(lm, t, it["hypothesis"], k)}, nli, lm)
    # Hard negatives for the check sheet (Sec. 4): candidates in a seeded order, generated until n_hardneg
    # parse or hardneg_max_candidates are tried. A shortfall is not an error: the sheet uses what parsed and
    # the shortfall is reported (check_sheet.build).
    pool_wic = [it for d in WIC_DATA for it in wic[d]]
    c = cfg["check"]
    for task, pool, fn in (("wic", pool_wic, lambda it: M.wic_hardneg(lm, tpl(cfg, "wic_hardneg"), it)),
                           ("nli", nli, lambda it: M.nli_hardneg(lm, tpl(cfg, "nli_hardneg"), it))):
        p = out / f"{task}_hardneg.jsonl"
        if done(p):
            continue
        order = np.random.default_rng(item_seed(task, "check", "hardneg")).permutation(len(pool))[: c["hardneg_max_candidates"]]
        rows = []
        for j in order:
            r = fn(pool[int(j)])
            r["id"] = pool[int(j)]["id"]
            rows.append(r)
            if sum(x["parsed"] for x in rows) >= c["n_hardneg"]:
                break
        write_jsonl(p, rows)
    # WordNet exclusions (Sec. 5) and NLI units (Sec. 2): decided here, before any reference is read.
    from inventory import senses
    excl = {d: {it["id"]: len(senses(wn, it["lemma"], it["pos"])) for it in items if len(senses(wn, it["lemma"], it["pos"])) < 2}
            for d, items in wic.items()}
    units = {it["id"]: D.nli_unit(it) for it in nli}
    (out / "wordnet_exclusions.json").write_text(json.dumps(excl, indent=1))
    (out / "nli_units.json").write_text(json.dumps(units, ensure_ascii=False))
    lemmas = {d: {it["lemma"] for it in items} for d, items in wic.items()}
    train = {x["lemma"] for x in read_jsonl(cfg["_root"] / cfg["train_data"]["wic"])}
    rep = {"wordnet_excluded": {d: len(v) for d, v in excl.items()}, "nli_units": len(set(units.values())),
           "train200_lemma_overlap": {d: len(train & lemmas[d]) for d in WIC_DATA},
           "hardneg_parsed": {t: sum(r["parsed"] for r in read_jsonl(out / f"{t}_hardneg.jsonl")) for t in ("wic", "nli")}}
    (out / "transform_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


def build_forms(trans_dir, data=WIC_ALL):
    """Rebuild all forms from the transform outputs. Paraphrases that did not parse give no form."""
    t = Path(trans_dir)
    forms = {}
    pick = lambda r: (r["s1"]["text"], r["s2"]["text"]) if r["s1"]["parsed"] and r["s2"]["parsed"] else None
    for d in data:
        if not (t / f"{d}_inputs.jsonl").exists():
            continue
        items = read_jsonl(t / f"{d}_inputs.jsonl")
        p1 = {r["id"]: r for r in read_jsonl(t / f"{d}_wic_para1.jsonl")}
        p2 = {r["id"]: r for r in read_jsonl(t / f"{d}_wic_para2.jsonl")}
        forms[d] = {it["id"]: D.wic_forms(it, pick(p1[it["id"]]), pick(p2[it["id"]])) for it in items}
    items = read_jsonl(t / "chaos_inputs.jsonl")
    p1 = {r["id"]: r for r in read_jsonl(t / "chaos_nli_para1.jsonl")}
    p2 = {r["id"]: r for r in read_jsonl(t / "chaos_nli_para2.jsonl")}
    side = lambda r, s: r[s]["text"] if r[s]["parsed"] else None
    forms["chaos"] = {it["id"]: D.nli_forms(it, side(p1[it["id"]], "premise"), side(p1[it["id"]], "hypothesis"),
                                            side(p2[it["id"]], "premise"), side(p2[it["id"]], "hypothesis")) for it in items}
    return forms


# ---------- Step 3: base models ----------

def base(cfg, a):
    from base_models import run_deberta, run_flan, run_roberta
    out = Path(a.out) / "base"
    out.mkdir(parents=True, exist_ok=True)
    assert done(Path(a.out) / "transform" / "transform_summary.json"), "run transform first (Sec. 0, Step 1)"
    for task in ("wic", "nli"):
        assert done(Path(a.out) / "check" / f"{task}_result.json"), "score the paraphrase check first (Sec. 0, Step 2)"
    record_env(cfg, a, "base")
    forms = build_forms(Path(a.out) / "transform")
    prompts = cfg["_root"] / "prompts"
    for name in ("deberta", "flan", "roberta"):
        check_weights(cfg, Path(a.models) / name, name)
    jobs = [(d, name, fn, D.WIC_S + D.WIC_B) for d in WIC_DATA for name, fn in (("deberta", run_deberta), ("flan", run_flan))]
    jobs.append((GRAIN, "deberta", run_deberta, D.WIC_S + D.WIC_B))
    jobs.append(("chaos", "roberta", run_roberta, D.NLI_S + D.NLI_B))
    for d, name, fn, form_names in jobs:
        for f in form_names:
            p = out / f"{d}_{name}_{f}.jsonl"
            if done(p):
                continue
            items = [fs[f] for fs in forms[d].values() if f in fs]
            t = time.monotonic()
            rows, skipped = fn(Path(a.models) / name, items, prompts)
            write_jsonl(p, rows)
            p.with_suffix(".meta.json").write_text(json.dumps({"n": len(rows), "skipped_too_long": skipped,
                                                               "seconds": time.monotonic() - t}))
            print("BASE", d, name, f, len(rows), "skipped", len(skipped), flush=True)
    if not done(out / "chaos_premise_tokens.json"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(Path(a.models) / "roberta")
        n = {i: len(tok.encode(fs["orig"]["premise"], add_special_tokens=False)) for i, fs in forms["chaos"].items()}
        (out / "chaos_premise_tokens.json").write_text(json.dumps(n))


# ---------- Step 5: structured methods and controls ----------

def structured(cfg, a):
    from inventory import DonorPool, inventory_A, inventory_iv
    from lm import ChatLM
    assert done(Path(a.out) / "counts.json"), "run eval_analysis.py count first (Sec. 0, Step 4)"
    out = Path(a.out) / f"structured_{a.model_name}"
    out.mkdir(parents=True, exist_ok=True)
    t = Path(a.out) / "transform"
    record_env(cfg, a, f"structured_{a.model_name}")
    wn = check_wordnet(cfg, a.nltk)
    check_weights(cfg, a.model, a.model_name)
    lm = ChatLM(a.model)
    pool = DonorPool(wn)
    m = cfg["nli_m"]
    for d in WIC_ALL:
        if d == GRAIN and a.model_name != "qwen":
            continue  # GrainWiC is run with the primary configuration only
        items = read_jsonl(t / f"{d}_inputs.jsonl")
        invA = {it["id"]: inventory_A(wn, d, it["id"], it["lemma"], it["pos"]) for it in items}
        keep = [it for it in items if invA[it["id"]] is not None]
        jobs = {"A": lambda it: M.wic_structured(lm, tpl(cfg, "wic_A"), it, invA[it["id"]], "A"),
                "iv": lambda it: M.wic_structured(lm, tpl(cfg, "wic_A"), it,
                                                  inventory_iv(wn, pool, d, it["id"], it["lemma"], it["pos"], invA[it["id"]]), "iv"),
                "ii": lambda it: M.wic_direct(lm, tpl(cfg, "wic_ii"), it),
                "iii": lambda it: M.wic_two_step(lm, M.split_templates(tpl(cfg, "wic_iii")), it)}
        for name, fn in jobs.items():
            p = out / f"{d}_{name}.jsonl"
            if not done(p):
                run_file(p, fn, keep, lm)
    nli = read_jsonl(t / "chaos_inputs.jsonl")
    premise = {x["id"]: x["premise"] for x in nli}
    p = out / "chaos_atoms.jsonl"
    if not done(p):
        run_file(p, lambda it: M.nli_atoms(lm, tpl(cfg, "nli_atoms"), it), nli, lm)
    atoms = {r["id"]: r for r in read_jsonl(p)}

    def at(i):  # an empty decomposition falls back to the premise; the item is then marked as not parsed
        r = atoms[i]
        return r["atoms"] or [premise[i]], r["parsed"]

    def nli_A(it):
        al, ok = at(it["id"])
        r = M.nli_structured(lm, tpl(cfg, "nli_A"), it, al, "A")
        r["parsed"] = r["parsed"] and ok
        return r

    # Donors for (iv) come from examples whose atoms were parsed, so that an unparsed donor does not drop the
    # receiving item from the comparison (Sec. 5 only asks for another example with a different premise).
    donors = [x for x in nli if atoms[x["id"]]["parsed"]]

    def nli_iv(it):
        donor = M.nli_donor("chaos", donors, it)
        al, ok = at(donor["id"])
        r = M.nli_structured(lm, tpl(cfg, "nli_A"), it, al, "iv")
        r["parsed"] = r["parsed"] and ok
        r["donor"] = donor["id"]
        return r

    ts_ii, ts_iii = M.split_templates(tpl(cfg, "nli_ii")), M.split_templates(tpl(cfg, "nli_iii"))
    assert m <= len(ts_ii) and 2 * math.ceil(m / 2) <= len(ts_iii)
    jobs = {"A": nli_A, "iv": nli_iv, "ii": lambda it: M.nli_direct(lm, ts_ii, it, m),
            "iii": lambda it: M.nli_two_step(lm, ts_iii, it, math.ceil(m / 2))}
    for name, fn in jobs.items():
        p = out / f"chaos_{name}.jsonl"
        if not done(p):
            run_file(p, fn, nli, lm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["transform", "base", "structured"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wic-zip")
    ap.add_argument("--chaos-zip")
    ap.add_argument("--grain")
    ap.add_argument("--model")
    ap.add_argument("--model-name", default="qwen")
    ap.add_argument("--models")
    ap.add_argument("--nltk")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: keep the first N items of each data set")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"transform": transform, "base": base, "structured": structured}[a.stage](cfg, a)
