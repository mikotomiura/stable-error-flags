"""Model download at pinned revisions, weight digests, and base-model inference (preregistration v4, Sec. 3).

- fetch: download the five models at their pinned revisions and record the SHA-256 of every file. The
  DeBERTa and FLAN weights are checked against the digests recorded in the earlier study (R3).
- Base-model inputs follow the R3 formats (prompts/base_*.txt). DeBERTa, FLAN and RoBERTa run in fp32
  without truncation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from common import load_template, read_jsonl, sha_file, write_jsonl

PINS = {
    "deberta": ("Deehan1866/finetuned-deberta-wic-actual", "1e98ca77f359b7502bb168efaf18748026027e61"),
    "flan": ("google/flan-t5-base", "0ff14384284b4f8d1dbead09ccc58a1b3ecc8e46"),
    "roberta": ("FacebookAI/roberta-large-mnli", "2a8f12d27941090092df78e4ba6f0928eb5eac98"),
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "olmo": ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),
}
# Digests recorded in R3 (handoff/quick_reference/implementation/model_pins.json)
R3_DEBERTA_WEIGHT = "def4d035712fbbd7c56b5d3e740ef8ef88cf722b53e18e22f186a057005c4af8"
R3_FLAN_WEIGHT = "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8"
ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "merges.txt", "vocab.*", "tokenizer*"]


def fetch(root, names):
    from huggingface_hub import snapshot_download
    rep = {}
    for n in names:
        repo, rev = PINS[n]
        d = Path(snapshot_download(repo, revision=rev, allow_patterns=ALLOW, local_dir=Path(root) / n))
        files = {p.name: sha_file(p) for p in sorted(d.iterdir()) if p.is_file()}
        rep[n] = {"repo": repo, "revision": rev, "files_sha256": files}
        if n == "deberta":
            assert files["model.safetensors"] == R3_DEBERTA_WEIGHT, "DeBERTa weights differ from R3"
        if n == "flan":
            assert files["model.safetensors"] == R3_FLAN_WEIGHT, "FLAN weights differ from R3"
        print("FETCHED", n, flush=True)
    return rep


def verify_weights(model_dir, name, record_path):
    """Check every file of a downloaded model against the digests recorded at the first fetch."""
    want = json.loads(Path(record_path).read_text())[name]["files_sha256"]
    got = {p.name: sha_file(p) for p in sorted(Path(model_dir).iterdir()) if p.is_file() and p.name in want}
    assert got == want, f"{name}: weights differ from {record_path}"


def wic_tgt(item, k, left, right):
    t = item[f"sentence{k}"].split(" ")
    i = item[f"idx{k}"]
    return " ".join(t[:i] + [left + t[i] + right] + t[i + 1:])


def run_deberta(path, items, prompts_dir, device="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tpl = load_template(Path(prompts_dir) / "base_deberta.txt").split("\n=====\n")
    tok = AutoTokenizer.from_pretrained(path)
    assert "<tgt>" in tok.get_vocab() and "</tgt>" in tok.get_vocab()
    model = AutoModelForSequenceClassification.from_pretrained(path).float().to(device).eval()
    rows, skipped = [], []
    for it in items:
        a = tpl[0].format(sentence1_tgt=wic_tgt(it, 1, "<tgt>", "</tgt>"), sentence2_tgt=wic_tgt(it, 2, "<tgt>", "</tgt>"))
        b = tpl[1].format(sentence1_tgt=wic_tgt(it, 1, "<tgt>", "</tgt>"), sentence2_tgt=wic_tgt(it, 2, "<tgt>", "</tgt>"))
        enc = tok(a, b, return_tensors="pt", truncation=False)
        if enc["input_ids"].shape[-1] > 256:  # R3 limit; no truncation. The form counts as not transformable.
            skipped.append(it["id"])
            continue
        with torch.inference_mode():
            l0, l1 = model(**{k: v.to(device) for k, v in enc.items()}).logits[0].float().cpu().tolist()
        p = 1 / (1 + math.exp(-(l1 - l0)))  # class1 = same
        rows.append({"id": it["id"], "p_same": p, "pred": "T" if l1 >= l0 else "F"})
    return rows, skipped


def run_flan(path, items, prompts_dir, device="cuda"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tpl = load_template(Path(prompts_dir) / "base_flan.txt")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).float().to(device).eval()
    labels = {a: tok(text_target=a, return_tensors="pt")["input_ids"].to(device) for a in ("No", "Yes")}
    rows, skipped = [], []
    for it in items:
        text = tpl.format(lemma=it["lemma"], sentence1_mark=wic_tgt(it, 1, "[[", "]]"), sentence2_mark=wic_tgt(it, 2, "[[", "]]"))
        enc = tok(text, return_tensors="pt", truncation=False)
        if enc["input_ids"].shape[-1] > 512:
            skipped.append(it["id"])
            continue
        ll = {}
        with torch.inference_mode():
            for a, lab in labels.items():
                lg = model(**{k: v.to(device) for k, v in enc.items()}, labels=lab).logits.float()
                ll[a] = float(torch.log_softmax(lg, -1).gather(2, lab.unsqueeze(-1)).squeeze(-1).sum())
        s = ll["Yes"] - ll["No"]
        rows.append({"id": it["id"], "p_same": 1 / (1 + math.exp(-max(-700, min(700, s)))), "pred": "T" if s >= 0 else "F"})
    return rows, skipped


def run_roberta(path, items, prompts_dir, device="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tpl = load_template(Path(prompts_dir) / "base_roberta.txt").split("\n=====\n")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).float().to(device).eval()
    id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
    rows, skipped = [], []
    for it in items:
        enc = tok(tpl[0].format(premise=it["premise"], hypothesis=it["hypothesis"]),
                  tpl[1].format(premise=it["premise"], hypothesis=it["hypothesis"]), return_tensors="pt", truncation=False)
        if enc["input_ids"].shape[-1] > 512:
            skipped.append(it["id"])
            continue
        with torch.inference_mode():
            pr = torch.softmax(model(**{k: v.to(device) for k, v in enc.items()}).logits[0].float(), -1).cpu().tolist()
        probs = {id2[j]: pr[j] for j in range(len(pr))}
        rows.append({"id": it["id"], "probs": probs, "pred": max(probs, key=probs.get)})
    return rows, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "train-base"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--names", default="deberta,flan,roberta,qwen,olmo")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--wic", default=None)
    ap.add_argument("--mnli", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "fetch":
        (out / "weights_sha256.json").write_text(json.dumps(fetch(a.root, a.names.split(",")), indent=1))
    else:
        res = {}
        wic, nli = read_jsonl(a.wic), read_jsonl(a.mnli)
        for name, fn, items in (("deberta", run_deberta, wic), ("flan", run_flan, wic), ("roberta", run_roberta, nli)):
            t = time.monotonic()
            rows, skipped = fn(Path(a.root) / name, items, a.prompts)
            res[name] = {"seconds_per_item": (time.monotonic() - t) / len(items), "skipped_too_long": len(skipped)}
            write_jsonl(out / f"base_{name}.jsonl", rows)
        lab_w = {x["id"]: x["label"] for x in wic}
        lab_n = {x["id"]: x["label"] for x in nli}
        for name in ("deberta", "flan"):
            rows = read_jsonl(out / f"base_{name}.jsonl")
            res[name]["accuracy_report_only"] = sum(r["pred"] == lab_w[r["id"]] for r in rows) / len(rows)
        rows = read_jsonl(out / "base_roberta.jsonl")
        res["roberta"]["accuracy_report_only"] = sum(r["pred"] == lab_n[r["id"]] for r in rows) / len(rows)
        # Direction of baseline (b-yhat): the predicted label with the most errors (Sec. 5).
        err = {}
        for r in rows:
            if r["pred"] != lab_n[r["id"]]:
                err[r["pred"]] = err.get(r["pred"], 0) + 1
        res["roberta"]["errors_by_pred"] = err
        (out / "base_train_summary.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
