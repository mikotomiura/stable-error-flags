"""Draw the 200-item training-side samples (preregistration v4, Sec. 2).

Evaluation files are never parsed or displayed; only their SHA-256 digests are recorded.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np

SEED = 20261102
N = 200
EVAL_MEMBERS = ["dev/dev.data.txt", "dev/dev.gold.txt", "test/test.data.txt", "test/test.gold.txt"]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def wic(zip_path, out):
    z = zipfile.ZipFile(zip_path)
    # Evaluation files: hash the raw bytes only (no parsing, no display).
    member_sha = {m: sha_bytes(z.read(m)) for m in EVAL_MEMBERS}
    data = z.read("train/train.data.txt").decode("utf-8").rstrip("\n").split("\n")
    gold = z.read("train/train.gold.txt").decode("utf-8").rstrip("\n").split("\n")
    member_sha["train/train.data.txt"] = sha_bytes(z.read("train/train.data.txt"))
    member_sha["train/train.gold.txt"] = sha_bytes(z.read("train/train.gold.txt"))
    assert len(data) == len(gold) == 5428, (len(data), len(gold))
    idx = np.random.default_rng(SEED).choice(len(data), size=N, replace=False)
    rows = []
    for i in idx:
        lemma, pos, span, s1, s2 = data[i].split("\t")
        a, b = span.split("-")
        rows.append({"id": f"wic_train_{int(i)}", "row": int(i), "lemma": lemma, "pos": pos,
                     "idx1": int(a), "idx2": int(b), "sentence1": s1, "sentence2": s2,
                     "label": gold[i].strip()})
    write_jsonl(out / "wic_train200.jsonl", rows)
    return member_sha


def mnli(parquet_path, out):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path).to_pylist()
    idx = np.random.default_rng(SEED).choice(len(t), size=N, replace=False)
    names = {0: "entailment", 1: "neutral", 2: "contradiction"}  # GLUE ClassLabel order
    rows = [{"id": f"mnli_train_{int(i)}", "row": int(i), "idx": int(t[i]["idx"]),
             "premise": t[i]["premise"], "hypothesis": t[i]["hypothesis"],
             "label": names[t[i]["label"]]} for i in idx]
    write_jsonl(out / "mnli_train200.jsonl", rows)
    return len(t)


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wic-zip", required=True)
    ap.add_argument("--mnli-parquet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"seed": SEED, "n": N, "wic_member_sha256": wic(a.wic_zip, out), "mnli_train_rows": mnli(a.mnli_parquet, out)}
    for f in ("wic_train200.jsonl", "mnli_train200.jsonl"):
        rep[f + "_sha256"] = sha_bytes((out / f).read_bytes())
    print(json.dumps(rep, indent=1))
