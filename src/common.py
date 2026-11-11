"""Shared utilities: I/O, hashing, and per-item seeds (preregistration v4, Sec. 5)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PJ_SEED = 20261102


def item_seed(data: str, item_id: str, method: str) -> int:
    """Per-item seed (Sec. 5): int(sha256("20261102|<dataset>|<item id>|<method>").hexdigest()[:8], 16)."""
    return int(hashlib.sha256(f"{PJ_SEED}|{data}|{item_id}|{method}".encode()).hexdigest()[:8], 16)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_template(path) -> str:
    """Load a UTF-8 prompt template. A single trailing newline is dropped; fields use str.format syntax."""
    t = Path(path).read_text(encoding="utf-8")
    return t[:-1] if t.endswith("\n") else t


def fill(template: str, **kw) -> str:
    return template.format(**kw)


WORD_RE_CACHE: dict = {}


def count_word(text: str, word: str) -> int:
    """Count case-insensitive whole-word occurrences. Used for the exactly-once check of Sec. 4."""
    key = word.lower()
    if key not in WORD_RE_CACHE:
        WORD_RE_CACHE[key] = re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    return len(WORD_RE_CACHE[key].findall(text))


def find_word(text: str, word: str):
    """Return (start, end) of the unique occurrence, or None if the word does not occur exactly once."""
    count_word(text, word)
    ms = list(WORD_RE_CACHE[word.lower()].finditer(text))
    return (ms[0].start(), ms[0].end()) if len(ms) == 1 else None
