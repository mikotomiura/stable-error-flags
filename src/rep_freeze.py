"""Freeze R of the replication study (rep_preregistration.md, Sec. 0): list the SHA-256 of every frozen file.

  python src/rep_freeze.py            -> rep_manifest.json (and prints its SHA-256, the single digest for Sec. 14)
  python src/rep_freeze.py --check    -> compares the files with an existing rep_manifest.json, writes nothing

The 35 files of freeze 2 (freeze2_manifest.json) are included unchanged and checked against freeze2_manifest.json.
Not frozen: MCL-WiC raw data (hashes are in rep_config.json; not redistributed), experiments/20261109-rep-pilot/private/
(real names and Drive ids; decisions of 2026-11-09), caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = ["docs/rep_preregistration.md", "docs/rep_rater_instructions.md", "docs/rep_rater_criteria.csv",
         "rep_config.json", "freeze2_manifest.json", "prompts/*.txt", "src/*.py", "tests/*.py",
         "experiments/20261108-rep-power/**/*",
         "experiments/20261108-mclwic-data/data.md", "experiments/20261108-mclwic-data/notes.md",
         "experiments/20261108-mclwic-data/scripts/*", "experiments/20261108-mclwic-data/results/*",
         "experiments/20261108-mclwic-data/prompts_trials/**/*",
         "experiments/20261109-rep-pilot/**/*"]
EXCLUDE = ("experiments/20261109-rep-pilot/private/",)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files():
    out = set()
    for g in GLOBS:
        for p in ROOT.glob(g):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and "__pycache__" not in rel and not rel.startswith(EXCLUDE):
                out.add(rel)
    f2 = json.loads((ROOT / "freeze2_manifest.json").read_text())["files"]
    for rel, h in f2.items():
        assert sha(ROOT / rel) == h, f"freeze 2 file changed: {rel}"
        out.add(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    man = {"files": {rel: sha(ROOT / rel) for rel in files()}}
    out = ROOT / "rep_manifest.json"
    if a.check:
        old = json.loads(out.read_text())["files"]
        diff = sorted(k for k in set(old) | set(man["files"]) if old.get(k) != man["files"].get(k))
        print(json.dumps({"changed_or_missing": diff}, ensure_ascii=False, indent=1))
        raise SystemExit(1 if diff else 0)
    out.write_text(json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_files": len(man["files"]), "manifest_sha256": sha(out)}, indent=1))


if __name__ == "__main__":
    main()
