"""Freeze 2 (preregistration v4, Sec. 0): copy the selected prompt versions into prompts/ and list the SHA-256
of every frozen file.

  python src/freeze2.py --config config.json [--write]

Without --write, only checks that prompts/ matches the selected versions and prints the manifest. With
--write, copies the selected versions first. The manifest goes to freeze2_manifest.json; its own SHA-256 is
the single digest written to Sec. 14 and to the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

FROZEN_GLOBS = ["config.json", "prompts/*.txt", "src/*.py", "tests/*.py",
                "experiments/20261102-power-rank-based/sim.py",
                "experiments/20261102-prompt-development/results/weights_sha256.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cfg_path = Path(a.config).resolve()
    root = cfg_path.parent
    cfg = json.loads(cfg_path.read_text())
    trials = root / cfg["prompt_trials"]
    for method, rel in cfg["prompts"].items():
        if method.startswith("base_"):
            continue
        src = trials / method / f"{cfg['versions'][method]}.txt"
        dst = root / rel
        if a.write:
            shutil.copyfile(src, dst)
        assert dst.exists() and dst.read_bytes() == src.read_bytes(), f"{dst} differs from {src}"
    files = sorted({p for g in FROZEN_GLOBS for p in root.glob(g) if p.is_file()})
    manifest = {"files": {str(p.relative_to(root)): sha(p) for p in files}}
    out = root / "freeze2_manifest.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"n_files": len(files), "manifest_sha256": sha(out)}, indent=1))


if __name__ == "__main__":
    main()
