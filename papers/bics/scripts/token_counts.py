"""Prompt and token counts of A relative to the controls, from the run logs' meta.json totals (descriptive; S2).
  python -X utf8 papers/bics/scripts/token_counts.py   (from the PJ root)
Output: papers/bics/tables/token_counts.json. For ChaosNLI, A includes the atom-splitting step (chaos_atoms)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = {"1": ROOT / "experiments/20261103-evaluation/results/eval", "2": ROOT / "experiments/20261109-rep-main/rep"}


def meta(st, llm, f):
    p = RUNS[st] / f"structured_{llm}" / f"{f}.meta.json"
    return json.loads(p.read_text()) if p.exists() else None


res = {}
for st, data in (("1", "wic_test"), ("1", "chaos"), ("2", "mcl")):
    for llm in ("qwen", "olmo"):
        A, ii, iii = meta(st, llm, f"{data}_A"), meta(st, llm, f"{data}_ii"), meta(st, llm, f"{data}_iii")
        at = meta(st, llm, f"{data}_atoms") if data == "chaos" else None
        a_p = A["prompts"] + (at["prompts"] if at else 0)
        a_in = A["in_tokens"] + (at["in_tokens"] if at else 0)
        res[f"study{st}:{data}/{llm}"] = {
            "prompts": {"A": a_p, "ii": ii["prompts"], "iii": iii["prompts"]},
            "prompt_ratio_A_ii": round(a_p / ii["prompts"], 2),
            "in_token_ratio_A_ii": round(a_in / ii["in_tokens"], 2),
            "in_token_ratio_A_iii": round(a_in / iii["in_tokens"], 2)}
out = ROOT / "papers/bics/tables/token_counts.json"
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
