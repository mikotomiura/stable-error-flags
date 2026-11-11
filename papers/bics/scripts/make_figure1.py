"""Figure 1: forest plot of A minus each control (q-mean detection rate on stable errors, 95% lemma/premise-cluster
bootstrap interval), study 1 (main evaluation v4) next to study 2 (MCL-WiC replication) for the same base x LLM pair.
Reads only papers/bics/tables/table_main.csv (made by make_tables.py). Writes figures/fig1_forest.{pdf,png}.
  python -X utf8 papers/bics/scripts/make_figure1.py   (from the PJ root)
Palette: dataviz reference categorical slots 1-2 (#2a78d6, #eb6834), validated light mode (CVD dE 24.7);
study is also encoded by marker shape (circle / square) so identity is never colour alone."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TAB = ROOT / "papers/bics/tables/table_main.csv"
OUT = ROOT / "papers/bics/figures"
CTRL = [("i", "vs (i) confidence"), ("ii", "vs (ii) extra judgement"), ("iii", "vs (iii) two-step"),
        ("iv", "vs (iv) fake inventory")]
# row order (top to bottom): WiC-type pairs, then NLI. Each row holds study 1 and/or study 2.
ROWS = [("deberta", "qwen", "WiC-type", "DeBERTa × Qwen"),
        ("flan", "qwen", "WiC-type", "FLAN-T5 × Qwen"),
        ("deberta", "olmo", "WiC-type", "DeBERTa × OLMo"),
        ("flan", "olmo", "WiC-type", "FLAN-T5 × OLMo"),
        ("roberta", "qwen", "NLI", "RoBERTa × Qwen (NLI)"),
        ("roberta", "olmo", "NLI", "RoBERTa × OLMo (NLI)")]
STYLE = {"study1": dict(color="#2a78d6", marker="o", label="Study 1 (WiC test / ChaosNLI)", dy=-0.17),
         "study2": dict(color="#eb6834", marker="s", label="Study 2 (MCL-WiC, replication)", dy=0.17)}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def main():
    rows = list(csv.DictReader(open(TAB, encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.9), sharey=True, sharex=True)
    ypos = {(b, l): i for i, (b, l, _, _) in enumerate(ROWS)}
    for ax, (c, title) in zip(axes, CTRL):
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.axhline(3.5, color=GRID, lw=0.6, zorder=0)  # WiC-type above, NLI below
        for r in rows:
            st = STYLE[r["study"]]
            y = ypos[(r["base"], r["llm"])] + st["dy"]
            d, lo, hi = float(r[f"A-{c}"]), float(r[f"A-{c}_lo"]), float(r[f"A-{c}_hi"])
            ax.plot([lo, hi], [y, y], color=st["color"], lw=1.6, solid_capstyle="round", zorder=2)
            ax.plot(d, y, marker=st["marker"], ms=4.5, color=st["color"], mec="#fcfcfb", mew=0.8, zorder=3)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.2, 0.45)
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels(["", "0", "", ".2", "", ".4"])
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([lab for *_, lab in ROWS])
    axes[0].invert_yaxis()
    fig.supxlabel("A − control: difference in q-mean detection rate of stable errors (95% cluster-bootstrap interval)",
                  fontsize=7.5, color=INK, y=0.02)
    handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], lw=1.6, ms=4.5, label=s["label"])
               for s in STYLE.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_forest.{ext}", dpi=300, facecolor="#fcfcfb")
    print("->", OUT / "fig1_forest.pdf")


if __name__ == "__main__":
    main()
