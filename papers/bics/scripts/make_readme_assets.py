"""Figures for the public repository README (logo, overview diagram, animated detection curve).
  .venv-mac/bin/python -X utf8 papers/bics/scripts/make_readme_assets.py <out_dir>   (from the PJ root)

The animated GIF uses the released per-item outputs (papers/bics/data/item_outputs.csv; no text) for Study 1,
WiC test, DeBERTa x Qwen: the share of stable errors flagged when the top q of all ranked items is flagged, for A
(inventory-mediated judgement) and (i) confidence, against a random ranking. Ties at a boundary are split by item
order here (the paper splits them fractionally), so the curve is an illustration; the script prints the q-mean at
q = 0.1..0.5 so that it can be compared with the paper (0.509 and 0.303).
Palette: dataviz reference categorical slots 1-2 (#2a78d6 blue = A, #eb6834 orange = confidence), validated pair
(CVD dE 24.7, the same as Figure 1); random = muted grey, dashed.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

LOGO = """<svg xmlns="http://www.w3.org/2000/svg" width="780" height="120" viewBox="0 0 780 120" role="img" aria-label="stable-error-flags">
  <style>
    .ink{fill:#0b0b0b}.sub{fill:#52514e}.pole{stroke:#0b0b0b}
    @media (prefers-color-scheme: dark){.ink{fill:#f0efec}.sub{fill:#c3c2b7}.pole{stroke:#f0efec}}
  </style>
  <g transform="translate(12,10)">
    <line class="pole" x1="18" y1="8" x2="18" y2="100" stroke-width="5" stroke-linecap="round"/>
    <path d="M21 10 L86 10 L72 32 L86 54 L21 54 Z" fill="#2a78d6"/>
    <circle cx="44" cy="32" r="6" fill="#fcfcfb"/>
    <circle cx="62" cy="32" r="6" fill="#eb6834"/>
  </g>
  <text x="112" y="62" class="ink" font-family="Inter, Helvetica, Arial, sans-serif" font-size="40" font-weight="700">stable-error-flags</text>
  <text x="114" y="92" class="sub" font-family="Inter, Helvetica, Arial, sans-serif" font-size="16">flagging the errors a classifier repeats, with an inventory-mediated LLM judge</text>
</svg>
"""

OVERVIEW = """<svg xmlns="http://www.w3.org/2000/svg" width="920" height="300" viewBox="0 0 920 300" role="img" aria-label="Overview: stable errors and the four flags compared at the same flag rate">
  <style>
    text{font-family:Inter,Helvetica,Arial,sans-serif;fill:#0b0b0b}
    .muted{fill:#52514e}.box{fill:#ffffff;stroke:#d9d8d4}.arrow{stroke:#52514e;fill:none}
    @media (prefers-color-scheme: dark){text{fill:#f0efec}.muted{fill:#c3c2b7}.box{fill:#232322;stroke:#4a4a47}.arrow{stroke:#c3c2b7}}
  </style>
  <defs><marker id="h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#8a8984"/></marker></defs>
  <!-- step 1 -->
  <rect class="box" x="10" y="40" width="250" height="200" rx="14"/>
  <text x="30" y="70" font-size="17" font-weight="700">1  Stable errors</text>
  <text x="30" y="98" font-size="13" class="muted">The classifier gives the same wrong</text>
  <text x="30" y="116" font-size="13" class="muted">label on the original input,</text>
  <text x="30" y="134" font-size="13" class="muted">the swapped input and a paraphrase.</text>
  <rect x="30" y="160" width="62" height="26" rx="6" fill="#e34948" opacity="0.18"/><text x="38" y="178" font-size="12">original</text>
  <rect x="100" y="160" width="62" height="26" rx="6" fill="#e34948" opacity="0.18"/><text x="106" y="178" font-size="12">swapped</text>
  <rect x="170" y="160" width="70" height="26" rx="6" fill="#e34948" opacity="0.18"/><text x="176" y="178" font-size="12">paraphrase</text>
  <text x="30" y="216" font-size="12" class="muted">→ confidence stays high on them</text>
  <path class="arrow" d="M266 140 L314 140" stroke-width="2" marker-end="url(#h)"/>
  <!-- step 2 -->
  <rect class="box" x="320" y="10" width="300" height="280" rx="14"/>
  <text x="340" y="40" font-size="17" font-weight="700">2  Four flags, same budget</text>
  <rect x="340" y="60" width="12" height="12" rx="3" fill="#2a78d6"/><text x="360" y="71" font-size="14" font-weight="700">A</text><text x="378" y="71" font-size="13" class="muted">LLM picks WordNet senses</text>
  <text x="378" y="89" font-size="13" class="muted">(or atomic facts for NLI)</text>
  <rect x="340" y="110" width="12" height="12" rx="3" fill="#eb6834"/><text x="360" y="121" font-size="14" font-weight="700">(i)</text><text x="386" y="121" font-size="13" class="muted">classifier confidence</text>
  <rect x="340" y="150" width="12" height="12" rx="3" fill="#8a8984"/><text x="360" y="161" font-size="14" font-weight="700">(ii)</text><text x="390" y="161" font-size="13" class="muted">one more LLM judgement</text>
  <rect x="340" y="190" width="12" height="12" rx="3" fill="#8a8984"/><text x="360" y="201" font-size="14" font-weight="700">(iii)</text><text x="396" y="201" font-size="13" class="muted">two-step LLM judgement</text>
  <rect x="340" y="230" width="12" height="12" rx="3" fill="#8a8984"/><text x="360" y="241" font-size="14" font-weight="700">(iv)</text><text x="396" y="241" font-size="13" class="muted">fake inventory</text>
  <text x="340" y="272" font-size="12" class="muted">top q of items flagged, q = 0.1 … 0.5</text>
  <path class="arrow" d="M626 140 L674 140" stroke-width="2" marker-end="url(#h)"/>
  <!-- step 3 -->
  <rect class="box" x="680" y="40" width="230" height="200" rx="14"/>
  <text x="700" y="70" font-size="17" font-weight="700">3  Detection rate</text>
  <text x="700" y="98" font-size="13" class="muted">share of stable errors</text>
  <text x="700" y="116" font-size="13" class="muted">that are flagged</text>
  <text x="700" y="150" font-size="13">A − control,</text>
  <text x="700" y="168" font-size="13">paired, cluster bootstrap</text>
  <text x="700" y="202" font-size="12" class="muted">two studies; one preregistered</text>
  <text x="700" y="220" font-size="12" class="muted">replication (MCL-WiC)</text>
</svg>
"""


def scores(path):
    """Study 1, WiC test, DeBERTa x Qwen: score of A and of (i), and the stable-error mask."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["study"] == "1" and r["dataset"] == "WiC test" and r["classifier"] == "deberta" and r["llm"] == "qwen"]
    yhat = np.array([r["classifier_prediction"] for r in rows])
    p_same_A = np.array([float(r["p_same_A"]) for r in rows])
    pA = np.where(yhat == "T", p_same_A, 1 - p_same_A)
    p_i = np.array([float(r["p_classifier_prediction"]) for r in rows])
    stable = np.array([r["stable_error"] == "1" for r in rows])
    return 1 - pA, 1 - p_i, stable


def curve(score, stable, qs):
    order = np.argsort(-score, kind="stable")
    flagged_stable = np.cumsum(stable[order])
    n = len(score)
    return np.array([flagged_stable[max(int(round(q * n)) - 1, 0)] / stable.sum() if q > 0 else 0.0 for q in qs])


def gif(out, sA, si, st):
    qs = np.linspace(0, 0.5, 51)
    cA, ci = curve(sA, st, qs), curve(si, st, qs)
    q5 = [0.1, 0.2, 0.3, 0.4, 0.5]
    mA, mi = curve(sA, st, q5).mean(), curve(si, st, q5).mean()
    print(f"q-mean detection (illustration, order-split ties): A {mA:.3f}  confidence {mi:.3f}  (paper: 0.509, 0.303)")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
    frames = []
    tmp = Path(out).with_suffix("")
    tmp.mkdir(exist_ok=True)
    steps = list(range(1, len(qs) + 1)) + [len(qs)] * 12
    for k, upto in enumerate(steps):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(qs, qs, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.plot(qs[:upto], cA[:upto], color=BLUE, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot(qs[:upto], ci[:upto], color=ORANGE, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([qs[upto - 1]] * 2, [cA[upto - 1], ci[upto - 1]], s=46, c=[BLUE, ORANGE], edgecolors=SURFACE,
                   linewidths=2, zorder=4)
        ax.text(qs[upto - 1] + 0.008, cA[upto - 1], f"A  {cA[upto - 1]:.2f}", color=INK, va="center", fontsize=10)
        ax.text(qs[upto - 1] + 0.008, ci[upto - 1], f"confidence  {ci[upto - 1]:.2f}", color=INK, va="center", fontsize=10)
        ax.text(0.36, 0.33, "random", color=MUTED, fontsize=9, ha="left", va="top", rotation=0)
        ax.set_xlim(0, 0.62)
        ax.set_ylim(0, 0.85)
        ax.set_xlabel("flag budget q (share of all items flagged)")
        ax.set_ylabel("share of stable errors flagged")
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("WiC test · DeBERTa × Qwen · 208 stable errors", color=INK, fontsize=11, loc="left")
        fig.tight_layout()
        p = tmp / f"f{k:03d}.png"
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        frames.append(Image.open(p).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "logo.svg").write_text(LOGO, encoding="utf-8")
    (out / "overview.svg").write_text(OVERVIEW, encoding="utf-8")
    sA, si, st = scores(ROOT / "papers/bics/data/item_outputs.csv")
    print("items", len(st), "stable errors", int(st.sum()))
    gif(out / "detection.gif", sA, si, st)
    print("->", out)


if __name__ == "__main__":
    main()
