"""Typeset the anonymised manuscript in the TMLR LaTeX style (submission mode: anonymous, no [accepted]/[preprint]).
  python3 -X utf8 papers/bics/scripts/build_latex.py   (from the PJ root; run build_manuscript.py and make_bib.py first)

Inputs: papers/bics/manuscript_anon.md (body), papers/bics/supplement_anon.md (appendix S1-S9, after the references,
as the TMLR author guide allows), papers/bics/latex/references.bib (from docs/references.md via make_bib.py).
Output: papers/bics/latex/main.tex; then `latexmk -pdf main.tex` in papers/bics/latex/ gives main.pdf.
The style files tmlr.sty, tmlr.bst, fancyhdr.sty are copied unmodified from github.com/JmlrOrg/tmlr-style-file.

The converter handles only the Markdown this manuscript uses (headings, paragraphs, bold/italic/code, bullet and
numbered lists, pipe tables with an italic caption paragraph before them, one figure). Table and figure numbers are
taken from the captions in the Markdown (so the numbers the text refers to are kept exactly), and the script stops
if a table caption or a citation number cannot be resolved. [n] citations become \\citep{ref<n>} (\\citet after "of").
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "papers/bics"
OUT = P / "latex"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
% Submission version: anonymous. Camera-ready: \usepackage[accepted]{tmlr}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{enumitem}
\usepackage{makecell}
\usepackage{adjustbox}
\usepackage{hyperref}
\hypersetup{colorlinks=true,allcolors=blue!50!black}
\usepackage{xurl}
\usepackage{textcomp}
\graphicspath{{../figures/}}
\setlist{itemsep=1pt,topsep=3pt}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\Urlmuskip=0mu plus 1mu

\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{†}{\textdagger}
\newunicodechar{ŷ}{\ensuremath{\hat{y}}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{…}{\ldots}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{Π}{\ensuremath{\Pi}}
\newunicodechar{₁}{\ensuremath{_1}}
\newunicodechar{₂}{\ensuremath{_2}}
\newunicodechar{⁻}{\ensuremath{^{-}}}
\newunicodechar{⁹}{\ensuremath{^{9}}}
\newunicodechar{⁶}{\ensuremath{^{6}}}
\newunicodechar{⌈}{\ensuremath{\lceil}}
\newunicodechar{⌉}{\ensuremath{\rceil}}
\newunicodechar{′}{\ensuremath{'}}
\newunicodechar{§}{\S}

\title{TITLE}
\author{Anonymous authors\\Paper under double-blind review}

\begin{document}
\maketitle
"""

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
CITED = set()


def esc(s):
    return "".join(SPECIAL.get(c, c) for c in s)


def inline(s):
    """Escape and convert inline Markdown. Code spans and URLs are protected first."""
    keep = []

    def stash(x):
        keep.append(x)
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"`([^`]+)`", lambda m: stash(r"\nolinkurl{" + m[1] + "}" if " " not in m[1]
                                             else r"\texttt{" + esc(m[1]) + "}"), s)
    s = re.sub(r"https?://[^\s|)]+[^\s|).,;]", lambda m: stash(r"\url{" + m[0] + "}"), s)
    # inline math written with underscores in the Markdown (P_A, score_X, c_i, A_Study1, Σ_k)
    s = s.replace("Σ_k", stash(r"$\sum_k$"))

    def sub_math(m):
        base, sub = m[1], m[2]
        b = r"\mathrm{" + base + "}" if len(base) > 1 else base
        sb = r"\text{" + sub + "}" if len(sub) > 1 else sub
        return stash("$" + b + "_{" + sb + "}$")

    s = re.sub(r"(?<![\w/.])([A-Za-z]+)_([A-Za-z0-9]+)(?![\w/.])", sub_math, s)

    def cite(m):
        nums = re.findall(r"\d+", m[2])
        CITED.update(nums)
        keys = ",".join(f"ref{n}" for n in nums)
        if m[1] == "of ":
            return "of " + stash(r"\citet{" + keys + "}")
        return stash(r"~\citep{" + keys + "}")

    s = re.sub(r"(of |\s?)((?:\[\d+\])+)", cite, s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = s.replace("\"", "''").replace("''", "\x01")  # quotes
    s = re.sub(r"(^|[\s(\[{])\x01", r"\1``", s).replace("\x01", "''")
    s = re.sub(r"(^|[\s(\[{])'", r"\1`", s)
    s = re.sub(r"\[(P)\]", r"{[}\1{]}", s)  # keep [P] from being read as an optional argument
    while "\x00" in s:
        s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m[1])], s)
    return s


INTERVAL = re.compile(r"^(−?\d[\d.,]*%?(?: †)?) \[(−?[\d.]+), (−?[\d.]+)\]( †)?$")


def wrap(text, n, valign="tl"):
    """Greedy line breaks at n characters inside a \\makecell (used for short columns)."""
    if len(text) <= n:
        return inline(text)
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return r"\makecell[" + valign + "]{" + r" \\ ".join(("{}" if x.startswith("[") else "") + inline(x) for x in lines) + "}"


def cell(c, stack, wrap_at):
    m = INTERVAL.match(c)
    if m and stack:  # point estimate on the first line, interval on the second
        return (r"\makecell[tl]{" + inline(m[1]) + r" \\ {[}" + inline(m[2]) + ", " + inline(m[3]) + "{]}"
                + inline(m[4] or "") + "}")
    if m:
        return inline(re.sub(r"(\d) \[(−?\d)", r"\1~[\2", c).replace(", ", ",~"))
    if wrap_at and re.fullmatch(r"[−\d.,%() †—–→x]+", c):  # numbers and arrows never wrap
        return inline(c)
    return wrap(c, wrap_at) if wrap_at else inline(c)


def table(lines, caption, label_num, env="table"):
    """Short columns: natural width, long cells wrapped at 18 characters, the whole table shrunk only if it is
    still wider than the text (adjustbox max width). Columns with cells over 40 characters: tabularx X columns."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    ncol = len(head)
    body = [r + [""] * (ncol - len(r)) for r in body]
    natural = sum(max(len(r[i]) for r in [head] + body) for i in range(ncol)) + 3 * ncol
    stack = natural > 100 and any(INTERVAL.match(c) for r in body for c in r)
    longest = [max(len(r[i]) for r in body) for i in range(ncol)]
    para = [i for i in range(ncol) if longest[i] > 40 and natural > 100]
    wrap_at = 18 if natural > 100 else 0
    numeric = lambda c: bool(re.fullmatch(r"[−\d.,%() †—–→x]+", c))
    est = sum(max([min(len(r[i]), 18) if wrap_at and not numeric(r[i]) else (14 if stack and INTERVAL.match(r[i]) else len(r[i]))
                   for r in body] + [min(len(head[i]), 12 if wrap_at else 99)]) for i in range(ncol)) + 2 * ncol
    size = r"\footnotesize" if est <= 125 else r"\scriptsize"
    if para:
        spec = "".join("Y" if i in para else "l" for i in range(ncol))
        open_, close = r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\end{tabularx}"
    else:
        open_ = r"\begin{adjustbox}{max width=\linewidth}" + "\n" + r"\begin{tabular}{" + "l" * ncol + "}"
        close = r"\end{tabular}" + "\n" + r"\end{adjustbox}"
    hw = lambda i, h: inline(h) if i in para or not wrap_at else wrap(h, 12, "bl")
    out = []
    if caption is not None:
        out += [r"\begin{" + env + "}[tbp]", r"\renewcommand{\the" + env + "}{" + label_num + "}",
                r"\caption{" + inline(caption) + "}", r"\centering"]
    else:
        out += [r"\begin{center}"]
    out += [size, r"\setlength{\tabcolsep}{3pt}", open_, r"\toprule",
            " & ".join(r"\textbf{" + hw(i, h) + "}" if h else "" for i, h in enumerate(head)) + r" \\", r"\midrule"]
    for r in body:
        out.append(" & ".join(cell(c, stack, 0 if i in para else wrap_at) for i, c in enumerate(r)) + r" \\")
    out += [r"\bottomrule", close]
    out += [r"\end{" + env + "}"] if caption is not None else [r"\end{center}"]
    return "\n".join(out)


def convert(md, part):
    lines = [l.rstrip("\r") for l in md.split("\n")]
    out, i, title, abstract = [], 0, None, None
    pending_caption = None
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1
            continue
        if l.startswith("## References"):
            break
        if l.startswith("# "):
            title = l[2:].strip()
            i += 1
            continue
        if l.startswith("## Abstract"):
            j = i + 1
            while not lines[j].strip():
                j += 1
            abstract = inline(lines[j])
            i = j + 1
            continue
        m = re.match(r"(#{2,4}) (S?\d+(?:\.\d+)*)\s+(.*)", l)
        if m:
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[len(m[1])]
            out.append(f"\\{cmd}{{{inline(m[3])}}}\\label{{sec:{m[2]}}}% {m[2]}")
            out.append(f"%CHECK {cmd} {m[2]}")
            i += 1
            continue
        if l.startswith("#"):
            sys.exit(f"unhandled heading: {l}")
        if l.startswith("!["):
            src = re.match(r"!\[[^]]*\]\(([^)]+)\)", l)[1]
            j = i + 1
            while not lines[j].strip():
                j += 1
            cap = re.match(r"\*Figure (\d+)\. (.*)\*$", lines[j].strip())
            if not cap:
                sys.exit("figure without caption")
            out.append("\n".join([r"\begin{figure}[tbp]", r"\centering",
                                  r"\includegraphics[width=\linewidth]{" + Path(src).stem + ".pdf}",
                                  r"\renewcommand{\thefigure}{" + cap[1] + "}",
                                  r"\caption{" + inline(cap[2]) + "}", r"\end{figure}"]))
            i = j + 1
            continue
        cap = re.match(r"\*Table (S?\d+(?:\.\d+)?)\. (.*)\*$", l.strip())
        if cap:
            pending_caption = (cap[1], cap[2])
            i += 1
            continue
        if l.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            if pending_caption:
                tex = table(block, pending_caption[1], pending_caption[0])
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].startswith("|"):  # a second table under the same caption (Table 5)
                    block2 = []
                    while j < len(lines) and lines[j].startswith("|"):
                        block2.append(lines[j])
                        j += 1
                    inner = table(block2, None, None).replace(r"\begin{center}", r"\par\medskip").replace(r"\end{center}", "")
                    tex = tex.replace(r"\end{table}", inner + "\n" + r"\end{table}")
                    i = j
                out.append(tex)
                pending_caption = None
            else:
                out.append(table(block, None, None))
            continue
        if pending_caption:
            sys.exit(f"table caption not followed by a table: Table {pending_caption[0]}")
        if re.match(r"- ", l) or re.match(r"\d+(?:–\d+)?\. ", l):
            kind = "itemize" if l.startswith("- ") else "enumerate"
            items = []
            pat = r"- |\d+(?:–\d+)?\. "
            while i < len(lines) and lines[i].strip() and (re.match(pat, lines[i]) or lines[i].startswith("  ")):
                if re.match(pat, lines[i]):
                    lab = re.match(pat, lines[i])[0].strip()
                    items.append([lab, lines[i][len(lab) + 1:]])
                else:
                    items[-1][1] += " " + lines[i].strip()
                i += 1
            # numbered items keep the label of the Markdown (e.g. "5–6.") instead of LaTeX's counter
            fmt = lambda lab, x: (r"\item " if lab == "-" else r"\item[{" + inline(lab) + "}] ") + inline(x)
            out.append("\\begin{" + kind + "}\n" + "\n".join(fmt(lab, x) for lab, x in items) + "\n\\end{" + kind + "}")
            continue
        para = [l]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"#|\||- |\d+(?:–\d+)?\. |!\[|\*Table ", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(inline(" ".join(x.strip() for x in para)))
    return title, abstract, "\n\n".join(out)


def check_numbering(tex, part):
    """The Markdown section numbers must equal LaTeX's automatic numbering (the text refers to them as plain text)."""
    sec = sub = 0
    for kind, num in re.findall(r"%CHECK (\w+) (\S+)", tex):
        n = num.lstrip("S")
        if kind == "section":
            sec, sub = sec + 1, 0
            exp = str(sec)
        else:
            sub += 1
            exp = f"{sec}.{sub}"
        if n != exp:
            sys.exit(f"{part}: heading {num} but LaTeX would number it {exp}")
    return re.sub(r"\n%CHECK .*", "", tex)


def main():
    title, abstract, body = convert((P / "manuscript_anon.md").read_text(encoding="utf-8"), "main")
    body = check_numbering(body, "main")
    _, _, supp = convert((P / "supplement_anon.md").read_text(encoding="utf-8"), "supplement")
    supp = check_numbering(supp, "supplement")
    bib = (OUT / "references.bib").read_text(encoding="utf-8")
    missing = [n for n in CITED if f"@article{{ref{n}," not in bib]
    if missing:
        sys.exit(f"citations not in references.bib: {missing}")
    tex = PREAMBLE.replace("TITLE", inline(title)) + "\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n" + body
    tex += "\n\n\\bibliography{references}\n\\bibliographystyle{tmlr}\n\n\\clearpage\n\\appendix\n"
    tex += r"\renewcommand{\thesection}{S\arabic{section}}" + "\n"
    tex += "\\section*{Supplementary material}\n\n" + supp + "\n\n\\end{document}\n"
    (OUT / "main.tex").write_text(tex, encoding="utf-8")
    print(f"main.tex written; {len(CITED)} distinct citations")


if __name__ == "__main__":
    main()
