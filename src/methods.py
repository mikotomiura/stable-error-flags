"""Structured methods, controls, and input transformations (preregistration v4, Secs. 4-5).

`lm` is an lm.ChatLM and templates are strings (common.load_template). Every function returns a dict with
the probabilities (P(same) or a distribution over the three NLI labels) and a `parsed` flag.
"""
from __future__ import annotations

import math
import re

import numpy as np

from common import count_word, item_seed

SPLIT = "\n=====\n"
WIC_LABELS = ["same", "different"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]
POS_NAME = {"N": "noun", "V": "verb"}
MAX_ATOMS = 8
WIC_GLOSS_TOKENS = 32
NLI_NOTES_TOKENS = 64


def split_templates(t: str):
    return t.split(SPLIT)


# ---------- WiC ----------

def wic_tokens(item, k):
    return item[f"sentence{k}"].split(" ")


def wic_word(item, k):
    return wic_tokens(item, k)[item[f"idx{k}"]]


def wic_marked(item, k):
    t = wic_tokens(item, k)
    i = item[f"idx{k}"]
    return " ".join(t[:i] + ["[[" + t[i] + "]]"] + t[i + 1:])


def wic_sense_dist(lm, tpl, item, k, definitions, tag):
    opts = definitions + ["None of the above"]
    options = "\n".join(f"{j + 1}. {d}" for j, d in enumerate(opts))
    text = tpl.format(lemma=item["lemma"], pos_name=POS_NAME[item["pos"]], sentence=wic_marked(item, k), options=options,
                      n_opts=len(opts), none_num=len(opts), none_num_minus=len(opts) - 1)
    return lm.choose(text, [str(j + 1) for j in range(len(opts))], max_new_tokens=4, tag=tag)


def wic_structured(lm, tpl, item, inv, tag="A"):
    """Shared procedure of A and (iv): P(same) = sum_k p1(k) * p2(k), excluding the "none of the above" option."""
    r1 = wic_sense_dist(lm, tpl, item, 1, inv["definitions"], tag)
    r2 = wic_sense_dist(lm, tpl, item, 2, inv["definitions"], tag)
    K = len(inv["definitions"])
    p_same = sum(r1["probs"][j] * r2["probs"][j] for j in range(K))
    return {"p_same": p_same, "p1": r1["probs"], "p2": r2["probs"], "parsed": r1["parsed"] and r2["parsed"],
            "synsets": inv["synsets"], "tolerance": inv.get("tolerance"),
            "greedy": [r1["greedy"], r2["greedy"]], "n_prompts": 2,
            "none_prob": [r1["probs"][K], r2["probs"][K]],
            "choice_correct_none": [r1["greedy_choice"] == K, r2["greedy_choice"] == K]}


def wic_direct(lm, tpl, item):
    """Control (ii): mean P(same) over the two sentence orders."""
    outs = []
    for a, b in ((1, 2), (2, 1)):
        text = tpl.format(lemma=item["lemma"], sentence1=wic_marked(item, a), sentence2=wic_marked(item, b))
        outs.append(lm.choose(text, WIC_LABELS, tag="ii"))
    return {"p_same": float(np.mean([o["probs"][0] for o in outs])), "parsed": all(o["parsed"] for o in outs),
            "greedy": [o["greedy"] for o in outs], "n_prompts": 2}


GLOSS_RE = re.compile(r"^\s*([12])\s*[:.)]\s*(.+?)\s*$")


def parse_glosses(text):
    got = {}
    for line in text.splitlines():
        m = GLOSS_RE.match(line)
        if m and m.group(1) not in got:
            got[m.group(1)] = m.group(2)
    return (got["1"], got["2"]) if set(got) >= {"1", "2"} else None


def truncate_tokens(lm, s, n):
    ids = lm.tok.encode(s, add_special_tokens=False)
    return lm.tok.decode(ids[:n]) if len(ids) > n else s


def wic_two_step(lm, tpls, item):
    """Control (iii): the first prompt elicits a free-text gloss for each occurrence (at most 32 tokens each);
    the second prompt shows the glosses and asks for the judgment."""
    t1, t2 = tpls
    s1, s2 = wic_marked(item, 1), wic_marked(item, 2)
    g = lm.generate(t1.format(lemma=item["lemma"], sentence1=s1, sentence2=s2), max_new_tokens=2 * WIC_GLOSS_TOKENS + 16, tag="iii-1")
    parsed_g = parse_glosses(g["text"])
    if parsed_g is None:
        gl1 = gl2 = "(no description)"
    else:
        gl1, gl2 = (truncate_tokens(lm, x, WIC_GLOSS_TOKENS) for x in parsed_g)
    o = lm.choose(t2.format(lemma=item["lemma"], sentence1=s1, sentence2=s2, gloss1=gl1, gloss2=gl2), WIC_LABELS, tag="iii-2")
    return {"p_same": o["probs"][0], "parsed": parsed_g is not None and o["parsed"], "step1": g["text"],
            "step1_parsed": parsed_g is not None, "greedy": o["greedy"], "n_prompts": 2}


def wic_paraphrase(lm, tpl, item, k, tag):
    word = wic_word(item, k)
    g = lm.generate(tpl.format(word=word, sentence=item[f"sentence{k}"], sentence_marked=wic_marked(item, k)),
                    max_new_tokens=96, tag=tag)
    return wic_check_output(g, word, source=item[f"sentence{k}"])


def same_text(a, b):
    return " ".join(a.split()).lower() == " ".join(b.split()).lower()


def wic_check_output(g, word, source=None):
    """Parse a generated WiC sentence (decision 4 of the prompt-development record): a single line, not cut
    off by the token limit, containing the original word form exactly once, and (when `source` is given)
    not identical to the source sentence. Templates may mark the target word with [[ ]] in the input; any
    markers copied into the output are removed before counting."""
    text = clean_one_line(g["text"])
    ok = text is not None and not g["hit_limit"]
    if text is not None:
        text = text.replace("[[", "").replace("]]", "")
    ok = ok and count_word(text, word) == 1 and not (source is not None and same_text(text, source))
    return {"text": text if text is not None else g["text"], "raw": g["text"], "parsed": bool(ok)}


def clean_one_line(s):
    """Extract a single-line output. Returns None if the output is empty or spans several non-empty lines."""
    lines = [l.strip() for l in s.strip().splitlines() if l.strip()]
    if len(lines) != 1:
        return None
    t = lines[0]
    t = re.sub(r"^(Rewritten sentence|Sentence|New sentence|Rewritten text|Text|Edited hypothesis|Hypothesis)\s*:\s*", "", t, flags=re.I)
    t = t.strip().strip('"').strip()
    return t or None


# ---------- NLI ----------

ATOM_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.+?)\s*$")


def parse_atoms(text):
    atoms = [m.group(1) for l in text.splitlines() if (m := ATOM_RE.match(l))]
    return atoms


def nli_atoms(lm, tpl, item):
    g = lm.generate(tpl.format(premise=item["premise"]), max_new_tokens=256, tag="atoms")
    atoms = parse_atoms(g["text"])
    return {"atoms": atoms[:MAX_ATOMS], "n_raw": len(atoms), "raw": g["text"],
            "parsed": len(atoms) >= 1 and not g["hit_limit"]}


def combine_atoms(cs, es):
    """Probabilistic version of the aggregation rule of [39]:
    P(contradiction) = 1 - prod(1 - c_i), P(entailment) = (1 - P(contradiction)) * (1 - prod(1 - e_i)),
    P(neutral) = the remainder."""
    pc = 1 - math.prod(1 - c for c in cs)
    pe = (1 - pc) * (1 - math.prod(1 - e for e in es))
    return [pe, max(0.0, 1 - pc - pe), pc]  # order of NLI_LABELS


def nli_structured(lm, tpl, item, atoms, tag="A"):
    rs = [lm.choose(tpl.format(atom=a, hypothesis=item["hypothesis"]), NLI_LABELS, tag=tag) for a in atoms]
    cs = [r["probs"][2] for r in rs]
    es = [r["probs"][0] for r in rs]
    return {"probs": combine_atoms(cs, es), "c": cs, "e": es, "parsed": bool(rs) and all(r["parsed"] for r in rs),
            "greedy": [r["greedy"] for r in rs], "n_prompts": len(rs)}


def nli_direct(lm, tpls, item, m):
    outs = [lm.choose(t.format(premise=item["premise"], hypothesis=item["hypothesis"]), NLI_LABELS, tag="ii") for t in tpls[:m]]
    return {"probs": list(np.mean([o["probs"] for o in outs], axis=0)), "parsed": all(o["parsed"] for o in outs),
            "per_template_parsed": [o["parsed"] for o in outs], "greedy": [o["greedy"] for o in outs], "n_prompts": len(outs)}


def nli_two_step(lm, tpls, item, n_pairs):
    outs, steps = [], []
    for j in range(n_pairs):
        t1, t2 = tpls[2 * j], tpls[2 * j + 1]
        g = lm.generate(t1.format(premise=item["premise"], hypothesis=item["hypothesis"]), max_new_tokens=NLI_NOTES_TOKENS, tag="iii-1")
        notes = " ".join(g["text"].split()) or "(no notes)"
        o = lm.choose(t2.format(premise=item["premise"], hypothesis=item["hypothesis"], notes=notes), NLI_LABELS, tag="iii-2")
        outs.append(o)
        steps.append({"notes": g["text"], "nonempty": bool(g["text"].strip()), "hit_limit": g["hit_limit"]})
    return {"probs": list(np.mean([o["probs"] for o in outs], axis=0)),
            "parsed": all(o["parsed"] for o in outs) and all(s["nonempty"] for s in steps),
            "steps": steps, "greedy": [o["greedy"] for o in outs], "n_prompts": 2 * n_pairs}


def nli_paraphrase(lm, tpl, text, tag):
    g = lm.generate(tpl.format(text=text), max_new_tokens=min(256, 2 * len(lm.tok.encode(text)) + 32), tag=tag)
    out = clean_one_line(g["text"])
    ok = out is not None and not g["hit_limit"] and not same_text(out, text)
    return {"text": out if out is not None else " ".join(g["text"].split()), "raw": g["text"], "parsed": ok}


def word_edit_distance(a, b):
    a = [w.strip(".,;:!?\"'").lower() for w in a.split()]
    b = [w.strip(".,;:!?\"'").lower() for w in b.split()]
    a = [w for w in a if w]
    b = [w for w in b if w]
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def nli_hardneg(lm, tpl, item):
    g = lm.generate(tpl.format(premise=item["premise"], hypothesis=item["hypothesis"]), max_new_tokens=96, tag="nli-hardneg")
    t = clean_one_line(g["text"])
    ok = t is not None and word_edit_distance(item["hypothesis"], t) == 1
    return {"text": t if t is not None else g["text"], "parsed": ok}


def wic_hardneg(lm, tpl, item, k=1):
    word = wic_word(item, k)
    g = lm.generate(tpl.format(word=word, sentence=item[f"sentence{k}"], sentence_marked=wic_marked(item, k)),
                    max_new_tokens=96, tag="wic-hardneg")
    return wic_check_output(g, word, source=item[f"sentence{k}"])


def nli_donor(data, items, item):
    """Control (iv) for NLI: draw, with the per-item seed, another example whose premise string differs."""
    pool = [x for x in items if x["premise"].strip() != item["premise"].strip()]
    rng = np.random.default_rng(item_seed(data, item["id"], "iv"))
    return pool[int(rng.integers(len(pool)))]
