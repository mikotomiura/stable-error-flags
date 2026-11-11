"""Render the tables and the reference list of the manuscript from their sources and expand the INCLUDE markers.
  python -X utf8 papers/bics/scripts/build_manuscript.py   (from the PJ root)
Inputs: tables/table_main.csv, tables/table2_flow.csv, tables/table2_projection.csv, tables/posthoc_round1.json,
experiments/20261110-adjudication/results/analysis.json, docs/references.md.
Outputs: tables/table1.md (main results; called Table 2 in the text), tables/table2.md (item flow; Table 5), tables/table_share.md (Table 3), tables/table_adjudication.md (Table 4), tables/tableS8.md, tables/tableS9.md,
tables/references.md, manuscript_built.md and supplement_built.md (if supplement.md exists)."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "papers/bics"
TAB = P / "tables"
NAMES = {"deberta": "DeBERTa", "flan": "FLAN-T5", "roberta": "RoBERTa", "qwen": "Qwen", "olmo": "OLMo"}
STATUS = {"H1/H2 (exploratory: check failed)": "H1/H2 cell; exploratory (check failed)",
          "H3 (exploratory: check failed)": "H3 cell; exploratory (check failed)",
          "H1/H2 (aborted: 91 < 100)": "H1/H2 cell; aborted (91 < 100)",
          "H1b (aborted)": "H1b cell; aborted", "secondary": "secondary"}


def rows(name):
    return list(csv.DictReader(open(TAB / name, encoding="utf-8")))


def ci(r, c):
    f = lambda x: f"{float(x):.3f}".replace("-", "−")
    return f"{f(r[f'A-{c}'])} [{f(r[f'A-{c}_lo'])}, {f(r[f'A-{c}_hi'])}]"


def dag(r):
    return " †" if r["llm"] == "olmo" and r["data"] != "ChaosNLI" else ""


def table_results():
    out = ["*Table 2. Difference in detection rate of stable errors between A and each control (q-mean over "
           "q = 0.1–0.5; 95% cluster-bootstrap interval). n: stable errors (independent units); n can be smaller than in Table 5 "
           "because items on which a primary method could not be parsed are excluded from the comparison (S2). Status: Section 2.4. "
           "Detection rates of each method, all-error results and other datasets are in S4. †: on WiC-type data OLMo's (ii) and "
           "(iii) predict 'different' on more than 99% of items and are not working judges (S8, P8); these differences are "
           "not evidence about the structure.*", "",
           "| Study | Data | Classifier × LLM | Status | n (units) | A − (i) | A − (ii) | A − (iii) | A − (iv) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows("table_main.csv"):
        out.append(f"| {r['study'][-1]} | {r['data']} | {NAMES[r['base']]} × {NAMES[r['llm']]} | {STATUS[r['status']]} | "
                   f"{r['n_stable']} ({r['units']}) | {ci(r, 'i')} | {ci(r, 'ii')}{dag(r)} | {ci(r, 'iii')}{dag(r)} | {ci(r, 'iv')} |")
    return "\n".join(out) + "\n"


def table_flow():
    fl = rows("table2_flow.csv")
    pj = rows("table2_projection.csv")
    n = lambda x: f"{int(x):,}"
    out = ["*Table 5. From input items to stable errors (top), and the factors of the preregistered expectation for "
           "Study 2 (bottom). Excluded items stay in the ranking of every method but not in the error sets. "
           "Stop rule: at least 100 stable errors and 50 units for the primary classifier. All MCL-WiC nouns and verbs have at least two WordNet 3.0 senses, hence the 0.*", "",
           "| Study | Data | Classifier | Items | < 2 WordNet senses | Paraphrase not formed | Rejected by check | "
           "Reference tie | Errors (stable + unstable) | Stable errors (units) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in fl:
        out.append(f"| {r['study'][-1]} | {r['data']} | {NAMES[r['base']]} | {n(r['items'])} | {n(r['wordnet_lt2_senses'])} | "
                   f"{n(r['paraphrase_not_formed'])} | {n(r['excluded_by_human_check'])} | {n(r['reference_tie'])} | "
                   f"{n(r['errors_classified'])} | {n(r['stable_errors'])} ({n(r['units'])}) |")
    out += ["", "| Study 2 factor | DeBERTa: projected → observed (ratio) | FLAN-T5: projected → observed (ratio) |",
            "|---|---|---|"]
    by = {}
    for r in pj:
        by.setdefault(r["factor"], {})[r["base"]] = r
    for f, d in by.items():
        fmt = lambda x: x if f.startswith("stable errors") else f"{float(x):.3f}"
        cell = lambda r: f"{fmt(r['projected'])} → {fmt(r['actual'])} ({float(r['ratio']):.2f})"
        out.append(f"| {f} | {cell(d['deberta'])} | {cell(d['flan'])} |")
    return "\n".join(out) + "\n"


def references(text):
    cited = sorted({int(m) for g in re.findall(r"\[(\d+(?:\]\[\d+)*)\]", text) for m in re.findall(r"\d+", g)})
    ref = {}
    for line in (ROOT / "docs/references.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \[(\d+)\] \| [^|]+ \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            venue = re.sub(r"（[^）]*）", "", m.group(3)).strip()  # drop Japanese bookkeeping notes
            title = re.sub(r"（[^）]*）", "", m.group(2)).strip()
            ref[int(m.group(1))] = f"[{m.group(1)}] {title}. {venue}. {m.group(4).strip()}"
    missing = [c for c in cited if c not in ref]
    assert not missing, f"cited but not in docs/references.md: {missing}"
    return "\n".join(ref[c] + "  " for c in cited) + "\n"


def table_s4():
    import json
    A = {"1": json.loads((ROOT / "experiments/20261103-evaluation/results/eval/analysis.json").read_text(encoding="utf-8")),
         "2": json.loads((ROOT / "experiments/20261109-rep-main/rep/analysis.json").read_text(encoding="utf-8"))}
    data = {"wic_test": "WiC test", "wic_dev": "WiC dev", "grain": "GrainWiC coarse", "chaos": "ChaosNLI",
            "mcl": "MCL-WiC dev+test"}
    f = lambda x: f"{x:.3f}".replace("-", "−")
    iv = lambda d: f"{f(d['d'])} [{f(d['lo'])}, {f(d['hi'])}]"
    out = ["*Table S4.1. Detection rate (q-mean) of stable errors for every method, and secondary differences. "
           "All-error: A − (i) on all errors, stable or not. (i)+A: mean of the confidence rank and the A rank, minus (i). "
           "A − (b-K): A minus a score built only from ŷ and the number of senses (WiC only).*", "",
           "| Study | Data | Classifier × LLM | n (units) | A | (i) | (ii) | (iii) | (iv) | All-error A − (i) | (i)+A − (i) | A − (b-K) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, a in A.items():
        for k, r in a["results"].items():
            if not isinstance(r, dict) or "detection" not in r:
                continue
            d0, base, llm = k.split("/")
            det = r["detection"]
            bk = r.get("diff_secondary", {}).get("b-K")
            out.append(f"| {st} | {data[d0]} | {NAMES[base]} × {NAMES[llm]} | {r['n_stable']} ({r['units_stable']}) | "
                       + " | ".join(f(det[m]) for m in ("A", "i", "ii", "iii", "iv")) + " | "
                       + f"{iv(r['all_errors_A_minus_i'])} | {iv(r['diff_i_plus_A_vs_i'])} | {iv(bk) if bk else '—'} |")
    return "\n".join(out) + "\n"


CELLS = [("1", "wic_test", "deberta", "qwen"), ("1", "wic_test", "flan", "qwen"), ("1", "wic_test", "deberta", "olmo"),
         ("2", "mcl", "deberta", "qwen"), ("2", "mcl", "flan", "qwen"), ("2", "mcl", "deberta", "olmo"),
         ("2", "mcl", "flan", "olmo")]
DATA = {"wic_test": "WiC test", "mcl": "MCL-WiC dev+test"}
fmt = lambda x: f"{x:.3f}".replace("-", "−")
fmt2 = lambda x: f"{x:.2f}".replace("-", "−")


def posthoc():
    import json
    return json.loads((TAB / "posthoc_round1.json").read_text(encoding="utf-8"))


def posthoc2():
    import json
    return json.loads((TAB / "posthoc_round2.json").read_text(encoding="utf-8"))


def table_share():
    """Table 3 (main text): share of A - (i) that (ii) already reaches, raw and after the label-free bias correction [P]."""
    p = posthoc()["P2_share"]
    q = posthoc2()["Q1_debiased"]
    main = {(r["study"][-1], r["base"], r["llm"]): r for r in rows("table_main.csv")}
    iv = lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]"
    out = ["*Table 3. How much of A's advantage over confidence an unstructured second judgement already reaches [P] "
           "(S8). Share = ((ii) − (i)) / (A − (i)) on the same stable errors (95% cluster-bootstrap interval; not bounded "
           "by 0 and 1). Debiased (ii): the same judge with its log-odds shifted by one constant per configuration so that "
           "it predicts 'different' as often as A does (no labels used). †: OLMo's (ii) predicts 'different' on more "
           "than 99% of items and is near chance as a judge (S8); these rows are not evidence about the structure.*", "",
           "| Study | Data | Classifier × LLM | A − (i) | Share of (ii) | A − (ii) | Share of debiased (ii) | A − debiased (ii) |",
           "|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        r = main[(st, b, l)]
        c = f"study{st}:{d}/{b}/{l}"
        s, sd = p[c]["ii"], q[c]["share_ii_deb"]
        dag = " †" if l == "olmo" else ""
        out.append(f"| {st} | {DATA[d]} | {NAMES[b]} × {NAMES[l]}{dag} | {fmt(float(r['A-i']))} | "
                   f"{fmt2(s['share'])} [{fmt2(s['lo'])}, {fmt2(s['hi'])}] | "
                   f"{fmt(float(r['A-ii']))} [{fmt(float(r['A-ii_lo']))}, {fmt(float(r['A-ii_hi']))}] | "
                   f"{fmt2(sd['share'])} [{fmt2(sd['lo'])}, {fmt2(sd['hi'])}] | {iv(q[c]['A_minus_ii_deb'])} |")
    return "\n".join(out) + "\n"


def posthoc3():
    import json
    return json.loads((TAB / "posthoc_round3.json").read_text(encoding="utf-8"))


def table_s8_round3():
    """Tables S8.8-S8.11 (round-3 post-hoc analyses)."""
    q = posthoc3()
    iv = lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]"
    out = ["*Table S8.8. Sensitivity of the debiased controls to the target share of 'different' (A's share as in "
           "Table 3, 0.5, the classifier's own share), and stratified flags (top q within each predicted-label stratum) [P].*", "",
           "| Study | Classifier × LLM | Target: A | 0.5 | classifier | A − debiased (iii), target A | A − (ii), stratified flags | A − (iii), stratified flags |",
           "|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        c = f"study{st}:{d}/{b}/{l}"
        v, s = q["R1_debias_sensitivity"][c], q["R1_stratified"][c]
        dg = " †" if l == "olmo" else ""
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{dg} | {iv(v['A_minus_ii_deb_A'])} | {iv(v['A_minus_ii_deb_0.5'])} | "
                   f"{iv(v['A_minus_ii_deb_classifier'])} | {iv(v['A_minus_iii_deb_A'])} | {iv(s['A_minus_ii'])} | {iv(s['A_minus_iii'])} |")
    out += ["", "*Table S8.9. Using confidence and A together at the same flag budget [P]: union (flags taken alternately "
            "from A's and (i)'s rankings) and mean of ranks, against A alone (detection of stable errors).*", "",
            "| Study | Classifier × LLM | A | (i) | Union | Mean of ranks | Union − A | Union − (i) |", "|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        v = q["R2_union"][f"study{st}:{d}/{b}/{l}"]
        e = v["detection"]
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]} | {e['A']:.3f} | {e['i']:.3f} | {e['union_A_i']:.3f} | "
                   f"{e['mean_rank_i_A']:.3f} | {iv(v['union_minus_A'])} | {iv(v['union_minus_i'])} |")
    out += ["", "*Table S8.10. AUROC of each score for stable errors against all other ranked items [P], and the "
            "Study 1 − Study 2 difference of A − (ii) on this scale (independent bootstraps per study).*", "",
            "| Study | Classifier × LLM | A | (i) | (ii) | (iii) | A − (ii) |", "|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        v = q["R3_auroc"][f"study{st}:{d}/{b}/{l}"]
        a = v["auroc"]
        dg = " †" if l == "olmo" else ""
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{dg} | {a['A']:.3f} | {a['i']:.3f} | {a['ii']:.3f} | {a['iii']:.3f} | {iv(v['A_minus_ii'])} |")
    out += ["", "| Classifier × LLM | Study 1 − Study 2, AUROC A − (ii) |", "|---|---|"]
    for k, v in q["R3_between_study"].items():
        b, l = k.split("/")
        out.append(f"| {NAMES[b]} × {NAMES[l]} | {fmt(v['study1_minus_study2'])} [{fmt(v['lo'])}, {fmt(v['hi'])}] |")
    out += ["", "*Table S8.11. ChaosNLI: the LLM judges as classifiers (accuracy; share of each predicted label) and "
            "AURC over abstention rates 0–0.95 [P].*", "",
            "| LLM | Judge | Accuracy | entailment | neutral | contradiction |", "|---|---|---|---|---|---|"]
    for k, v in q["R4_nli"].items():
        l = k.split("/")[-1]
        for m in ("A", "ii", "iii"):
            s = v[m]["label_share"]
            out.append(f"| {NAMES[l]} | ({m}) | {v[m]['accuracy']:.3f} | {s['entailment']:.2f} | {s['neutral']:.2f} | {s['contradiction']:.2f} |")
        s = v["reference_label_share"]
        out.append(f"| {NAMES[l]} | reference | — | {s['entailment']:.2f} | {s['neutral']:.2f} | {s['contradiction']:.2f} |")
    out += ["", "| LLM | AURC (i) | AURC A | AURC (i)+A |", "|---|---|---|---|"]
    for k, v in q["R4_nli"].items():
        a = v["aurc"]
        out.append(f"| {NAMES[k.split('/')[-1]]} | {a['i']:.3f} | {a['A']:.3f} | {a['i+A']:.3f} |")
    return "\n".join(out) + "\n"


def DG(l):
    """† on OLMo rows of WiC-type tables whose columns involve (ii) or (iii) (not working judges; Table 2)."""
    return " †" if l == "olmo" else ""


def posthoc4():
    import json
    return json.loads((TAB / "posthoc_round4.json").read_text(encoding="utf-8"))


def table_s8_round4():
    """Tables S8.12-S8.14 (round-4 post-hoc analyses)."""
    q2, q3, q4 = posthoc2(), posthoc3(), posthoc4()
    iv = lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]"
    out = ["*Table S8.12. AURC over abstention rates 0–0.95 (lower is better) on all ranked items with a reference, "
           "with 95% unit-bootstrap intervals of the differences (flags fixed, as in the frozen risk–coverage analysis) [P].*", "",
           "| Study | Classifier × LLM | AURC (i) | AURC A | AURC (i)+A | A − (i) | (i)+A − A |", "|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        c = f"study{st}:{d}/{b}/{l}"
        v, w = q2["Q4_combined"][c], q4["S2_aurc_iA_minus_A"][c]
        a = v["aurc"]
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]} | {a['i']:.3f} | {a['A']:.3f} | {a['i+A']:.3f} | "
                   f"{iv(v['aurc_A_minus_i'])} | {iv(w['iA_minus_A'])} |")
    for k, v in q3["R4_nli"].items():
        l = k.split("/")[-1]
        a, w = v["aurc"], q4["S2_aurc_iA_minus_A"][k]
        out.append(f"| 1 | ChaosNLI RoBERTa × {NAMES[l]} | {a['i']:.3f} | {a['A']:.3f} | {a['i+A']:.3f} | "
                   f"{iv(v['aurc_A_minus_i'])} | {iv(w['iA_minus_A'])} |")
    out += ["", "*Table S8.13. Confidence on stable errors against correctly classified items only (unstable errors "
            "left out) [P]: AUROC of (i) and A, (i) − 0.5, and (i)'s detection rate minus the random expectation 0.30 "
            "(unit bootstrap). (i) does not depend on the LLM; the Qwen rows are shown.*", "",
            "| Study | Classifier | Correct items | AUROC (i) | AUROC A (Qwen) | (i) − 0.5 | Detection (i) − 0.30 |",
            "|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        if l != "qwen":
            continue
        v = q4["S3_confidence_vs_correct"][f"study{st}:{d}/{b}/{l}"]
        out.append(f"| {st} | {NAMES[b]} | {v['n_correct']:,} | {v['auroc_vs_correct']['i']:.3f} | "
                   f"{v['auroc_vs_correct']['A']:.3f} | {iv(v['auroc_i_minus_half'])} | {iv(v['detection_i_minus_random'])} |")
    out += ["", "*Table S8.14. DeBERTa's stable errors on which both adjudicators chose the label opposite to the "
            "reference [P]: detection rates and A − (ii) (unit bootstrap). †: OLMo's (ii) is not a working judge. "
            "Study 2 has 8 such errors; its intervals are not interpretable.*", "",
            "| Study | LLM | n (units) | A | (i) | (ii) | debiased (ii) | A − (ii) | A − debiased (ii) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, v in q4["S1_both_opposite"].items():
        st = k[5]
        l = k.split("/")[-1]
        e = v["detection"]
        out.append(f"| {st} | {NAMES[l]}{DG(l)} | {v['n']} ({v['units']}) | {e['A']:.3f} | {e['i']:.3f} | {e['ii']:.3f} | "
                   f"{e['ii_deb']:.3f} | {iv(v['A_minus_ii'])} | {iv(v['A_minus_ii_deb'])} |")
    return "\n".join(out) + "\n"


def table_s8_round2():
    """Tables S8.4-S8.7 (round-2 post-hoc analyses)."""
    q = posthoc2()
    iv = lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]"
    out = ["*Table S8.4. Ceiling-normalised detection [P]: (detection − 0.30) / (ceiling − 0.30), where 0.30 is the "
           "q-mean of a random ranking and the ceiling is the q-mean of min(1, qN/n) for N ranked items and n stable "
           "errors. A − control on the same scale.*", "",
           "| Study | Classifier × LLM | N | n (share) | Ceiling | A | (i) | (ii) | A − (i) | A − (ii) | A − debiased (ii) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        v = q["Q2_normalised"][f"study{st}:{d}/{b}/{l}"]
        nd = v["normalised_detection"]
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{DG(l)} | {v['ranked_items']:,} | {v['stable_errors']} ({v['prevalence']:.3f}) | "
                   f"{v['ceiling']:.3f} | {fmt(nd['A'])} | {fmt(nd['i'])} | {fmt(nd['ii'])} | {iv(v['A_minus_i'])} | "
                   f"{iv(v['A_minus_ii'])} | {iv(v['A_minus_ii_deb'])} |")
    out += ["", "| Classifier × LLM | Study 1 − Study 2, normalised A − (ii) | normalised A − debiased (ii) |", "|---|---|---|"]
    for k, v in q["Q2_between_study"].items():
        b, l = k.split("/")
        out.append(f"| {NAMES[b]} × {NAMES[l]}{DG(l)} | " + " | ".join(
            f"{fmt(v[c]['study1_minus_study2'])} [{fmt(v[c]['lo'])}, {fmt(v[c]['hi'])}]" for c in ("ii", "ii_deb")) + " |")
    out += ["", "*Table S8.5. The LLM judges as classifiers over all ranked items [P]: accuracy against the reference, "
            "share predicted 'different', share of items whose score is tied with another item, and the largest share "
            "of tied items at a flag boundary. (i): the classifier's confidence (ties only). †: OLMo's (ii) and (iii) are not working judges (Table 2).*", "",
            "| Study | LLM | Judge | Accuracy | Predicted different | Tied scores | Ties at boundary (max) |",
            "|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        if b != "deberta":
            continue
        v = q["Q3_judges"][f"study{st}:{d}/{b}/{l}"]
        for m in ("A", "ii", "iii", "iv"):
            x = v[m]
            out.append(f"| {st} | {NAMES[l]} | ({m}) | {x['accuracy']:.3f} | {x['share_predict_different']:.3f} | "
                       f"{x['share_items_with_tied_score']:.3f} | {max(x['ties_at_boundaries'].values()):.4f} |")
        if l == "qwen":  # (i) is the classifier, the same for both LLMs: ties only
            x = v["i"]
            out.append(f"| {st} | — | (i) | — | — | {x['share_items_with_tied_score']:.3f} | "
                       f"{max(x['ties_at_boundaries'].values()):.4f} |")
    out += ["", "*Table S8.6. Combining ranks with confidence (mean of ranks, as the frozen (i)+A) on stable errors, and "
            "AURC over abstention rates 0–0.95 on all items with a reference [P].*", "",
            "| Study | Classifier × LLM | (i)+(ii) − (i) | (i)+A − (i)+(ii) | (i)+A − A | AURC (i) | AURC A | AURC (i)+A | AURC (ii) | AURC (i)+(ii) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        v = q["Q4_combined"][f"study{st}:{d}/{b}/{l}"]
        a = v["aurc"]
        out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{DG(l)} | {iv(v['i+ii_minus_i'])} | {iv(v['i+A_minus_i+ii'])} | "
                   f"{iv(v['i+A_minus_A'])} | " + " | ".join(f"{a[m]:.3f}" for m in ("i", "A", "i+A", "ii", "i+ii")) + " |")
    out += ["", "*Table S8.7. A − (ii) and A − debiased (ii) within the classifier's prediction ŷ, with cluster-bootstrap "
            "intervals within the stratum [P].*", "",
            "| Study | Classifier × LLM | ŷ | n (units) | A − (ii) | A − debiased (ii) |", "|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        v = q["Q5_yhat_strata"][f"study{st}:{d}/{b}/{l}"]
        for y, nm in (("yhat_T", "same"), ("yhat_F", "different")):
            if y in v:
                x = v[y]
                out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{DG(l)} | {nm} | {x['n']} ({x['units']}) | {iv(x['A_minus_ii'])} | "
                           f"{iv(x['A_minus_ii_deb'])} |")
    return "\n".join(out) + "\n"


def table_s8():
    """Table S8: post-hoc P1 (between-study differences), P3 (by predicted label), P4 (WordNet-example split)."""
    p = posthoc()
    out = ["*Table S8.1. Study 1 − Study 2 difference in A − control, with an independent cluster bootstrap per study "
           "[P].*", "", "| Classifier × LLM | A − (i) | A − (ii) | A − (iii) |", "|---|---|---|---|"]
    for k, v in p["P1_between_study"].items():
        b, l = k.split("/")
        out.append(f"| {NAMES[b]} × {NAMES[l]}{DG(l)} | " + " | ".join(
            f"{fmt(v[c]['study1_minus_study2'])} [{fmt(v[c]['lo'])}, {fmt(v[c]['hi'])}]" for c in ("i", "ii", "iii")) + " |")
    out += ["", "*Table S8.2. Detection rate of stable errors by the classifier's prediction ŷ (same / different), and "
            "the share of all ranked items each method predicts as different [P].*", "",
            "| Study | Classifier × LLM | ŷ | n | A | (i) | (ii) | (iii) | A − (ii) |", "|---|---|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        g = p["P3_granularity"].get(f"study{st}:{d}/{b}/{l}")
        if not g:
            continue
        for y, nm in (("yhat_T", "same"), ("yhat_F", "different")):
            det = g[y]["detection"]
            out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{DG(l)} | {nm} | {g[y]['n_stable']} | " + " | ".join(
                fmt(det[m]) for m in ("A", "i", "ii", "iii")) + f" | {fmt(det['A'] - det['ii'])} |")
    out += ["", "| Study | Classifier × LLM | Predicted different: A | (ii) | (iii) | classifier | reference |",
            "|---|---|---|---|---|---|---|"]
    for st, d, b, l in CELLS:
        g = p["P3_granularity"].get(f"study{st}:{d}/{b}/{l}")
        if g:
            s = g["share_predict_different_all_items"]
            out.append(f"| {st} | {NAMES[b]} × {NAMES[l]}{DG(l)} | {fmt(s['A'])} | {fmt(s['ii'])} | {fmt(s['iii'])} | "
                       f"{fmt(s['base'])} | {fmt(g['share_gold_different_all_items'])} |")
    w = p["P4_wordnet_examples"]
    out += ["", f"*Table S8.3. WiC test split by whether both sentences are WordNet usage examples "
            f"({w['items_both_sentences_in_wordnet_examples']:,} of {w['items_total']:,} items) [P].*", "",
            "| Classifier × LLM | Part | n (units) | A − (i) | A − (ii) | A − (iii) |", "|---|---|---|---|---|---|"]
    for k, v in w.items():
        if not isinstance(v, dict):
            continue
        b, l = k.split("/")[1:]
        for part, nm in (("wordnet_examples", "both WordNet examples"), ("other", "other")):
            x = v[part]
            out.append(f"| {NAMES[b]} × {NAMES[l]}{DG(l)} | {nm} | {x['n_stable']} ({x['units']}) | " + " | ".join(
                f"{fmt(x['A_minus'][c]['d'])} [{fmt(x['A_minus'][c]['lo'])}, {fmt(x['A_minus'][c]['hi'])}]"
                for c in ("i", "ii", "iii")) + " |")
    return "\n".join(out) + "\n"


def adjudication():
    import json
    return json.loads((ROOT / "experiments/20261110-adjudication/results/analysis.json").read_text(encoding="utf-8"))


def table_adjudication():
    """Table 4 (main text): human adjudication of DeBERTa's stable errors [P], counts per LLM cell."""
    a = adjudication()
    q6 = posthoc2()["Q6"]["adjudication_by_role"]
    out = ["*Table 4. Human adjudication of DeBERTa's stable errors [P] (two raters independent of the study, blind to "
           "the reference, the model and the source dataset; S9). Confirmed: both raters chose the reference label. "
           "Contested: all other errors; the next column counts those where both raters chose the opposite label. Counts "
           "are those of each comparison (OLMo excludes its parse failures). Baseline: on the stable-correct items mixed "
           "into the same sheet, both raters chose the reference label on "
           f"{q6['study1_control']['both_agree_with_reference']} of {q6['study1_control']['n']} (Study 1) and "
           f"{q6['study2_control']['both_agree_with_reference']} of {q6['study2_control']['n']} (Study 2). Study 2's "
           "contested class has 15 errors in 14 units; its intervals are not interpretable. †: OLMo's (ii) is not a working judge "
           "(Table 2); its rows are not evidence about the structure. Debiased (ii): as in Table 3.*", "",
           "| Study | LLM | Stable errors | Confirmed | Contested | Both raters opposite | A − (ii), confirmed | A − (ii), contested | A − debiased (ii), confirmed | A − debiased (ii), contested |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    r7 = posthoc3()["R7_adjudication"]
    for st, d in (("1", "wic_test"), ("2", "mcl")):
        for l in ("qwen", "olmo"):
            cell = a["cells"][f"{d}/deberta/{l}"]
            iv = lambda k: (lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]")(cell[k]["A_minus"]["ii"])
            nc, nn = cell["certain"]["n"], cell["contested"]["n"]
            if l == "qwen":
                q = r7[f"study{st}:{d}/deberta/qwen"]
                f3 = lambda x: f"{fmt(x['d'])} [{fmt(x['lo'])}, {fmt(x['hi'])}]"
                deb = f"{f3(q['confirmed']['A_minus_ii_deb'])} | {f3(q['contested']['A_minus_ii_deb'])}"
                out.append(f"| {st} | {NAMES[l]} | {nc + nn} | {nc} | {nn} | {cell['suspected_reference_error']['n']} | "
                           f"{iv('certain')} | {iv('contested')} | {deb} |")
            else:
                out.append(f"| {st} | {NAMES[l]} † | {nc + nn} | {nc} | {nn} | {cell['suspected_reference_error']['n']} | "
                           f"{iv('certain')} | {iv('contested')} | — | — |")
    return "\n".join(out) + "\n"


def table_s9():
    a = adjudication()
    ag = a["rater_agreement"]
    out = ["*Table S9.1. Agreement between the two adjudicators (three categories: same, different, undecided).*", "",
           "| Items | n | Agreement | κ |", "|---|---|---|---|"]
    for k, nm in (("all", "all"), ("controls", "stable-correct controls"), ("errors", "stable errors")):
        out.append(f"| {nm} | {ag[k]['n']} | {ag[k]['percent']:.3f} | {ag[k]['kappa3']:.3f} |")
    out += ["", "*Table S9.2. Share of each rater's judgements that agree / disagree with the reference.*", "",
            "| Rater | Study 1 controls | Study 1 errors | Study 2 controls | Study 2 errors |", "|---|---|---|---|---|"]
    for i, (r, v) in enumerate(a["agreement_with_reference"].items(), start=1):
        out.append(f"| {i} | " + " | ".join(
            f"{v[k]['agree']:.3f} / {v[k]['opposite']:.3f} (n {v[k]['n']})"
            for k in ("study1_control", "study1_error", "study2_control", "study2_error")) + " |")
    out += ["", "*Table S9.3. Detection rate and A − control on certain and contested stable errors.*", "",
            "| Cell | Class | n (units) | A | (i) | (ii) | (iii) | A − (i) | A − (ii) | A − (iii) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, cell in a["cells"].items():
        d, b, l = k.split("/")
        for cls in ("certain", "contested"):
            x = cell.get(cls)
            if not x:
                continue
            det = x["detection"]
            out.append(f"| {DATA[d]} {NAMES[b]} × {NAMES[l]} | {cls} | {x['n']} ({x['units']}) | " + " | ".join(
                fmt(det[m]) for m in ("A", "i", "ii", "iii")) + " | " + " | ".join(
                f"{fmt(x['A_minus'][c]['d'])} [{fmt(x['A_minus'][c]['lo'])}, {fmt(x['A_minus'][c]['hi'])}]"
                for c in ("i", "ii", "iii")) + " |")
    out += ["", "| Cell | Reading (rule fixed before the ratings) |", "|---|---|"]
    for k, v in ((k, c["reading"]) for k, c in a["cells"].items()):
        d, b, l = k.split("/")
        note = " (†: OLMo's (ii) is not a working judge; not evidence about the structure)" if l == "olmo" else ""
        out.append(f"| {DATA[d]} {NAMES[b]} × {NAMES[l]} | {v}{note} |")
    return "\n".join(out) + "\n"


def table_s3():
    r = rows("tableS3_check.csv")
    h = rows("tableS3_halves.csv")
    out = ["*Table S3.1. Paraphrase check by rater. Study 1: audit sample per task (gate). Study 2: every item (filter). "
           "Controls: hidden rows that should be rejected.*", "",
           "| Study / task | Rater | Rows | Kept | Kept rate [Wilson 95%] | Controls kept / used | Verdict | Agreement | κ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for x in r:
        w = f" {x['wilson95']}" if x["wilson95"] else ""
        out.append(f"| {x['task']} | {x['rater']} | {int(x['paraphrase_rows']):,} | {int(x['paraphrase_rows_kept']):,} | "
                   f"{float(x['kept_rate']):.3f}{w} | {x['controls_kept']} / {x['controls_used']} | {x['verdict']} | "
                   f"{float(x['agreement_pct']):.3f} | {float(x['kappa']):.3f} |")
    out += ["", "*Table S3.2. Study 2, kept rate by half of the sheet (row order).*", "",
            "| Half | Rater | Rows | Kept | Kept rate | Controls kept / shown |", "|---|---|---|---|---|---|"]
    for x in h:
        out.append(f"| {x['half']} | {x['rater']} | {int(x['paraphrase_rows']):,} | {int(x['kept']):,} | "
                   f"{float(x['kept_rate']):.3f} | {x['controls_kept']} / {x['controls']} |")
    return "\n".join(out) + "\n"


def build(src, dst):
    text = (P / src).read_text(encoding="utf-8")
    body = text.split("<!-- INCLUDE tables/references.md -->")[0]
    (TAB / "references.md").write_text(references(body), encoding="utf-8") if src == "manuscript.md" else None
    def inc(m):
        return (P / m.group(1)).read_text(encoding="utf-8")
    out = re.sub(r"<!-- INCLUDE (\S+) -->", inc, text)
    if src != "manuscript.md":
        out = out.replace("<!-- REFERENCES -->", references(text))
    (P / dst).write_text(out, encoding="utf-8")
    print("->", P / dst)


ANON = [(re.compile(r"https://osf\.io/\w+"), "[anonymized view-only link]"),
        (re.compile(r"\bthe researcher's\b"), "an author's"), (re.compile(r"\bthe researcher\b"), "an author"),
        (re.compile(r"\bThe researcher\b"), "An author"), (re.compile(r"\ba co-author\b"), "another author"),
        (re.compile(r"\(researcher's decision\)"), "(authors' decision)")]


VOL = ROOT / "experiments/20261109-rep-main/private/osf_view_only_links.json"


def anonymize(src, dst):
    """Submission build for double-blind review: drop HTML comments, put the anonymous OSF view-only links (kept in a
    private file) in place of the registration URLs, and replace author roles."""
    import json
    text = re.sub(r"<!--.*?-->\n?", "", (P / src).read_text(encoding="utf-8"), flags=re.S)
    vol = json.loads(VOL.read_text(encoding="utf-8")) if VOL.exists() else {}
    text = re.sub(r"https://osf\.io/(\w+)", lambda m: vol.get(m.group(1), "[anonymized view-only link]"), text)
    for pat, rep in ANON[1:]:
        text = pat.sub(rep, text)
    left = re.findall(r"osf\.io/\w+/?(?!\?view_only=)(?=[\s|)]|$)|researcher\b|co-author|<!--", text)
    assert not left, f"{dst}: identifying strings remain: {set(left)}"
    (P / dst).write_text(text, encoding="utf-8")
    print("->", P / dst)


def main():
    (TAB / "table1.md").write_text(table_results(), encoding="utf-8")
    (TAB / "table2.md").write_text(table_flow(), encoding="utf-8")
    (TAB / "tableS4.md").write_text(table_s4(), encoding="utf-8")
    (TAB / "tableS3.md").write_text(table_s3(), encoding="utf-8")
    (TAB / "table_share.md").write_text(table_share(), encoding="utf-8")
    (TAB / "table_adjudication.md").write_text(table_adjudication(), encoding="utf-8")
    (TAB / "tableS8.md").write_text(table_s8() + "\n" + table_s8_round2() + "\n" + table_s8_round3() + "\n" + table_s8_round4(), encoding="utf-8")
    (TAB / "tableS9.md").write_text(table_s9(), encoding="utf-8")
    build("manuscript.md", "manuscript_built.md")
    if (P / "supplement.md").exists():
        build("supplement.md", "supplement_built.md")
        anonymize("supplement_built.md", "supplement_anon.md")
    anonymize("manuscript_built.md", "manuscript_anon.md")


if __name__ == "__main__":
    main()
