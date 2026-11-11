"""Forced-choice sense judgement for A and (iv) in the replication study (rep_preregistration.md).

Changes from methods.wic_structured and prompts/wic_A.txt: the "None of the above" option is removed, so the model
must pick one of the listed glosses, and the question is worded for the closest meaning with a number-only answer
(prompts/rep_wic_A.txt v3). P(same) = sum_k p1(k) * p2(k) as before. To keep the row layout of the main
evaluation (whose analysis reads K = len(p1) - 1 and the "none" fields), a zero is appended to p1 and p2 for the
absent option, none_prob is [0, 0] and choice_correct_none is [False, False]. The appended zero is not a model
output; it is recorded as such ("forced_choice": true).
"""
from __future__ import annotations

from methods import POS_NAME, wic_marked


def sense_dist_forced(lm, tpl, item, k, definitions, tag):
    options = "\n".join(f"{j + 1}. {d}" for j, d in enumerate(definitions))
    text = tpl.format(lemma=item["lemma"], pos_name=POS_NAME[item["pos"]], sentence=wic_marked(item, k),
                      options=options, n_opts=len(definitions))
    return lm.choose(text, [str(j + 1) for j in range(len(definitions))], max_new_tokens=4, tag=tag)


def wic_structured_forced(lm, tpl, item, inv, tag="A"):
    r1 = sense_dist_forced(lm, tpl, item, 1, inv["definitions"], tag)
    r2 = sense_dist_forced(lm, tpl, item, 2, inv["definitions"], tag)
    K = len(inv["definitions"])
    p_same = sum(r1["probs"][j] * r2["probs"][j] for j in range(K))
    return {"p_same": p_same, "p1": list(r1["probs"]) + [0.0], "p2": list(r2["probs"]) + [0.0],
            "parsed": r1["parsed"] and r2["parsed"], "synsets": inv["synsets"], "tolerance": inv.get("tolerance"),
            "greedy": [r1["greedy"], r2["greedy"]], "n_prompts": 2, "none_prob": [0.0, 0.0],
            "choice_correct_none": [False, False], "forced_choice": True}
