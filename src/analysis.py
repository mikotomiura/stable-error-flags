"""Confirmatory analysis (preregistration v4, Secs. 1, 7, 8 and 9).

From the references and the scores of each method, compute detection rates, paired differences, bootstrap
intervals, and the decisions for H1-H4.

- Stable error (Sec. 1): an item on which the base model gives the same prediction under every form in set S
  and that prediction differs from the reference.
- Flags (Sec. 7): all items are ranked by score (descending) and the top round(q * N) are flagged. Remaining
  flags at a tied boundary are split equally among the tied items (fractional flags). The primary metric is
  the mean flag over q in {0.10, 0.20, 0.30, 0.40, 0.50}.
- Differences and intervals (Sec. 8): d = mean(x_A - x_c) over the same set of stable errors, with a
  percentile interval from resampling independent units with replacement (10,000 draws, seed 20261102).
  Only the units of stable errors are resampled; flags are not recomputed.
- Decisions: fixed sequence H1 -> H2 -> H3 -> H4, each at one-sided alpha = 0.025 (lower end of the two-sided
  95% interval above 0). H1, H3 and H4 require beating all three controls (intersection-union test, no
  multiplicity correction). H1 additionally requires positive point estimates against all three controls
  on the development split.
- Negative reading (Sec. 1): if the upper end is below 0.08, "no gain of 8 points or more"; otherwise
  "inconclusive".
"""
from __future__ import annotations

import numpy as np

QS = (0.10, 0.20, 0.30, 0.40, 0.50)
N_BOOT = 10_000
SEED = 20261102
NEG_BOUND = 0.08
CONTROLS = ("i", "ii", "iii")
SUPPORTED, REJECTED, ABORTED, EXPLORATORY = "supported", "rejected", "aborted", "exploratory"
NO_GAIN, INCONCLUSIVE = "no gain of 8 points or more", "inconclusive"


# ---------- Stable errors (Secs. 1-2) ----------

def classify_errors(preds_by_form: dict, ref: dict, excluded=frozenset()):
    """Split the base model's errors into stable and unstable ones.

    preds_by_form: {form: {id: prediction}}, including the original form "orig".
    ref: {id: reference}; None marks items without a reference (e.g. tied majority votes).
    excluded: ids that are never counted as errors (e.g. items that could not be transformed).
    Returns (stable_ids, unstable_ids).
    """
    forms = list(preds_by_form)
    assert "orig" in forms
    stable, unstable = [], []
    for i, y in ref.items():
        if y is None or i in excluded:
            continue
        p = preds_by_form["orig"][i]
        if p == y:
            continue
        if all(preds_by_form[f][i] == p for f in forms):
            stable.append(i)
        else:
            unstable.append(i)
    return stable, unstable


# ---------- Flags (Sec. 7) ----------

def fractional_flags(scores, q):
    """Flag the top round(q * N) items; remaining flags at a tied boundary are split equally."""
    s = np.asarray(scores, float)
    n = len(s)
    k = int(round(q * n))
    out = np.zeros(n)
    if k == 0:
        return out
    thr = np.sort(s)[::-1][k - 1]
    above = s > thr
    tied = s == thr
    out[above] = 1.0
    out[tied] = (k - int(above.sum())) / int(tied.sum())
    return out


def q_mean_flags(scores, qs=QS):
    """Primary metric per item: the flag averaged over the five flag rates."""
    return np.mean([fractional_flags(scores, q) for q in qs], axis=0)


# ---------- Differences and intervals (Sec. 8) ----------

def boot_weights(n_units, n_boot=N_BOOT, seed=SEED):
    """Resampling counts (n_boot x n_units). The same draws are shared across comparisons."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_boot)


def unit_index(unit_labels):
    """Map unit labels (lemma or premise string) to 0..U-1."""
    _, inv = np.unique(np.asarray(unit_labels, dtype=object).astype(str), return_inverse=True)
    return inv


def diff_ci(xa, xb, unit, weights=None):
    """d = mean(xa - xb) and its two-sided 95% percentile interval under unit resampling (ratio estimator)."""
    d = np.asarray(xa, float) - np.asarray(xb, float)
    unit = np.asarray(unit)
    n_u = int(unit.max()) + 1
    if weights is None:
        weights = boot_weights(n_u)
    su = np.bincount(unit, weights=d, minlength=n_u)
    nu = np.bincount(unit, minlength=n_u).astype(float)
    est = (weights @ su) / (weights @ nu)
    lo, hi = np.quantile(est, [0.025, 0.975])
    return {"d": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)), "units": n_u}


def negative_reading(ci):
    return NO_GAIN if ci["hi"] < NEG_BOUND else INCONCLUSIVE


# ---------- Decisions (Secs. 1, 8, 9) ----------

def detection_on(ids, flags: dict):
    """flags: {method: {id: q-mean flag}} -> {method: array over ids}."""
    return {m: np.array([f[i] for i in ids]) for m, f in flags.items()}


def iut(x: dict, unit, controls=CONTROLS):
    """Intersection-union test: A must beat every control (lower end > 0)."""
    w = boot_weights(int(np.asarray(unit).max()) + 1)
    cis = {c: diff_ci(x["A"], x[c], unit, w) for c in controls}
    return {"pass": all(ci["lo"] > 0 for ci in cis.values()), "ci": cis,
            "failed": [c for c, ci in cis.items() if ci["lo"] <= 0]}


def content_share(x, unit):
    """(A - iv) / (A - ii): point estimate, interval, and the share of draws with a non-positive denominator."""
    unit = np.asarray(unit)
    n_u = int(unit.max()) + 1
    w = boot_weights(n_u)
    nu = np.bincount(unit, minlength=n_u).astype(float)
    num = np.bincount(unit, weights=x["A"] - x["iv"], minlength=n_u)
    den = np.bincount(unit, weights=x["A"] - x["ii"], minlength=n_u)
    bn, bd = (w @ num) / (w @ nu), (w @ den) / (w @ nu)
    ok = bd > 0
    est = bn[ok] / bd[ok]
    return {"point": float(num.sum() / den.sum()), "lo": float(np.quantile(est, 0.025)) if ok.any() else None,
            "hi": float(np.quantile(est, 0.975)) if ok.any() else None, "denominator_le0_frac": float(1 - ok.mean())}


def decide(test=None, dev=None, nli=None, nli_high=None, aborted=frozenset(), exploratory=frozenset()):
    """Apply the fixed testing sequence H1 -> H2 -> H3 -> H4.

    Each data argument is a pair (x: {method: array}, unit) or None. A hypothesis without data must be listed
    in `aborted`; `exploratory` lists hypotheses demoted by Sec. 9 (e.g. a failed paraphrase check).
    Returns {hypothesis: {status, confirmatory, ...}}.
    """
    out = {}
    confirm = True  # a hypothesis is confirmatory only if every earlier one was confirmatorily supported

    def put(h, res, passed):
        nonlocal confirm
        if h in aborted:
            res.update(status=ABORTED, confirmatory=False)
            confirm = False
        elif h in exploratory or not confirm:
            res.update(status=EXPLORATORY, confirmatory=False)
            confirm = False
        else:
            res.update(status=SUPPORTED if passed else REJECTED, confirmatory=True)
            if not passed:
                for c in res.get("failed", []):
                    res.setdefault("negative_reading", {})[c] = negative_reading(res["ci"][c])
                confirm = False
        out[h] = res

    if test is not None:
        # H1
        r = iut(*test)
        if dev is not None:
            dx, _ = dev
            pts = {c: float((dx["A"] - dx[c]).mean()) for c in CONTROLS}
            r.update(dev_points=pts, dev_pass=all(v > 0 for v in pts.values()))
            passed = r["pass"] and r["dev_pass"]
            if r["pass"] and not r["dev_pass"]:
                r["failed"] = []  # test passes, dev does not: no negative reading (Sec. 1, table)
        else:
            r["dev_condition"] = "dropped (Sec. 9)"
            passed = r["pass"]
        put("H1", r, passed)
        # H2
        x, u = test
        ci = diff_ci(x["A"], x["iv"], u)
        r2 = {"ci": {"iv": ci}, "pass": ci["lo"] > 0, "failed": [] if ci["lo"] > 0 else ["iv"],
              "report_iv_minus_ii": diff_ci(x["iv"], x["ii"], u)}
        if diff_ci(x["A"], x["ii"], u)["lo"] > 0:
            r2["content_share"] = content_share(x, u)
        put("H2", r2, r2["pass"])
    else:
        assert {"H1", "H2"} <= set(aborted), "hypotheses without data must be passed as aborted"
        put("H1", {}, False)
        put("H2", {}, False)
    for h, arg in (("H3", nli), ("H4", nli_high)):
        if arg is None:  # no data: aborted, or exploratory with an empty subset (H4, Sec. 9)
            assert h in aborted or h in exploratory, "hypotheses without data must be passed as aborted"
            put(h, {} if h in aborted else {"n": 0, "note": "empty subset; nothing computed"}, False)
            continue
        r = iut(*arg)
        put(h, r, r["pass"])
    return out
