"""Analyse a real compatibility assay at the run level (Section 2.5 of the accompanying article)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ZERO_MEAN = 0.5  # a run whose mean count is zero is set to 0.5 before taking logs


def analyze(df: pd.DataFrame, margin: float = 0.70, alpha: float = 0.05) -> dict:
    """Run-level non-inferiority analysis of plate counts, with a three-outcome decision.

    df: one row per plate with columns `run`, `group` ("control"/"treatment") and `count`, for ONE product.
        If a `product` column is present it must hold a single value (filter the data frame first).
    The per-run log ratio of mean counts (natural log; a zero run mean becomes 0.5) is averaged over runs and a
    two-sided (1 - 2*alpha) confidence interval [L, U] is built with a t distribution on runs - 1 df.
    Decision: "compatible" if L > margin; "harmful" if U < margin; "inconclusive" otherwise.
    Returns the ratio, L, U, the two-sided p value for a difference, the one-sided non-inferiority p value,
    the decision and the per-run ratios.
    """
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1), e.g. 0.70")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    need = {"run", "group", "count"}
    if not need <= set(df.columns):
        raise ValueError(f"columns required: {sorted(need)}")
    if "product" in df.columns and df["product"].dropna().nunique() > 1:
        raise ValueError("the data frame holds several products; analyse one product per call "
                         "(pooling products mixes different true ratios and gives a meaningless average)")
    g = df.assign(group=df["group"].astype(str).str.lower().str.strip())
    bad = set(g["group"]) - {"control", "treatment"}
    if bad:
        raise ValueError(f"unknown group labels: {sorted(bad)} (use control/treatment)")
    m = g.groupby(["run", "group"])["count"].mean().unstack()
    if not {"control", "treatment"} <= set(m.columns):
        raise ValueError("each run needs control and treatment plates")
    m = m.dropna()
    r = len(m)
    if r < 2:
        raise ValueError("at least 2 runs with both groups are needed for a run-level analysis")
    zero_runs = [str(k) for k in m.index[(m <= 0).any(axis=1)]]
    m = m.where(m > 0, ZERO_MEAN)
    d = np.log(m["treatment"]) - np.log(m["control"])
    dbar, se = d.mean(), d.std(ddof=1) / np.sqrt(r)
    tcrit = stats.t.ppf(1 - alpha, r - 1)
    if se > 0:
        p_diff = float(2 * stats.t.sf(abs(dbar / se), r - 1))
        p_ni = float(stats.t.sf((dbar - np.log(margin)) / se, r - 1))
    else:
        p_diff = p_ni = float("nan")
    lower, upper = float(np.exp(dbar - tcrit * se)), float(np.exp(dbar + tcrit * se))
    decision = "compatible" if lower > margin else "harmful" if upper < margin else "inconclusive"
    out = {"runs": r, "ratio": float(np.exp(dbar)), "lower_limit": lower, "upper_limit": upper,
           "confidence_two_sided": 1 - 2 * alpha, "alpha_one_sided": alpha, "margin": margin,
           "p_difference": p_diff, "p_noninferiority": p_ni, "decision": decision,
           "per_run_ratio": {str(k): float(v) for k, v in np.exp(d).items()}}
    if zero_runs:
        out["zero_mean_runs"] = zero_runs
    return out
