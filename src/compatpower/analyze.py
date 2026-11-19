"""Analyse a real compatibility assay at the run level (Box 1 of the accompanying article)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def analyze(df: pd.DataFrame, margin: float = 0.70, alpha: float = 0.05) -> dict:
    """Run-level non-inferiority analysis of plate counts.

    df: one row per plate with columns `run`, `group` ("control"/"treatment") and `count`
        (one product per call; filter the data frame first).
    Returns the treatment/control ratio, its lower one-sided (1 - alpha) confidence limit, the two-sided
    p value for a difference, the non-inferiority p value, and the decision ("compatible" when the lower
    limit exceeds `margin`).
    """
    need = {"run", "group", "count"}
    if not need <= set(df.columns):
        raise ValueError(f"columns required: {sorted(need)}")
    g = df.assign(group=df["group"].str.lower().str.strip())
    m = g.groupby(["run", "group"])["count"].mean().unstack()
    if not {"control", "treatment"} <= set(m.columns):
        raise ValueError("each run needs control and treatment plates")
    m = m.dropna()
    r = len(m)
    if r < 2:
        raise ValueError("at least 2 runs with both groups are needed for a run-level analysis")
    if (m <= 0).any().any():
        raise ValueError("a run has a zero mean count; check the data (or dilution)")
    d = np.log(m["treatment"]) - np.log(m["control"])
    dbar, se = d.mean(), d.std(ddof=1) / np.sqrt(r)
    tcrit = stats.t.ppf(1 - alpha, r - 1)
    p_diff = float(2 * stats.t.sf(abs(dbar / se), r - 1)) if se > 0 else float("nan")
    p_ni = float(stats.t.sf((dbar - np.log(margin)) / se, r - 1)) if se > 0 else float("nan")
    lower = float(np.exp(dbar - tcrit * se))
    return {"runs": r, "ratio": float(np.exp(dbar)), "lower_limit": lower,
            "upper_limit": float(np.exp(dbar + tcrit * se)), "confidence": 1 - alpha, "margin": margin,
            "p_difference": p_diff, "p_noninferiority": p_ni,
            "decision": "compatible" if lower > margin else "not shown compatible",
            "per_run_ratio": {str(k): float(v) for k, v in np.exp(d).items()}}
