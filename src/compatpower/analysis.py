"""Vectorised tests applied to simulated assays (all on the log(count + 0.5) scale)."""
from __future__ import annotations

import numpy as np
from scipy import stats

OFFSET = 0.5


def _run_differences(yc: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """Per-run difference of log mean counts, shape (n_sim, runs)."""
    return np.log(yt.mean(2) + OFFSET) - np.log(yc.mean(2) + OFFSET)


def test_run_level(yc: np.ndarray, yt: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Paired t test on per-run differences (run = experimental unit). Needs >= 2 runs."""
    runs = yc.shape[1]
    if runs < 2:
        raise ValueError("run-level analysis needs at least 2 runs")
    d = _run_differences(yc, yt)
    m, se = d.mean(1), d.std(1, ddof=1) / np.sqrt(runs)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(m / se), runs - 1)
    return np.nan_to_num(p, nan=1.0) < alpha


def test_plates_pooled(yc: np.ndarray, yt: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Welch t test pooling all plates (plates treated as replicates; common practice)."""
    n = yc.shape[0]
    lc = np.log(yc.reshape(n, -1) + OFFSET)
    lt = np.log(yt.reshape(n, -1) + OFFSET)
    p = stats.ttest_ind(lt, lc, axis=1, equal_var=False).pvalue
    return np.nan_to_num(p, nan=1.0) < alpha


def test_plates_blocked(yc: np.ndarray, yt: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Additive ANOVA (run + group) on plates; equals a random-run-intercept model. Needs >= 2 runs."""
    _, runs, plates = yc.shape
    if runs < 2:
        raise ValueError("blocked analysis needs at least 2 runs")
    lc, lt = np.log(yc + OFFSET), np.log(yt + OFFSET)
    mc, mt = lc.mean(2), lt.mean(2)
    d = mt - mc
    dbar = d.mean(1)
    ss_within = ((lc - mc[..., None]) ** 2).sum((1, 2)) + ((lt - mt[..., None]) ** 2).sum((1, 2))
    ss_inter = plates * ((d - dbar[:, None]) ** 2).sum(1) / 2
    df = 2 * runs * plates - runs - 1
    se = np.sqrt((ss_within + ss_inter) / df * 2 / (plates * runs))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(dbar / se), df)
    return np.nan_to_num(p, nan=1.0) < alpha


def tost(yc: np.ndarray, yt: np.ndarray, margin: tuple[float, float] = (0.8, 1.25),
         alpha: float = 0.05) -> np.ndarray:
    """Two one-sided tests for equivalence of the treatment/control ratio.

    Run level when there are >= 2 runs; plate level (Welch-type SE) when there is a single run.
    """
    lo, hi = np.log(margin[0]), np.log(margin[1])
    if not lo < 0 < hi:
        raise ValueError("margin must bracket 1, e.g. (0.8, 1.25)")
    n, runs, plates = yc.shape
    if runs >= 2:
        d = _run_differences(yc, yt)
        m, se, df = d.mean(1), d.std(1, ddof=1) / np.sqrt(runs), runs - 1
    else:
        lc = np.log(yc.reshape(n, -1) + OFFSET)
        lt = np.log(yt.reshape(n, -1) + OFFSET)
        m = lt.mean(1) - lc.mean(1)
        se = np.sqrt(lt.var(1, ddof=1) / plates + lc.var(1, ddof=1) / plates)
        df = 2 * plates - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.maximum(stats.t.sf((m - lo) / se, df), stats.t.cdf((m - hi) / se, df))
    return np.nan_to_num(p, nan=1.0) < alpha
