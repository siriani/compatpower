"""Power, equivalence and allocation for compatibility assay designs."""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis import noninferiority as _noninf, test_plates_blocked, test_plates_pooled, test_run_level, tost
from .model import AssayModel, simulate_counts

ANALYSES = {"run": test_run_level, "pooled": test_plates_pooled, "blocked": test_plates_blocked}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def power(runs: int, plates: int, reduction: float, model: AssayModel = AssayModel(),
          analysis: str = "run", alpha: float = 0.05, n_sim: int = 2000, seed: int | None = 1) -> float:
    """Probability of a significant difference (power if reduction > 0; type I error if reduction = 0).

    analysis: "run" (run = experimental unit; recommended), "pooled" (plates as replicates) or
    "blocked" (plates, run as block). "run" and "blocked" need runs >= 2.
    """
    if analysis not in ANALYSES:
        raise ValueError(f"analysis must be one of {list(ANALYSES)}")
    yc, yt = simulate_counts(model, runs, plates, reduction, n_sim, _rng(seed))
    return float(ANALYSES[analysis](yc, yt, alpha).mean())


def equivalence(runs: int, plates: int, margin: tuple[float, float] = (0.8, 1.25),
                model: AssayModel = AssayModel(), true_reduction: float = 0.0, alpha: float = 0.05,
                n_sim: int = 2000, seed: int | None = 1) -> float:
    """Probability that TOST declares equivalence (ratio within `margin`) given the true reduction."""
    yc, yt = simulate_counts(model, runs, plates, true_reduction, n_sim, _rng(seed))
    return float(tost(yc, yt, margin, alpha).mean())


def noninferiority(runs: int, plates: int, margin: float = 0.66, model: AssayModel = AssayModel(),
                   true_reduction: float = 0.0, alpha: float = 0.05, n_sim: int = 2000, seed: int | None = 1) -> float:
    """Probability that a run-level non-inferiority test declares compatibility (ratio > `margin`).

    With true_reduction = 0 this is the power to show compatibility of a harmless product; with
    true_reduction = 1 - margin it is the type I error. The default margin 0.66 is the ratio implied by the
    "compatible" class of the biological index IB (IB = 100 x ratio when all endpoints drop equally; IB > 66).
    """
    yc, yt = simulate_counts(model, runs, plates, true_reduction, n_sim, _rng(seed))
    return float(_noninf(yc, yt, margin, alpha).mean())


def design_table(runs: Iterable[int] = (3, 4, 5, 6, 8, 10), plates: Iterable[int] = (3, 5, 10, 20),
                 reductions: Iterable[float] = (0.2, 0.3, 0.5), margin: tuple[float, float] = (0.75, 1 / 0.75),
                 model: AssayModel = AssayModel(), alpha: float = 0.05, n_sim: int = 2000,
                 seed: int | None = 1) -> pd.DataFrame:
    """Run-level power for each reduction and probability of showing equivalence, for each runs x plates."""
    rows = []
    for i, (r, n) in enumerate(itertools.product(runs, plates)):
        s = None if seed is None else seed + i
        row = {"runs": r, "plates": n, "total_plates": r * n}
        for red in reductions:
            row[f"power_{round(red * 100)}pct"] = power(r, n, red, model, "run", alpha, n_sim, s)
        row["equivalence"] = equivalence(r, n, margin, model, 0.0, alpha, n_sim, s)
        row["noninferiority_066"] = noninferiority(r, n, 0.66, model, 0.0, alpha, n_sim, s)
        rows.append(row)
    return pd.DataFrame(rows)


def allocate(total_plates: int, reduction: float = 0.3, model: AssayModel = AssayModel(),
             margin: tuple[float, float] = (0.75, 1 / 0.75), min_plates: int = 2, alpha: float = 0.05,
             n_sim: int = 2000, seed: int | None = 1) -> pd.DataFrame:
    """Compare ways of splitting a fixed number of plates per group into runs x plates."""
    rows = []
    for r in range(2, total_plates // min_plates + 1):
        if total_plates % r:
            continue
        n = total_plates // r
        rows.append({"runs": r, "plates": n,
                     "power": power(r, n, reduction, model, "run", alpha, n_sim, seed),
                     "equivalence": equivalence(r, n, margin, model, 0.0, alpha, n_sim, seed)})
    return pd.DataFrame(rows)


def required_runs(plates: int, reduction: float = 0.3, target: float = 0.8, model: AssayModel = AssayModel(),
                  max_runs: int = 30, alpha: float = 0.05, n_sim: int = 2000, seed: int | None = 1) -> int | None:
    """Smallest number of runs reaching `target` power with `plates` per group per run (None if > max_runs)."""
    for r in range(2, max_runs + 1):
        if power(r, plates, reduction, model, "run", alpha, n_sim, seed) >= target:
            return r
    return None
