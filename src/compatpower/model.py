"""Simulation model for in vitro compatibility assays.

Counts (e.g. colony-forming units, CFU) on each plate follow a negative binomial (NB2) distribution:

    y_rgi ~ NB2(mu_rg, k),  Var(y) = mu + mu^2 / k
    log mu_rg = log mu0 + a_r + g * (log(1 - reduction) + c_r)
    a_r ~ N(0, sd_run)   run (day) effect shared by control and treatment
    c_r ~ N(0, tau)      run x treatment interaction: the product effect varies between runs

r = run (independent repetition of the assay), g = 0 for control and 1 for treatment, i = plate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class AssayModel:
    """Parameters of the data-generating model.

    mu0: mean count per control plate.
    k: negative binomial size (dispersion) parameter within a run; smaller k = more variable plates.
    sd_run: SD of the run effect on the log scale (day-to-day drift of the mean).
    tau: SD of the run x treatment interaction on the log scale.
    """

    mu0: float = 25.0
    k: float = 4.3
    sd_run: float = 0.48
    tau: float = 0.15

    def __post_init__(self) -> None:
        if self.mu0 <= 0 or self.k <= 0 or self.sd_run < 0 or self.tau < 0:
            raise ValueError("mu0 and k must be > 0; sd_run and tau must be >= 0")

    def as_dict(self) -> dict:
        return asdict(self)


def simulate_counts(model: AssayModel, runs: int, plates: int, reduction: float, n_sim: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate control and treatment counts with shape (n_sim, runs, plates)."""
    if runs < 1 or plates < 1:
        raise ValueError("runs and plates must be >= 1")
    if not 0 <= reduction < 1:
        raise ValueError("reduction must be in [0, 1)")
    a = rng.normal(0, model.sd_run, (n_sim, runs, 1))
    c = rng.normal(0, model.tau, (n_sim, runs, 1))
    mu_c = model.mu0 * np.exp(a)
    mu_t = model.mu0 * np.exp(a + np.log1p(-reduction) + c)
    shape = (n_sim, runs, plates)
    k = model.k
    yc = rng.negative_binomial(k, k / (k + np.broadcast_to(mu_c, shape)))
    yt = rng.negative_binomial(k, k / (k + np.broadcast_to(mu_t, shape)))
    return yc, yt
