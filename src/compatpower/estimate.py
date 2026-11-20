"""Estimate model parameters from a laboratory's own plate counts.

Input: a table with one row per plate and columns `run`, `group` ("control"/"treatment") and `count`.
Optionally `product`, to estimate tau from the same product tested in several runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

from .model import AssayModel


def _nb_negloglik(log_k: float, groups: list[np.ndarray]) -> float:
    k = np.exp(log_k)
    ll = 0.0
    for y in groups:
        mu = max(y.mean(), 1e-9)
        ll += np.sum(special.gammaln(y + k) - special.gammaln(k) - special.gammaln(y + 1)
                     + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))
    return -ll


def estimate_k(groups: list[np.ndarray]) -> float:
    """Maximum-likelihood NB size parameter k with a separate mean per group."""
    res = optimize.minimize_scalar(_nb_negloglik, bounds=(-4, 10), args=(groups,), method="bounded")
    return float(np.exp(res.x))


def estimate_model(df: pd.DataFrame, tau: float | None = None, controls_only: bool = True) -> tuple[AssayModel, dict]:
    """Fit mu0, k and sd_run from control plates (k also from treatment plates if controls_only=False).

    sd_run is estimated by the method of moments: variance of log control means across runs minus their
    expected sampling variance. tau is estimated only if a `product` column has the same product in >= 2 runs;
    otherwise the given value (or 0.15) is used. Returns the model and a report dict with diagnostics.
    """
    need = {"run", "group", "count"}
    if not need <= set(df.columns):
        raise ValueError(f"columns required: {sorted(need)}")
    df = df.assign(group=df["group"].str.lower().str.strip())
    src = df[df.group == "control"] if controls_only else df
    groups = [g["count"].to_numpy(float) for _, g in src.groupby(["run", "group"])]
    ctrl = [g["count"].to_numpy(float) for _, g in df[df.group == "control"].groupby("run")]
    if not ctrl:
        raise ValueError("no control plates found (group == 'control')")
    k = estimate_k(groups)
    log_means = np.array([np.log(y.mean()) for y in ctrl])
    mu0 = float(np.exp(log_means.mean()))
    report = {"n_runs_control": len(ctrl), "n_plates": int(len(df)), "k": k}
    if len(ctrl) >= 2:
        samp = np.mean([(1 / y.mean() + 1 / k) / len(y) for y in ctrl])
        sd_run = float(np.sqrt(max(log_means.var(ddof=1) - samp, 0.0)))
    else:
        sd_run = 0.0
        report["warning_sd_run"] = "only one run: sd_run set to 0 (cannot be estimated)"
    tau_hat = None
    if "product" in df.columns:
        diffs = []
        for (prod, run), g in df[df.group == "treatment"].groupby(["product", "run"]):
            c = df[(df.run == run) & (df.group == "control")]["count"]
            if len(c):
                diffs.append((prod, np.log(g["count"].mean() + .5) - np.log(c.mean() + .5),
                              (1 / max(g["count"].mean(), .5) + 1 / k) / len(g) + (1 / max(c.mean(), .5) + 1 / k) / len(c)))
        d = pd.DataFrame(diffs, columns=["product", "diff", "samp"])
        rep = d.groupby("product").filter(lambda x: len(x) >= 2)
        if len(rep):
            v = rep.groupby("product").apply(lambda x: x["diff"].var(ddof=1) - x["samp"].mean(), include_groups=False)
            tau_hat = float(np.sqrt(max(v.mean(), 0.0)))
            report["n_products_repeated"] = int(rep["product"].nunique())
    if tau_hat is None:
        tau_hat = 0.15 if tau is None else tau
        report["warning_tau"] = "tau not estimable (no product tested in >= 2 runs); using assumed value"
    report.update(mu0=mu0, sd_run=sd_run, tau=tau_hat)
    return AssayModel(mu0=mu0, k=k, sd_run=sd_run, tau=tau_hat), report
