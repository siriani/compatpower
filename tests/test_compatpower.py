import numpy as np
import pandas as pd
import pytest

from compatpower import AssayModel, allocate, equivalence, estimate_model, power, simulate_counts
from compatpower.cli import main

M = AssayModel(mu0=25, k=4.3, sd_run=0.48, tau=0.15)


def test_type_one_error_run_level_is_nominal():
    p = power(5, 10, 0.0, AssayModel(tau=0.3), "run", n_sim=4000, seed=3)
    assert 0.035 < p < 0.065


def test_pooled_plates_inflate_error_with_interaction():
    p = power(5, 20, 0.0, AssayModel(tau=0.3), "pooled", n_sim=3000, seed=4)
    assert p > 0.12


def test_power_increases_with_runs():
    assert power(8, 10, 0.3, M, n_sim=1500) > power(3, 10, 0.3, M, n_sim=1500)


def test_single_run_cannot_show_equivalence():
    assert equivalence(1, 20, (0.8, 1.25), M, n_sim=1500) < 0.01


def test_allocation_favours_more_runs():
    a = allocate(60, 0.3, M, n_sim=800)
    assert a.iloc[-1]["power"] > a.iloc[0]["power"]


def test_estimate_recovers_parameters():
    rng = np.random.default_rng(0)
    true = AssayModel(mu0=30, k=5, sd_run=0.3, tau=0.0)
    yc, yt = simulate_counts(true, 40, 10, 0.2, 1, rng)
    rows = [(r, g, int(v)) for g, y in (("control", yc), ("treatment", yt)) for r in range(40) for v in y[0, r]]
    m, rep = estimate_model(pd.DataFrame(rows, columns=["run", "group", "count"]))
    assert 3 < m.k < 8 and 0.15 < m.sd_run < 0.45 and 22 < m.mu0 < 40
    assert "warning_tau" in rep


def test_bad_inputs():
    with pytest.raises(ValueError):
        AssayModel(k=0)
    with pytest.raises(ValueError):
        power(1, 5, 0.3, M, "run")


def test_cli_power(capsys):
    assert main(["power", "--runs", "4", "--plates", "5", "--n-sim", "200"]) == 0
    assert 0 <= float(capsys.readouterr().out) <= 1
