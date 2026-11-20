import numpy as np
import pandas as pd
import pytest

from compatpower import AssayModel, allocate, equivalence, estimate_model, noninferiority, power, simulate_counts
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


def test_noninferiority_more_powerful_than_tost_and_valid():
    ni = noninferiority(5, 10, 0.70, M, n_sim=2000, seed=2)
    eq = equivalence(5, 10, (0.70, 1 / 0.70), M, n_sim=2000, seed=2)
    assert ni > eq
    err = noninferiority(5, 10, 0.70, M, true_reduction=0.30, n_sim=4000, seed=5)
    assert 0.03 < err < 0.07


def test_analyze_example():
    from compatpower import analyze
    df = pd.read_csv("examples/example_counts.csv")
    res = analyze(df, margin=0.70)
    assert res["runs"] == 4 and 0 < res["lower_limit"] < res["ratio"] < res["upper_limit"]
    assert res["decision"] in ("compatible", "harmful", "inconclusive")


def _toy(ratio, runs=6, plates=10, seed=0, product=None):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(runs):
        day = np.exp(rng.normal(0, 0.4))
        for g, f in (("control", 1.0), ("treatment", ratio)):
            for v in rng.poisson(40 * day * f, plates):
                rows.append((r, g, int(v)))
    df = pd.DataFrame(rows, columns=["run", "group", "count"])
    if product is not None:
        df["product"] = product
    return df


def test_analyze_three_outcomes():
    from compatpower import analyze
    assert analyze(_toy(1.0))["decision"] == "compatible"
    assert analyze(_toy(0.3))["decision"] == "harmful"
    assert analyze(_toy(0.7, runs=2, seed=1))["decision"] == "inconclusive"


def test_analyze_rejects_several_products_and_handles_zero_runs():
    from compatpower import analyze
    df = pd.concat([_toy(1.0, product="A"), _toy(0.5, product="B")])
    with pytest.raises(ValueError):
        analyze(df)
    z = _toy(1.0)
    z.loc[(z.run == 0) & (z.group == "treatment"), "count"] = 0
    res = analyze(z)
    assert res["zero_mean_runs"] == ["0"] and np.isfinite(res["lower_limit"])


def test_required_runs_ni_monotone():
    from compatpower import required_runs_ni
    r0 = required_runs_ni(10, 0.70, 0.0, model=M, n_sim=800)
    r1 = required_runs_ni(10, 0.70, 0.10, model=M, n_sim=800)
    assert r0 is not None and 4 <= r0 <= 8 and (r1 is None or r1 > r0)


def test_cli_runs(capsys):
    assert main(["runs", "--plates", "10", "--n-sim", "300"]) == 0
    assert int(capsys.readouterr().out) >= 2
