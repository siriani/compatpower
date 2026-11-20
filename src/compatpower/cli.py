"""Command-line interface: `compatpower power|equivalence|noninferiority|runs|table|allocate|analyze|estimate`."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import __version__
from .design import allocate, design_table, equivalence, noninferiority, power, required_runs_ni
from .estimate import estimate_model
from .model import AssayModel


def _model_args(p: argparse.ArgumentParser) -> None:
    d = AssayModel()
    p.add_argument("--mu0", type=float, default=d.mu0, help="mean count per control plate")
    p.add_argument("--k", type=float, default=d.k, help="negative binomial k within a run")
    p.add_argument("--sd-run", type=float, default=d.sd_run, help="SD of run effect (log scale)")
    p.add_argument("--tau", type=float, default=d.tau, help="SD of run x treatment interaction (log scale)")
    p.add_argument("--n-sim", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)


def _model(a: argparse.Namespace) -> AssayModel:
    return AssayModel(mu0=a.mu0, k=a.k, sd_run=a.sd_run, tau=a.tau)


def _margin(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    return lo, hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compatpower", description="Power and design of in vitro compatibility assays")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("power", help="power for a design"); _model_args(p)
    p.add_argument("--runs", type=int, required=True); p.add_argument("--plates", type=int, required=True)
    p.add_argument("--reduction", type=float, default=0.3); p.add_argument("--analysis", default="run", choices=["run", "pooled", "blocked"])

    e = sub.add_parser("equivalence", help="probability of showing equivalence"); _model_args(e)
    e.add_argument("--runs", type=int, required=True); e.add_argument("--plates", type=int, required=True)
    e.add_argument("--margin", type=_margin, default=(0.75, 1 / 0.75), help="e.g. 0.75,1.333")

    ni = sub.add_parser("noninferiority", help="probability of showing compatibility (one-sided, ratio > margin)"); _model_args(ni)
    ni.add_argument("--runs", type=int, required=True); ni.add_argument("--plates", type=int, required=True)
    ni.add_argument("--margin", type=float, default=0.70, help="default 0.70 (IOBC harmless class); 0.66 = IB compatible class")

    t = sub.add_parser("table", help="design table runs x plates"); _model_args(t)
    t.add_argument("--margin", type=_margin, default=(0.75, 1 / 0.75)); t.add_argument("--out", help="CSV output")

    al = sub.add_parser("allocate", help="split a fixed number of plates per group into runs"); _model_args(al)
    al.add_argument("--total", type=int, required=True); al.add_argument("--reduction", type=float, default=0.3)
    al.add_argument("--margin", type=_margin, default=(0.75, 1 / 0.75))

    an = sub.add_parser("analyze", help="run-level analysis of real plate counts (one product): compatible / harmful / inconclusive")
    an.add_argument("csv", help="columns: run, group (control/treatment), count [, product]")
    an.add_argument("--margin", type=float, default=0.70); an.add_argument("--alpha", type=float, default=0.05)
    an.add_argument("--product", help="analyse only this value of the product column")

    rr = sub.add_parser("runs", help="runs needed to declare compatibility (non-inferiority)"); _model_args(rr)
    rr.add_argument("--plates", type=int, required=True); rr.add_argument("--margin", type=float, default=0.70)
    rr.add_argument("--reduction", type=float, default=0.0, help="true reduction of the product (0 = no effect)")
    rr.add_argument("--target", type=float, default=0.8)

    es = sub.add_parser("estimate", help="estimate mu0, k, sd_run (and tau) from a CSV of plate counts")
    es.add_argument("csv", help="columns: run, group (control/treatment), count [, product]")

    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        from .analyze import analyze
        df = pd.read_csv(a.csv)
        if a.product is not None:
            df = df[df["product"].astype(str) == a.product]
        print(json.dumps(analyze(df, a.margin, a.alpha), indent=1))
        return 0
    if a.cmd == "estimate":
        model, rep = estimate_model(pd.read_csv(a.csv))
        print(json.dumps(rep, indent=1, default=float))
        return 0
    m = _model(a)
    if a.cmd == "power":
        print(f"{power(a.runs, a.plates, a.reduction, m, a.analysis, a.alpha, a.n_sim, a.seed):.3f}")
    elif a.cmd == "equivalence":
        print(f"{equivalence(a.runs, a.plates, a.margin, m, 0.0, a.alpha, a.n_sim, a.seed):.3f}")
    elif a.cmd == "noninferiority":
        print(f"{noninferiority(a.runs, a.plates, a.margin, m, 0.0, a.alpha, a.n_sim, a.seed):.3f}")
    elif a.cmd == "runs":
        print(required_runs_ni(a.plates, a.margin, a.reduction, a.target, m, alpha=a.alpha, n_sim=a.n_sim, seed=a.seed))
    elif a.cmd == "table":
        df = design_table(margin=a.margin, model=m, alpha=a.alpha, n_sim=a.n_sim, seed=a.seed)
        df.to_csv(a.out, index=False) if a.out else print(df.round(3).to_string(index=False))
    elif a.cmd == "allocate":
        print(allocate(a.total, a.reduction, m, a.margin, alpha=a.alpha, n_sim=a.n_sim, seed=a.seed).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
