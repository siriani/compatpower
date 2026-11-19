# compatpower

**Power, equivalence and design of in vitro compatibility assays between biological control agents and pesticides.**

In vitro compatibility assays (for example, colony-forming units of an entomopathogenic fungus on medium with a pesticide) are often run on a single day with 3–5 plates per treatment. "Compatible" is then declared when the treatment does not differ significantly from the control. `compatpower` simulates such assays to answer three planning questions:

- How likely is my design to detect a given reduction (power)?
- How likely is it to **demonstrate compatibility** with an equivalence test (TOST on the treatment/control ratio)?
- Given a fixed number of plates, how should I split them into independent runs (days) and plates per run?

## Model
Plate counts follow a negative binomial (variance = μ + μ²/k). Each run (day) shifts both groups by a random effect (`sd_run`). The product effect may vary between runs, which is the run × treatment interaction (`tau`). Analyses compared:
- `run`: the run is the experimental unit (paired *t* test on per-run log mean differences). This is recommended and keeps the nominal error rate.
- `pooled`: plates are treated as replicates, as is common practice. Error inflates when `tau > 0`.
- `blocked`: plates with the run as a block, equivalent to a random-run-intercept mixed model.

## Install
```bash
pip install compatpower            # after release
pip install git+https://github.com/siriani/compatpower
```

## Quick start
```python
from compatpower import AssayModel, power, equivalence, design_table, allocate, estimate_model

m = AssayModel(mu0=25, k=4.3, sd_run=0.48, tau=0.15)   # defaults: CFU of a B. bassiana isolate
power(runs=6, plates=10, reduction=0.3, model=m)          # ~0.70
equivalence(runs=8, plates=20, margin=(0.75, 1/0.75), model=m)
allocate(total_plates=60, reduction=0.3, model=m)         # 2x30 vs 10x6 vs 20x3 ...
design_table(model=m)                                     # runs x plates table

import pandas as pd                                      # your own counts: run, group, count [, product]
model, report = estimate_model(pd.read_csv("my_counts.csv"))
```

Command line:
```bash
compatpower power --runs 6 --plates 10 --reduction 0.3
compatpower allocate --total 60
compatpower table --out design.csv
compatpower estimate examples/example_counts.csv
```

## Defaults and caveats
The default parameters are estimated from CFU counts of *Beauveria bassiana* isolate IBCB 170 over three runs. `sd_run` is imprecise (three runs), and `tau` was assumed (0.15), because it can be estimated only when the same product is tested in several runs. Estimate the parameters from your own data whenever possible (`estimate_model`). Analyses use log(count + 0.5). A negative binomial mixed model is the natural alternative for final analyses.

## Citation
Please cite the software through its Zenodo DOI (see `CITATION.cff`) and the accompanying article (in preparation).

## License
MIT
