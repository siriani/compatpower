"""compatpower: power, equivalence and design of in vitro compatibility assays
between biological control agents and pesticides."""
__version__ = "0.1.2"

from .model import AssayModel, simulate_counts
from .design import allocate, design_table, equivalence, noninferiority, power, required_runs, required_runs_ni
from .estimate import estimate_model, estimate_k
from .analyze import analyze

__all__ = ["AssayModel", "simulate_counts", "power", "equivalence", "noninferiority", "design_table", "allocate",
           "required_runs", "required_runs_ni", "estimate_model", "estimate_k", "analyze", "__version__"]
