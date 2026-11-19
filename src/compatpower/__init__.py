"""compatpower: power, equivalence and design of in vitro compatibility assays
between biological control agents and pesticides."""
__version__ = "0.1.0"

from .model import AssayModel, simulate_counts
from .design import allocate, design_table, equivalence, noninferiority, power, required_runs
from .estimate import estimate_model, estimate_k

__all__ = ["AssayModel", "simulate_counts", "power", "equivalence", "noninferiority", "design_table", "allocate",
           "required_runs", "estimate_model", "estimate_k", "__version__"]
