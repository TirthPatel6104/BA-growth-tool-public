"""
backend/ml
==========
Model benchmarking toolkit for the Basal Area Growth Tool.

The production model (``models/rgr_model.py``) is a fixed-form ordinary least
squares regression.  This package answers the question *"is that the right
model?"* in a statistically honest way:

* ``features``    – one place that turns raw growth intervals into a design matrix
* ``registry``    – a catalogue of candidate algorithms (linear, regularised,
                    GAM-style splines, kernel, tree ensembles, boosting, neural,
                    mixed effects) with graceful fallback when optional libraries
                    are not installed
* ``evaluation``  – metrics (R², RMSE, MAE, p95 absolute error, bias, …) and
                    leakage-aware cross-validation (plot-grouped, temporal, naive)
* ``conformal``   – distribution-free prediction intervals + coverage checks
* ``stats``       – paired significance tests and bootstrap confidence intervals
* ``benchmark``   – orchestration producing a leaderboard per model group
* ``reports``     – JSON / Markdown writers

Everything here runs on the synthetic demo data in SQLite; the same code runs
unchanged against a real SQL Server extract.
"""

from .benchmark import BenchmarkConfig, benchmark_all, benchmark_group  # noqa: F401
from .registry import available_models, list_models  # noqa: F401

__all__ = [
    "BenchmarkConfig",
    "benchmark_all",
    "benchmark_group",
    "available_models",
    "list_models",
]
