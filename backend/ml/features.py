"""
ml/features.py
==============
Design-matrix construction shared by every candidate model.

Two feature sets are supported:

``base``
    The seven predictors of the production RGR equation
    (BA, BA², AGSfrac, TCR, DQ, q, Cut).  Using exactly these lets the
    benchmark isolate *algorithm* differences from *feature* differences.

``extended``
    ``base`` plus a handful of cheap, biologically motivated transforms:

    * ``logBA``      – growth responds to density on a log scale
    * ``BAxAGS``     – stocking × quality interaction
    * ``DQ_over_BA`` – a crude stand-structure index (big trees per unit BA)
    * ``Interval``   – length of the remeasurement period (RGR is annualised,
                       but longer intervals average out disturbance)

The cleaning rules are identical to ``models.rgr_model.RGRModel`` so the
production OLS baseline in the leaderboard sees exactly what it sees in
training (TCR default = 5, Cut default = 0, rows missing a predictor dropped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

BASE_FEATURES: List[str] = ["BA", "BA2", "AGSfrac", "TCR", "DQ", "q", "Cut"]
EXTENDED_FEATURES: List[str] = BASE_FEATURES + ["logBA", "BAxAGS", "DQ_over_BA", "Interval"]
SPATIAL_FEATURES: List[str] = EXTENDED_FEATURES + ["Latitude", "Longitude"]
TARGET: str = "RGR"
GROUP_COLUMN: str = "PlotKey"
TIME_COLUMN: str = "EndYear"

# ``spatial`` adds plot coordinates.  Location is a legitimate proxy for site
# quality, but it also *identifies the plot*: with naive K-fold a flexible
# learner can memorise each plot's intercept from its neighbours in the training
# fold, which is exactly the leakage that plot-grouped CV prevents.  The
# benchmark's leakage check is therefore most instructive on this feature set.
FEATURE_SETS = {"base": BASE_FEATURES, "extended": EXTENDED_FEATURES, "spatial": SPATIAL_FEATURES}


@dataclass
class Dataset:
    """A cleaned, model-ready view of one model group's growth intervals."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    years: pd.Series
    feature_names: List[str]

    @property
    def n_samples(self) -> int:
        return int(len(self.y))

    @property
    def n_groups(self) -> int:
        return int(self.groups.nunique())


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def build_dataset(df: pd.DataFrame, feature_set: str = "base") -> Dataset:
    """
    Turn a raw growth-interval frame (``GROWTH_DATA_COLUMNS`` shape) into
    ``Dataset``.  Raises ``ValueError`` for an unknown feature set or when the
    target column is missing.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Valid: {sorted(FEATURE_SETS)}")
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' is required to build a dataset")

    work = df.copy()
    work["BA"] = _coerce_numeric(work, "BA")
    work["BA2"] = work["BA"] ** 2
    work["AGSfrac"] = _coerce_numeric(work, "AGSfrac")
    work["TCR"] = _coerce_numeric(work.assign(TCR=work["TCR"].fillna(5)), "TCR")
    work["DQ"] = _coerce_numeric(work, "DQ")
    work["q"] = _coerce_numeric(work, "q")
    work["Cut"] = _coerce_numeric(work.assign(Cut=work["Cut"].fillna(0)), "Cut")
    work[TARGET] = _coerce_numeric(work, TARGET)

    if feature_set in ("extended", "spatial"):
        work["logBA"] = np.log(work["BA"].clip(lower=0.1))
        work["BAxAGS"] = work["BA"] * work["AGSfrac"]
        work["DQ_over_BA"] = work["DQ"] / work["BA"].clip(lower=0.1)
        if "GrowthInterval_Years" in work.columns:
            work["Interval"] = _coerce_numeric(work, "GrowthInterval_Years")
        else:
            work["Interval"] = np.nan
        work["Interval"] = work["Interval"].fillna(work["Interval"].median() if work["Interval"].notna().any() else 5.0)

    if feature_set == "spatial":
        missing = [c for c in ("Latitude", "Longitude") if c not in work.columns]
        if missing:
            raise ValueError(f"Feature set 'spatial' needs columns {missing}")
        work["Latitude"] = _coerce_numeric(work, "Latitude")
        work["Longitude"] = _coerce_numeric(work, "Longitude")

    features = FEATURE_SETS[feature_set]
    work = work.dropna(subset=features + [TARGET])

    if GROUP_COLUMN in work.columns:
        groups = work[GROUP_COLUMN].astype(str)
    else:  # no plot identifier → every row is its own group (plain K-fold behaviour)
        groups = pd.Series(np.arange(len(work)).astype(str), index=work.index)

    if TIME_COLUMN in work.columns:
        years = _coerce_numeric(work, TIME_COLUMN).fillna(0).astype(int)
    else:
        years = pd.Series(np.zeros(len(work), dtype=int), index=work.index)

    return Dataset(
        X=work[features].astype(float).reset_index(drop=True),
        y=work[TARGET].astype(float).reset_index(drop=True),
        groups=groups.reset_index(drop=True),
        years=years.reset_index(drop=True),
        feature_names=list(features),
    )


def continuous_feature_indices(feature_names: List[str]) -> List[int]:
    """Indices of features that are sensible inputs for spline / polynomial expansion."""
    binary = {"Cut"}
    return [i for i, name in enumerate(feature_names) if name not in binary]


def feature_set_names() -> List[str]:
    return sorted(FEATURE_SETS)


def describe(dataset: Dataset) -> dict:
    """Small summary used in reports."""
    y = dataset.y
    years = dataset.years
    summary: dict = {
        "n_samples": dataset.n_samples,
        "n_plots": dataset.n_groups,
        "features": dataset.feature_names,
        "target_mean": float(y.mean()),
        "target_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
    }
    if years.max() > 0:
        summary["year_range"] = [int(years.min()), int(years.max())]
    return summary


def intervals_per_plot(dataset: Dataset) -> Optional[float]:
    if dataset.n_groups == 0:
        return None
    return float(dataset.n_samples / dataset.n_groups)
