"""
ml/evaluation.py
================
Metrics and leakage-aware cross-validation.

Why three CV strategies?
------------------------
``group``    GroupKFold on PlotKey.  A plot's intervals never straddle the
             train/test boundary, so the score answers "how well do we predict
             growth on plots we have never measured?" – the operational question.
``temporal`` Forward chaining on EndYear: train on earlier remeasurements, test
             on later ones.  Mimics deploying today's coefficients on next
             decade's inventory.
``kfold``    Naive shuffled K-fold.  Included deliberately as a *leakage
             control*: because repeated measures of the same plot leak across
             folds, it over-states skill.  The gap to ``group`` is reported.

Metrics
-------
r2, adjusted r2, rmse, mae, median_ae, p90/p95/p99 absolute error, max_ae,
bias (mean residual), mape (guarded), residual sd, and the share of
predictions within ±1 and ±2 production RMSE bands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .features import Dataset
from .registry import ModelSpec, needs_groups

CV_STRATEGIES = ("group", "temporal", "kfold")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, n_features: int = 0) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0}

    resid = y_true - y_pred
    abs_err = np.abs(resid)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1) if n - n_features - 1 > 0 else r2
    rmse = float(np.sqrt(ss_res / n))

    nonzero = np.abs(y_true) > 1e-6
    mape = float(np.mean(abs_err[nonzero] / np.abs(y_true[nonzero])) * 100.0) if nonzero.any() else float("nan")

    return {
        "n": n,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": rmse,
        "mae": float(abs_err.mean()),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "p95_ae": float(np.percentile(abs_err, 95)),
        "p99_ae": float(np.percentile(abs_err, 99)),
        "max_ae": float(abs_err.max()),
        "bias": float(resid.mean()),
        "mape": mape,
        "resid_sd": float(resid.std(ddof=1)) if n > 1 else 0.0,
    }


PRIMARY_METRIC = "rmse"           # lower is better; ties broken by p95_ae
RANKING_METRICS = ("rmse", "p95_ae")
HIGHER_IS_BETTER = {"r2", "adj_r2"}


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def _shuffled_group_kfold(groups: pd.Series, n_splits: int, seed: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold with randomised group→fold assignment (sklearn's is deterministic by size)."""
    rng = np.random.default_rng(seed)
    unique = groups.unique()
    permuted = rng.permutation(unique)
    relabel = {g: i for i, g in enumerate(permuted)}
    encoded = groups.map(relabel).to_numpy()
    yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=encoded)


def _temporal_splits(
    years: pd.Series, n_splits: int, min_train_frac: float = 0.4
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Forward-chaining splits on measurement year.

    The earliest ``min_train_frac`` of rows (by year) always trains; the
    remaining rows are cut into ``n_splits`` chronological blocks, each tested
    after training on everything that ended before it.
    """
    order = np.argsort(years.to_numpy(), kind="stable")
    sorted_years = years.to_numpy()[order]
    n = len(order)
    start = int(n * min_train_frac)
    block_edges = np.linspace(start, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = block_edges[k], block_edges[k + 1]
        if hi <= lo:
            continue
        cutoff_year = sorted_years[lo]
        # train strictly before the first test year to avoid same-year leakage
        train_idx = order[sorted_years < cutoff_year]
        test_idx = order[lo:hi]
        if len(train_idx) < 10 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def make_splits(dataset: Dataset, strategy: str, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if strategy not in CV_STRATEGIES:
        raise ValueError(f"Unknown CV strategy '{strategy}'. Valid: {CV_STRATEGIES}")
    n = dataset.n_samples
    if strategy == "group":
        k = min(n_splits, dataset.n_groups)
        if k < 2:
            raise ValueError("Grouped CV needs at least 2 distinct plots")
        return list(_shuffled_group_kfold(dataset.groups, k, seed))
    if strategy == "temporal":
        if dataset.years.nunique() < 3:
            raise ValueError("Temporal CV needs at least 3 distinct measurement years")
        splits = list(_temporal_splits(dataset.years, n_splits))
        if not splits:
            raise ValueError("Temporal CV produced no usable folds")
        return splits
    k = min(n_splits, n)
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n)))


def group_overlap(dataset: Dataset, splits: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """Number of plots appearing in both train and test of any fold (0 ⇒ leakage-free)."""
    overlap = 0
    for train_idx, test_idx in splits:
        overlap += len(set(dataset.groups.iloc[train_idx]) & set(dataset.groups.iloc[test_idx]))
    return overlap


# ---------------------------------------------------------------------------
# Cross-validation of one model
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    model: str
    strategy: str
    folds: List[Dict[str, float]] = field(default_factory=list)
    oof_pred: Optional[np.ndarray] = None       # out-of-fold predictions (NaN where never tested)
    fit_seconds: float = 0.0
    n_fits: int = 0

    def metric_summary(self) -> Dict[str, Dict[str, float]]:
        if not self.folds:
            return {"mean": {}, "std": {}}
        frame = pd.DataFrame(self.folds)
        numeric = frame.select_dtypes(include=[np.number])
        return {
            "mean": {k: float(v) for k, v in numeric.mean().items()},
            "std": {k: float(v) for k, v in numeric.std(ddof=1).fillna(0.0).items()},
        }

    def pooled_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """Metrics over all out-of-fold predictions pooled (robust for small folds)."""
        if self.oof_pred is None:
            return {}
        mask = ~np.isnan(self.oof_pred)
        return compute_metrics(dataset.y.to_numpy()[mask], self.oof_pred[mask], len(dataset.feature_names))


def _fit(estimator, X, y, groups):
    if needs_groups(estimator):
        estimator.fit(X, y, groups=groups)
    else:
        estimator.fit(X, y)
    return estimator


def cross_validate(
    spec: ModelSpec,
    dataset: Dataset,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
    seed: int,
) -> CVResult:
    result = CVResult(model=spec.name, strategy=strategy)
    oof = np.full(dataset.n_samples, np.nan)
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=float)
    groups = dataset.groups.to_numpy()

    for fold_no, (train_idx, test_idx) in enumerate(splits):
        estimator = spec.build(dataset.feature_names, seed)
        t0 = time.perf_counter()
        _fit(estimator, X[train_idx], y[train_idx], groups[train_idx])
        result.fit_seconds += time.perf_counter() - t0
        result.n_fits += 1
        pred = np.asarray(estimator.predict(X[test_idx]), dtype=float).ravel()
        oof[test_idx] = pred
        fold_metrics = compute_metrics(y[test_idx], pred, len(dataset.feature_names))
        fold_metrics["fold"] = fold_no
        fold_metrics["n_train"] = int(len(train_idx))
        result.folds.append(fold_metrics)

    result.oof_pred = oof
    return result


def fit_full(spec: ModelSpec, dataset: Dataset, seed: int):
    """Fit on every row (used for coefficients, importance, intervals)."""
    estimator = spec.build(dataset.feature_names, seed)
    return _fit(estimator, dataset.X.to_numpy(dtype=float), dataset.y.to_numpy(dtype=float), dataset.groups.to_numpy())


# ---------------------------------------------------------------------------
# Biological plausibility (Penner 2023 expected signs) – linear models only
# ---------------------------------------------------------------------------

EXPECTED_SIGNS = {
    "BA": "negative",
    "AGSfrac": "positive",
    "TCR": "positive",
    "DQ": "negative",
    "q": "positive",
    "Cut": "negative",
}


def sign_check(coefficients: Optional[Dict[str, float]]) -> Optional[Dict[str, object]]:
    if not coefficients:
        return None
    wrong: List[str] = []
    for name, expected in EXPECTED_SIGNS.items():
        value = coefficients.get(name)
        if value is None:
            continue
        if expected == "positive" and value < 0:
            wrong.append(f"{name}(-)")
        elif expected == "negative" and value > 0:
            wrong.append(f"{name}(+)")
    return {"wrong_signs": wrong, "wrong_sign_count": len(wrong), "checked": list(EXPECTED_SIGNS)}
