"""
ml/conformal.py
===============
Distribution-free prediction intervals and their evaluation.

A point estimate of RGR is only half an answer for a forest manager; the
question is "how wrong could this be?".  Two interval methods are compared:

``split_conformal``
    Wraps *any* point model.  Plots are split into a proper-training part and
    a calibration part; the (1-α) quantile of calibration |residuals| becomes a
    symmetric half-width that is guaranteed (in expectation) to cover the true
    value with probability ≥ 1-α for exchangeable data.  Group-aware splitting
    keeps the guarantee honest under repeated measures.

``quantile_gbm``
    Native heteroscedastic intervals from two gradient-boosting models fitted to
    the α/2 and 1-α/2 pinball losses.  Width adapts to the inputs (e.g. wider
    after a cut), at the cost of no finite-sample guarantee.

Both are scored on held-out plots by *empirical coverage* (should ≈ 1-α) and
*mean width* (narrower is more useful at equal coverage).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .evaluation import _fit, make_splits
from .features import Dataset
from .registry import ModelSpec


def _coverage_and_width(y, lower, upper) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y >= lower) & (y <= upper)
    return {
        "coverage": float(inside.mean()) if len(y) else float("nan"),
        "mean_width": float(np.mean(upper - lower)) if len(y) else float("nan"),
        "median_width": float(np.median(upper - lower)) if len(y) else float("nan"),
        "n": int(len(y)),
    }


def _split_groups(groups: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_cal = max(1, int(round(len(unique) * frac)))
    cal_groups = set(unique[:n_cal].tolist())
    is_cal = np.array([g in cal_groups for g in groups])
    return np.where(~is_cal)[0], np.where(is_cal)[0]


def split_conformal_evaluate(
    spec: ModelSpec,
    dataset: Dataset,
    alpha: float = 0.10,
    n_splits: int = 5,
    seed: int = 42,
    calibration_frac: float = 0.3,
) -> Dict[str, float]:
    """
    For each grouped CV fold: fit on proper-train plots, calibrate on
    calibration plots, score coverage/width on the fold's test plots.
    """
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=float)
    groups = dataset.groups.to_numpy()
    folds: List[Dict[str, float]] = []

    for k, (train_idx, test_idx) in enumerate(make_splits(dataset, "group", n_splits, seed)):
        fit_rel, cal_rel = _split_groups(groups[train_idx], calibration_frac, seed + k)
        fit_idx, cal_idx = train_idx[fit_rel], train_idx[cal_rel]
        if len(fit_idx) < 10 or len(cal_idx) < 5:
            continue
        est = spec.build(dataset.feature_names, seed)
        _fit(est, X[fit_idx], y[fit_idx], groups[fit_idx])
        cal_resid = np.abs(y[cal_idx] - est.predict(X[cal_idx]))
        # finite-sample corrected quantile
        n_cal = len(cal_resid)
        q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        half_width = float(np.quantile(cal_resid, q_level))
        pred = est.predict(X[test_idx])
        folds.append(_coverage_and_width(y[test_idx], pred - half_width, pred + half_width))

    return _aggregate(folds, method="split_conformal", alpha=alpha, model=spec.name)


def quantile_gbm_evaluate(
    dataset: Dataset,
    alpha: float = 0.10,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=float)
    folds: List[Dict[str, float]] = []

    def _q(q: float):
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0, random_state=seed,
        )

    for train_idx, test_idx in make_splits(dataset, "group", n_splits, seed):
        lo = _q(alpha / 2).fit(X[train_idx], y[train_idx])
        hi = _q(1 - alpha / 2).fit(X[train_idx], y[train_idx])
        lower = lo.predict(X[test_idx])
        upper = hi.predict(X[test_idx])
        upper = np.maximum(upper, lower)  # quantile crossing guard
        folds.append(_coverage_and_width(y[test_idx], lower, upper))

    return _aggregate(folds, method="quantile_gbm", alpha=alpha, model="hist_gbm_quantile")


def _aggregate(folds: List[Dict[str, float]], **meta) -> Dict[str, float]:
    if not folds:
        return {**meta, "coverage": float("nan"), "mean_width": float("nan"), "n_folds": 0}
    cov = np.array([f["coverage"] for f in folds])
    width = np.array([f["mean_width"] for f in folds])
    target = 1 - meta.get("alpha", 0.1)
    return {
        **meta,
        "target_coverage": target,
        "coverage": float(cov.mean()),
        "coverage_std": float(cov.std(ddof=1)) if len(cov) > 1 else 0.0,
        "mean_width": float(width.mean()),
        "n_folds": len(folds),
        "calibrated": bool(abs(cov.mean() - target) <= 0.05),
    }


def conformal_half_width(spec: ModelSpec, dataset: Dataset, alpha: float = 0.10, seed: int = 42,
                         calibration_frac: float = 0.3) -> Optional[float]:
    """Single calibrated half-width for deployment (fit on all plots minus a calibration slice)."""
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=float)
    groups = dataset.groups.to_numpy()
    fit_idx, cal_idx = _split_groups(groups, calibration_frac, seed)
    if len(fit_idx) < 10 or len(cal_idx) < 5:
        return None
    est = spec.build(dataset.feature_names, seed)
    _fit(est, X[fit_idx], y[fit_idx], groups[fit_idx])
    resid = np.abs(y[cal_idx] - est.predict(X[cal_idx]))
    n_cal = len(resid)
    q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return float(np.quantile(resid, q_level))
