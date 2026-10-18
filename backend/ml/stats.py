"""
ml/stats.py
===========
Statistical comparison helpers so a leaderboard difference is not mistaken
for a real one.

* ``paired_fold_test``   – paired t-test on per-fold RMSE (model vs baseline);
                           folds are the same for both models.
* ``wilcoxon_oof``       – Wilcoxon signed-rank test on per-sample absolute
                           out-of-fold errors (non-parametric, many more pairs).
* ``bootstrap_ci``       – percentile bootstrap CI for RMSE / p95 of OOF errors.
* ``rank_models``        – deterministic ranking by RMSE then p95 absolute error.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sps


def paired_fold_test(model_folds: Sequence[float], baseline_folds: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(model_folds, dtype=float)
    b = np.asarray(baseline_folds, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"t_stat": None, "p_value": None, "n_folds": int(len(a))}
    diff = a - b
    if np.allclose(diff, 0):
        return {"t_stat": 0.0, "p_value": 1.0, "n_folds": int(len(a))}
    t_stat, p_value = sps.ttest_rel(a, b)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n_folds": int(len(a))}


def wilcoxon_oof(model_oof: np.ndarray, baseline_oof: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    mask = ~np.isnan(model_oof) & ~np.isnan(baseline_oof)
    if mask.sum() < 10:
        return {"statistic": None, "p_value": None, "n": int(mask.sum())}
    err_model = np.abs(y[mask] - model_oof[mask])
    err_base = np.abs(y[mask] - baseline_oof[mask])
    diff = err_model - err_base
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(mask.sum())}
    try:
        statistic, p_value = sps.wilcoxon(err_model, err_base, zero_method="zsplit")
    except ValueError:
        return {"statistic": None, "p_value": None, "n": int(mask.sum())}
    return {"statistic": float(statistic), "p_value": float(p_value), "n": int(mask.sum())}


def bootstrap_ci(y: np.ndarray, oof: np.ndarray, n_boot: int = 500, seed: int = 42,
                 level: float = 0.95) -> Dict[str, List[float]]:
    mask = ~np.isnan(oof)
    resid = y[mask] - oof[mask]
    n = len(resid)
    if n < 10:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = resid[idx]
    rmse = np.sqrt(np.mean(samples ** 2, axis=1))
    p95 = np.percentile(np.abs(samples), 95, axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "rmse": [float(np.percentile(rmse, lo)), float(np.percentile(rmse, hi))],
        "p95_ae": [float(np.percentile(p95, lo)), float(np.percentile(p95, hi))],
        "level": [level],
    }


def rank_models(rows: List[Dict], primary: str = "rmse", secondary: str = "p95_ae") -> List[Dict]:
    """Attach ``rank`` (1 = best) to a list of dicts holding ``metrics_mean``."""
    def _key(row):
        m = row.get("metrics_mean", {})
        return (m.get(primary, float("inf")), m.get(secondary, float("inf")))

    ordered = sorted(rows, key=_key)
    for position, row in enumerate(ordered, start=1):
        row["rank"] = position
    return ordered


def relative_improvement(model_value: float, baseline_value: float) -> Optional[float]:
    """Percent reduction relative to the baseline (positive = model better) for lower-is-better metrics."""
    if baseline_value is None or baseline_value == 0 or model_value is None:
        return None
    return float((baseline_value - model_value) / abs(baseline_value) * 100.0)
