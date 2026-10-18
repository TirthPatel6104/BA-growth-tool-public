"""
ml/benchmark.py
===============
Orchestrates the model comparison for one or many model groups.

For every model group (species or forest unit):

1. build the design matrix (``features.build_dataset``)
2. for each candidate algorithm run leakage-aware cross-validation
   (plot-grouped by default; temporal and naive K-fold optional)
3. rank by out-of-fold RMSE, tie-break by p95 absolute error
4. test whether each model is *significantly* different from the OLS baseline
5. for linear models, check biological sign plausibility of coefficients
6. quantify the leakage inflation of naive K-fold versus grouped CV
7. evaluate prediction intervals (split conformal around the best model and
   the OLS baseline, plus native quantile boosting)
8. compute permutation importance for the winning model

The output is a plain dict (JSON-serialisable) consumed by the CLI, the API
and the frontend "Model Benchmarks" tab.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import conformal
from .evaluation import (
    CV_STRATEGIES,
    cross_validate,
    fit_full,
    group_overlap,
    make_splits,
    sign_check,
)
from .features import Dataset, build_dataset, describe
from .registry import DEFAULT_MODELS, ModelSpec, available_models, linear_coefficients
from .stats import bootstrap_ci, paired_fold_test, rank_models, relative_improvement, wilcoxon_oof

logger = logging.getLogger(__name__)

BASELINE = "ols"


@dataclass
class BenchmarkConfig:
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    cv_strategy: str = "group"            # primary strategy used for ranking
    n_splits: int = 5
    seed: int = 42
    feature_set: str = "base"
    alpha: float = 0.10                   # prediction-interval miscoverage
    leakage_check: bool = True            # also run naive K-fold and report inflation
    temporal_check: bool = True           # also run forward-chaining CV
    intervals: bool = True
    importance: bool = True
    min_samples: int = 30

    def to_dict(self) -> Dict:
        return asdict(self)


ProgressCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# One model group
# ---------------------------------------------------------------------------

def benchmark_group(df: pd.DataFrame, key: str, config: Optional[BenchmarkConfig] = None,
                    progress: Optional[ProgressCallback] = None) -> Dict:
    config = config or BenchmarkConfig()
    if config.cv_strategy not in CV_STRATEGIES:
        raise ValueError(f"cv_strategy must be one of {CV_STRATEGIES}")

    dataset = build_dataset(df, config.feature_set)
    report: Dict = {
        "group": key,
        "dataset": describe(dataset),
        "config": config.to_dict(),
        "models": {},
        "skipped": {},
    }
    if dataset.n_samples < config.min_samples:
        report["error"] = f"Only {dataset.n_samples} usable rows (need >= {config.min_samples})"
        return report

    specs = available_models(config.models)
    if not any(s.name == BASELINE for s in specs):
        specs.insert(0, available_models([BASELINE])[0])

    primary_splits = make_splits(dataset, config.cv_strategy, config.n_splits, config.seed)
    report["cv"] = {
        "strategy": config.cv_strategy,
        "n_splits": len(primary_splits),
        "plot_overlap_between_train_and_test": group_overlap(dataset, primary_splits),
    }

    extra_splits = {}
    if config.leakage_check and config.cv_strategy != "kfold":
        extra_splits["kfold"] = make_splits(dataset, "kfold", config.n_splits, config.seed)
    if config.temporal_check and config.cv_strategy != "temporal":
        try:
            extra_splits["temporal"] = make_splits(dataset, "temporal", config.n_splits, config.seed)
        except ValueError as exc:
            report["cv"]["temporal_unavailable"] = str(exc)

    cv_results: Dict[str, Dict] = {}
    y = dataset.y.to_numpy(dtype=float)

    for spec in specs:
        if progress:
            progress(key, spec.name)
        t0 = time.perf_counter()
        try:
            primary = cross_validate(spec, dataset, primary_splits, config.cv_strategy, config.seed)
            extras = {
                name: cross_validate(spec, dataset, splits, name, config.seed)
                for name, splits in extra_splits.items()
            }
        except Exception as exc:  # noqa: BLE001 – one bad model must not kill the run
            logger.exception("[%s] %s failed", key, spec.name)
            report["skipped"][spec.name] = f"{type(exc).__name__}: {exc}"
            continue

        summary = primary.metric_summary()
        entry: Dict = {
            "family": spec.family,
            "description": spec.description,
            "linear": spec.linear,
            "metrics_mean": summary["mean"],
            "metrics_std": summary["std"],
            "pooled_oof": primary.pooled_metrics(dataset),
            "folds": primary.folds,
            "fit_seconds_total": round(primary.fit_seconds, 4),
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "other_strategies": {
                name: {"metrics_mean": res.metric_summary()["mean"], "pooled_oof": res.pooled_metrics(dataset)}
                for name, res in extras.items()
            },
            "bootstrap_ci": bootstrap_ci(y, primary.oof_pred, seed=config.seed),
        }

        if spec.linear:
            try:
                full = fit_full(spec, dataset, config.seed)
                coefs = linear_coefficients(full, dataset.feature_names)
                entry["coefficients"] = coefs
                entry["sign_check"] = sign_check(coefs)
                if hasattr(full, "random_effect_sd_"):
                    entry["random_effect_sd"] = float(full.random_effect_sd_)
                    entry["converged"] = bool(getattr(full, "converged_", True))
            except Exception as exc:  # noqa: BLE001
                entry["coefficients_error"] = str(exc)

        cv_results[spec.name] = {"entry": entry, "primary": primary}

    if not cv_results:
        report["error"] = "No model completed cross-validation"
        return report

    # ---- significance vs baseline ------------------------------------------------
    base = cv_results.get(BASELINE)
    for name, item in cv_results.items():
        entry = item["entry"]
        if base is None or name == BASELINE:
            entry["vs_baseline"] = None
            continue
        model_rmse = [f["rmse"] for f in item["primary"].folds]
        base_rmse = [f["rmse"] for f in base["primary"].folds]
        entry["vs_baseline"] = {
            "baseline": BASELINE,
            "rmse_improvement_pct": relative_improvement(
                entry["metrics_mean"].get("rmse"), base["entry"]["metrics_mean"].get("rmse")),
            "p95_improvement_pct": relative_improvement(
                entry["metrics_mean"].get("p95_ae"), base["entry"]["metrics_mean"].get("p95_ae")),
            "paired_t_on_fold_rmse": paired_fold_test(model_rmse, base_rmse),
            "wilcoxon_on_abs_errors": wilcoxon_oof(item["primary"].oof_pred, base["primary"].oof_pred, y),
        }

    # ---- ranking -------------------------------------------------------------------
    rows = [{"model": name, **item["entry"]} for name, item in cv_results.items()]
    ranked = rank_models(rows)
    report["models"] = {row["model"]: {k: v for k, v in row.items() if k != "model"} for row in ranked}
    report["ranking"] = [row["model"] for row in ranked]
    best_name = ranked[0]["model"]
    report["best_model"] = best_name
    best_spec = next(s for s in specs if s.name == best_name)

    # ---- leakage inflation -----------------------------------------------------------
    if "kfold" in extra_splits:
        inflation = {}
        for name, item in cv_results.items():
            grouped_r2 = item["entry"]["metrics_mean"].get("r2")
            naive_r2 = item["entry"]["other_strategies"].get("kfold", {}).get("metrics_mean", {}).get("r2")
            if grouped_r2 is not None and naive_r2 is not None:
                inflation[name] = {
                    "r2_grouped": grouped_r2,
                    "r2_naive_kfold": naive_r2,
                    "r2_inflation": float(naive_r2 - grouped_r2),
                }
        values = [v["r2_inflation"] for v in inflation.values()]
        report["leakage_check"] = {
            "explanation": (
                "Naive K-fold lets remeasurements of the same plot land in both train and test. "
                "Positive inflation means the naive score overstates real-world skill. "
                "The median is the headline figure; the mean is sensitive to degenerate models."
            ),
            "per_model": inflation,
            "median_r2_inflation": float(np.median(values)) if values else None,
            "mean_r2_inflation": float(np.mean(values)) if values else None,
        }

    # ---- prediction intervals ----------------------------------------------------------
    if config.intervals:
        if progress:
            progress(key, "prediction intervals")
        intervals: Dict[str, Dict] = {}
        try:
            intervals["split_conformal_best"] = conformal.split_conformal_evaluate(
                best_spec, dataset, config.alpha, config.n_splits, config.seed)
            if best_name != BASELINE:
                base_spec = next(s for s in specs if s.name == BASELINE)
                intervals["split_conformal_ols"] = conformal.split_conformal_evaluate(
                    base_spec, dataset, config.alpha, config.n_splits, config.seed)
            intervals["quantile_gbm"] = conformal.quantile_gbm_evaluate(
                dataset, config.alpha, config.n_splits, config.seed)
        except Exception as exc:  # noqa: BLE001
            intervals["error"] = str(exc)
        report["intervals"] = intervals

    # ---- permutation importance (best model, grouped hold-out) -------------------------
    if config.importance:
        if progress:
            progress(key, "permutation importance")
        try:
            report["importance"] = _permutation_importance(best_spec, dataset, primary_splits[0], config.seed)
        except Exception as exc:  # noqa: BLE001
            report["importance"] = {"error": str(exc)}

    return report


def _permutation_importance(spec: ModelSpec, dataset: Dataset, split, seed: int) -> Dict:
    train_idx, test_idx = split
    X = dataset.X.to_numpy(dtype=float)
    y = dataset.y.to_numpy(dtype=float)
    est = spec.build(dataset.feature_names, seed)
    from .evaluation import _fit  # local import avoids a circular reference at module load

    _fit(est, X[train_idx], y[train_idx], dataset.groups.to_numpy()[train_idx])
    result = permutation_importance(est, X[test_idx], y[test_idx], n_repeats=20,
                                    random_state=seed, scoring="neg_root_mean_squared_error")
    order = np.argsort(result.importances_mean)[::-1]
    return {
        "model": spec.name,
        "metric": "increase in RMSE when the feature is shuffled (held-out plots)",
        "features": [
            {
                "feature": dataset.feature_names[i],
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
            }
            for i in order
        ],
    }


# ---------------------------------------------------------------------------
# Many model groups
# ---------------------------------------------------------------------------

def benchmark_all(data: Dict[str, pd.DataFrame], config: Optional[BenchmarkConfig] = None,
                  progress: Optional[ProgressCallback] = None) -> Dict:
    config = config or BenchmarkConfig()
    started = time.perf_counter()
    groups: Dict[str, Dict] = {}
    for key, df in data.items():
        logger.info("Benchmarking %s (%d rows)", key, len(df))
        groups[key] = benchmark_group(df, key, config, progress)

    leaderboard: List[Dict] = []
    wins: Dict[str, int] = {}
    ranks: Dict[str, List[int]] = {}
    for key, rep in groups.items():
        for name, entry in rep.get("models", {}).items():
            m = entry["metrics_mean"]
            leaderboard.append({
                "group": key,
                "model": name,
                "family": entry["family"],
                "rank": entry["rank"],
                "rmse": m.get("rmse"),
                "rmse_std": entry["metrics_std"].get("rmse"),
                "r2": m.get("r2"),
                "mae": m.get("mae"),
                "p95_ae": m.get("p95_ae"),
                "bias": m.get("bias"),
                "rmse_improvement_pct": (entry.get("vs_baseline") or {}).get("rmse_improvement_pct"),
                "p_value_vs_ols": ((entry.get("vs_baseline") or {}).get("paired_t_on_fold_rmse") or {}).get("p_value"),
                "wrong_sign_count": (entry.get("sign_check") or {}).get("wrong_sign_count"),
            })
            ranks.setdefault(name, []).append(entry["rank"])
        if rep.get("best_model"):
            wins[rep["best_model"]] = wins.get(rep["best_model"], 0) + 1

    leaderboard.sort(key=lambda r: (r["group"], r["rank"]))
    mean_rank = {name: float(np.mean(r)) for name, r in ranks.items()}
    overall = sorted(mean_rank.items(), key=lambda kv: kv[1])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "n_groups": len(groups),
        "groups": groups,
        "leaderboard": leaderboard,
        "summary": {
            "wins_by_model": dict(sorted(wins.items(), key=lambda kv: -kv[1])),
            "mean_rank_by_model": dict(overall),
            "overall_best": overall[0][0] if overall else None,
            "baseline": BASELINE,
        },
    }
