"""
scripts/benchmark_models.py
===========================
Compare candidate growth models on every model group with leakage-aware
cross-validation and write a leaderboard (JSON + Markdown).

Examples
--------
# Every available algorithm, all 13 model groups, plot-grouped 5-fold CV
python scripts/benchmark_models.py --all

# Quick run: a handful of fast models on two groups
python scripts/benchmark_models.py --group Mh --group FU_TH1 --fast

# Specific algorithms, extended feature set, temporal CV as the primary split
python scripts/benchmark_models.py --all --models ols,ridge,hist_gbm,xgboost --features extended --cv temporal

# List what the registry knows (and which optional back-ends are installed)
python scripts/benchmark_models.py --list-models

The data source is whatever the API uses (SQLite demo by default, PostgreSQL /
SQL Server via DATABASE_URL).  Nothing is written to the database.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection  # noqa: E402
from data.extractors import FOREST_UNIT_CONFIG, SPECIES_CONFIG, BAGrowthDataExtractor  # noqa: E402
from ml.benchmark import BenchmarkConfig, benchmark_all  # noqa: E402
from ml.evaluation import CV_STRATEGIES  # noqa: E402
from ml.features import feature_set_names  # noqa: E402
from ml.registry import DEFAULT_MODELS, FAST_MODELS, list_models  # noqa: E402
from ml.reports import write_json, write_markdown  # noqa: E402

logger = logging.getLogger("benchmark")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark candidate RGR models with grouped cross-validation.")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="Benchmark every species and forest-unit group.")
    scope.add_argument("--group", action="append", default=[],
                       help="Model group key, repeatable (e.g. Mh, Cedar, FU_TH1).")
    p.add_argument("--list-models", action="store_true", help="Print the model registry and exit.")
    p.add_argument("--models", type=str, default=None,
                   help=f"Comma-separated model names (default: all available). Known: {','.join(DEFAULT_MODELS)}")
    p.add_argument("--fast", action="store_true", help=f"Shortcut for --models {','.join(FAST_MODELS)}")
    p.add_argument("--cv", choices=CV_STRATEGIES, default="group", help="Primary CV strategy (default: group).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", choices=feature_set_names(), default="base")
    p.add_argument("--alpha", type=float, default=0.10, help="Prediction-interval miscoverage (default 0.10 → 90%%).")
    p.add_argument("--no-leakage-check", dest="leakage", action="store_false", default=True)
    p.add_argument("--no-temporal", dest="temporal", action="store_false", default=True)
    p.add_argument("--no-intervals", dest="intervals", action="store_false", default=True)
    p.add_argument("--no-importance", dest="importance", action="store_false", default=True)
    p.add_argument("--no-tcr", dest="populate_tcr", action="store_false", default=True,
                   help="Do not override TCR from the spatial lookup during extraction.")
    p.add_argument("--output", type=str, default=None, help="JSON output path (default: artifacts/benchmarks/).")
    p.add_argument("--markdown", type=str, default=None,
                   help="Markdown output path (default: artifacts/benchmarks/LEADERBOARD.md).")
    p.add_argument("--verbose", action="store_true")
    return p


def _resolve_groups(keys: List[str]) -> List[tuple]:
    combos = []
    for key in keys:
        if key in SPECIES_CONFIG:
            combos.append((key, None))
        elif key.startswith("FU_") and key[3:] in FOREST_UNIT_CONFIG:
            combos.append((None, key[3:]))
        elif key in FOREST_UNIT_CONFIG:
            combos.append((None, key))
        else:
            raise SystemExit(
                f"Unknown group '{key}'. Species: {list(SPECIES_CONFIG)}; forest units: "
                f"{[f'FU_{fu}' for fu in FOREST_UNIT_CONFIG]}"
            )
    return combos


def load_groups(combos: List[tuple], populate_tcr: bool) -> Dict[str, pd.DataFrame]:
    session = get_db_connection()
    extractor = BAGrowthDataExtractor(session, populate_tcr=populate_tcr)
    data: Dict[str, pd.DataFrame] = {}
    try:
        for species, fu in combos:
            key = species or f"FU_{fu}"
            try:
                df = extractor.extract_training_data(species, fu)
            except Exception as exc:  # noqa: BLE001
                logger.error("  %s: extraction failed – %s", key, exc)
                continue
            if df.empty:
                logger.warning("  %s: no rows", key)
                continue
            data[key] = df
    finally:
        session.close()
    return data


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")

    if args.list_models:
        for spec in list_models():
            flag = "available" if spec["available"] else f"missing '{spec['requires']}'"
            print(f"{spec['name']:16s} {spec['family']:9s} {flag:22s} {spec['description']}")
        return

    if not args.all and not args.group:
        raise SystemExit("Choose --all or at least one --group (or --list-models).")

    if args.fast:
        models = list(FAST_MODELS)
    elif args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(DEFAULT_MODELS)

    config = BenchmarkConfig(
        models=models,
        cv_strategy=args.cv,
        n_splits=args.folds,
        seed=args.seed,
        feature_set=args.features,
        alpha=args.alpha,
        leakage_check=args.leakage,
        temporal_check=args.temporal,
        intervals=args.intervals,
        importance=args.importance,
    )

    if args.all:
        combos = [(sp, None) for sp in SPECIES_CONFIG] + [(None, fu) for fu in FOREST_UNIT_CONFIG]
    else:
        combos = _resolve_groups(args.group)

    data = load_groups(combos, args.populate_tcr)
    if not data:
        raise SystemExit("No data extracted – is the database seeded?  (python scripts/seed_demo_db.py)")

    def _progress(group: str, stage: str) -> None:
        logger.info("  %-10s %s", group, stage)

    report = benchmark_all(data, config, progress=_progress)

    json_path = write_json(report, args.output)
    md_path = write_markdown(report, args.markdown)

    print("\nLeaderboard (primary CV = %s):" % config.cv_strategy)
    for key, rep in report["groups"].items():
        if rep.get("error"):
            print(f"  {key:10s} skipped – {rep['error']}")
            continue
        best = rep["models"][rep["best_model"]]["metrics_mean"]
        ols = rep["models"].get("ols", {}).get("metrics_mean", {})
        print(
            f"  {key:10s} best={rep['best_model']:14s} RMSE={best['rmse']:.5f}  R2={best['r2']:.3f}  "
            f"p95={best['p95_ae']:.5f}   | OLS RMSE={ols.get('rmse', float('nan')):.5f}"
        )
    print(f"\nOverall best by mean rank: {report['summary']['overall_best']}")
    print(f"JSON     → {json_path}")
    print(f"Markdown → {md_path}")


if __name__ == "__main__":
    main()
