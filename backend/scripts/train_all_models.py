"""
scripts/train_all_models.py
============================
Command-line entry point for training RGR models.

Examples
--------
# Train all configured models (8 species + 5 forest-unit)
python scripts/train_all_models.py --all

# Train only one species model (across all valid forest units)
python scripts/train_all_models.py --species Mh

# Train only one forest-unit model (across all species)
python scripts/train_all_models.py --forest-unit-model TH1

# Train without saving to database (dry-run)
python scripts/train_all_models.py --all --no-save

# Verbose output
python scripts/train_all_models.py --all --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPORTS_DIR = os.path.join(_BACKEND_DIR, "artifacts", "reports")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from training.pipeline import RGRTrainingPipeline
from data.extractors import SPECIES_CONFIG, FOREST_UNITS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train configured RGR coefficient models (species or forest-unit)."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",         action="store_true", help="Full re-train of all configured models.")
    group.add_argument("--incremental", action="store_true", help="Only retrain where new DB data exists since last training.")
    group.add_argument("--species",     type=str, help=f"Train one species model (across all FUs): {list(SPECIES_CONFIG)}")
    group.add_argument("--forest-unit-model", dest="forest_unit_model", type=str,
                       help=f"Train one forest-unit model (across all species): {FOREST_UNITS}")
    p.add_argument("--no-save",   dest="save", action="store_false", default=True,
                   help="Skip saving coefficients to database (dry-run).")
    p.add_argument("--no-tcr",    dest="populate_tcr", action="store_false", default=True,
                   help="Do not enrich TCR from the spatial shapefile.")
    p.add_argument("--workers",   type=int, default=4,
                   help="Parallel training workers (default: 4). Each worker uses its own DB connection.")
    p.add_argument("--version",   type=str, default="1.0",
                   help="Model version tag to store (default: 1.0).")
    p.add_argument("--verbose",   action="store_true", help="Enable DEBUG logging.")
    p.add_argument("--output",    type=str, default=None,
                   help="Optional path to write results JSON file.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = RGRTrainingPipeline(
        populate_tcr=args.populate_tcr,
        model_version=args.version,
        n_workers=args.workers,
    )

    if args.all:
        results = pipeline.train_all_models(save_to_db=args.save)
    elif args.incremental:
        results = pipeline.train_incremental(save_to_db=args.save)
    else:
        if args.species:
            key = args.species
            result = pipeline.train_single_model(args.species, None, save_to_db=args.save)
        else:
            key = f"FU_{args.forest_unit_model}"
            result = pipeline.train_single_model(None, args.forest_unit_model, save_to_db=args.save)
        results = {key: result} if result else {}

    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(_REPORTS_DIR, f"training_results_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
