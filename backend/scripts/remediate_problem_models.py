"""
scripts/remediate_problem_models.py
===================================
Runs targeted remediation diagnostics for problematic models by:
1) training a baseline linear model,
2) retraining after IQR outlier removal,
3) computing AGS data-quality diagnostics,
4) writing JSON + CSV reports with recommendations.

Examples
--------
python scripts/remediate_problem_models.py
python scripts/remediate_problem_models.py --models He Mr Cedar Be FU_HE1
python scripts/remediate_problem_models.py --iqr-multiplier 1.25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPORTS_DIR = os.path.join(_BACKEND_DIR, "artifacts", "reports")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection
from data.extractors import BAGrowthDataExtractor, SPECIES_CONFIG, FOREST_UNIT_CONFIG
from models.rgr_model import RGRModel

logger = logging.getLogger(__name__)

EXPECTED_SIGNS = {
    "BA": "negative",
    "AGSfrac": "positive",
    "TCR": "positive",
    "DQ": "negative",
    "q": "positive",
    "Cut": "negative",
}

DEFAULT_MODELS = ["He", "Mr", "Cedar", "Be", "FU_HE1"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remediation diagnostics for problematic models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys: species abbrev (He, Mr, Cedar, Be) or FU keys (FU_HE1).",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier filtering (default: 1.5).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum samples required to train (default: 15).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output filename prefix.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def _split_model_key(model_key: str) -> Tuple[Optional[str], Optional[str]]:
    if model_key.startswith("FU_"):
        return None, model_key.replace("FU_", "", 1)
    return model_key, None


def _wrong_signs(coefficients: Dict[str, float]) -> List[str]:
    wrong = []
    for name, expected in EXPECTED_SIGNS.items():
        value = float(coefficients.get(name, 0.0))
        if expected == "positive" and value < 0:
            wrong.append(f"{name}(-)")
        elif expected == "negative" and value > 0:
            wrong.append(f"{name}(+)")
    return wrong


def _fit_model(df: pd.DataFrame, min_samples: int) -> Dict:
    model = RGRModel(min_samples=min_samples)
    result = model.train(df)
    wrong = _wrong_signs(model.coefficients or {})
    return {
        "coefficients": model.coefficients,
        "metrics": {
            "r_squared": result.get("r_squared"),
            "rmse": result.get("rmse"),
            "mae": result.get("mae"),
            "n_samples": result.get("n_samples"),
            "n_plots": result.get("n_plots"),
        },
        "wrong_sign_count": len(wrong),
        "wrong_signs": wrong,
    }


def _remove_iqr_outliers(df: pd.DataFrame, multiplier: float) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    numeric_cols = ["BA", "AGSfrac", "TCR", "DQ", "q", "RGR"]
    filtered = df.copy()
    bounds: Dict[str, Dict[str, float]] = {}

    for col in numeric_cols:
        series = pd.to_numeric(filtered[col], errors="coerce")
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        low = q1 - multiplier * iqr
        high = q3 + multiplier * iqr
        bounds[col] = {"q1": q1, "q3": q3, "iqr": iqr, "low": low, "high": high}
        filtered = filtered[(filtered[col] >= low) & (filtered[col] <= high)]

    return filtered.copy(), bounds


def _ags_diagnostics(df: pd.DataFrame) -> Dict:
    ags_raw = pd.to_numeric(df["AGSfrac"], errors="coerce")
    missing_count = int(ags_raw.isna().sum())
    ags_median = ags_raw.median(skipna=True)
    ags_impute_value = float(ags_median) if pd.notna(ags_median) else 0.5
    ags_filled = ags_raw.fillna(ags_impute_value)
    imputed_count = int(ags_raw.isna().sum())

    tree_count = pd.to_numeric(df.get("TreeCount"), errors="coerce") if "TreeCount" in df.columns else pd.Series(dtype=float)
    shrinkage = pd.to_numeric(df.get("ShrinkageCount"), errors="coerce") if "ShrinkageCount" in df.columns else pd.Series(dtype=float)

    total_trees = float(tree_count.sum()) if not tree_count.empty else 0.0
    total_shrink = float(shrinkage.sum()) if not shrinkage.empty else 0.0
    shrinkage_rate = (total_shrink / total_trees) if total_trees > 0 else None

    return {
        "ags_missing_count": missing_count,
        "ags_missing_rate": float(missing_count / len(df)) if len(df) else 0.0,
        "ags_impute_value": ags_impute_value,
        "ags_imputed_rate": float(imputed_count / len(df)) if len(df) else 0.0,
        "ags_default_0_5_rate": float((ags_filled == 0.5).mean()) if len(df) else 0.0,
        "ags_low_lt_0_2_rate": float((ags_filled < 0.2).mean()) if len(df) else 0.0,
        "ags_high_gt_0_8_rate": float((ags_filled > 0.8).mean()) if len(df) else 0.0,
        "tree_count_median": float(tree_count.median()) if not tree_count.empty else None,
        "shrinkage_rate": float(shrinkage_rate) if shrinkage_rate is not None else None,
    }


def _recommendation(baseline: Dict, filtered: Optional[Dict], ags: Dict) -> str:
    if filtered is None:
        return "Insufficient samples after outlier filtering; collect more data before remediation."

    b_r2 = float(baseline["metrics"]["r_squared"] or 0.0)
    f_r2 = float(filtered["metrics"]["r_squared"] or 0.0)
    b_wrong = int(baseline["wrong_sign_count"])
    f_wrong = int(filtered["wrong_sign_count"])

    improved = f_r2 >= (b_r2 + 0.02)
    signs_better = f_wrong <= b_wrong

    if improved and signs_better:
        return "Use outlier-filtered model candidate for validation; improvement is material and signs are not worse."
    if ags.get("ags_missing_rate", 0.0) > 0.05 or ags.get("ags_imputed_rate", 0.0) > 0.20:
        return "Prioritize AGS QA (missing/default-heavy) before expecting stable coefficients."
    if f_wrong > 2:
        return "Coefficient signs remain unstable after outlier filtering; consider nonlinear terms and collect additional data."
    return "Outlier filtering gives limited benefit; test nonlinear features and interaction terms next."


def _run_one(
    extractor: BAGrowthDataExtractor,
    model_key: str,
    iqr_multiplier: float,
    min_samples: int,
) -> Dict:
    species_abbrev, forest_unit = _split_model_key(model_key)
    if species_abbrev is not None and species_abbrev not in SPECIES_CONFIG:
        raise ValueError(f"Unknown species model key: {model_key}")
    if forest_unit is not None and forest_unit not in FOREST_UNIT_CONFIG:
        raise ValueError(f"Unknown forest-unit model key: {model_key}")

    data = extractor.extract_training_data(species_abbrev, forest_unit, since_year=0)
    if data.empty:
        return {
            "model": model_key,
            "status": "no_data",
            "message": "No data returned from extractor.",
        }

    baseline = _fit_model(data, min_samples=min_samples)
    filtered_df, bounds = _remove_iqr_outliers(data, multiplier=iqr_multiplier)

    filtered_result: Optional[Dict] = None
    filter_message = None
    if len(filtered_df) >= min_samples:
        filtered_result = _fit_model(filtered_df, min_samples=min_samples)
    else:
        filter_message = f"Filtered sample size {len(filtered_df)} < min_samples {min_samples}"

    ags = _ags_diagnostics(data)
    recommendation = _recommendation(baseline, filtered_result, ags)

    return {
        "model": model_key,
        "status": "ok",
        "raw_rows": int(len(data)),
        "filtered_rows": int(len(filtered_df)),
        "rows_removed": int(len(data) - len(filtered_df)),
        "iqr_multiplier": iqr_multiplier,
        "filter_bounds": bounds,
        "baseline": baseline,
        "filtered": filtered_result,
        "filter_message": filter_message,
        "ags_diagnostics": ags,
        "recommendation": recommendation,
    }


def _to_summary_rows(results: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for item in results:
        if item.get("status") != "ok":
            rows.append(
                {
                    "model": item.get("model"),
                    "status": item.get("status"),
                    "r2_before": None,
                    "r2_after": None,
                    "wrong_signs_before": None,
                    "wrong_signs_after": None,
                    "rows_removed": None,
                    "ags_missing_rate": None,
                    "ags_imputed_rate": None,
                    "ags_impute_value": None,
                    "ags_default_0_5_rate": None,
                    "recommendation": item.get("message"),
                }
            )
            continue

        baseline = item["baseline"]
        filtered = item.get("filtered")
        rows.append(
            {
                "model": item["model"],
                "status": item["status"],
                "r2_before": baseline["metrics"].get("r_squared"),
                "r2_after": filtered["metrics"].get("r_squared") if filtered else None,
                "wrong_signs_before": baseline.get("wrong_sign_count"),
                "wrong_signs_after": filtered.get("wrong_sign_count") if filtered else None,
                "rows_removed": item.get("rows_removed"),
                "ags_missing_rate": item["ags_diagnostics"].get("ags_missing_rate"),
                "ags_imputed_rate": item["ags_diagnostics"].get("ags_imputed_rate"),
                "ags_impute_value": item["ags_diagnostics"].get("ags_impute_value"),
                "ags_default_0_5_rate": item["ags_diagnostics"].get("ags_default_0_5_rate"),
                "recommendation": item.get("recommendation"),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s", datefmt="%H:%M:%S")

    os.makedirs(_REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix or f"remediation_{ts}"
    json_path = os.path.join(_REPORTS_DIR, f"{prefix}.json")
    csv_path = os.path.join(_REPORTS_DIR, f"{prefix}_summary.csv")

    conn = get_db_connection()
    extractor = BAGrowthDataExtractor(conn, populate_tcr=True)
    results: List[Dict] = []

    try:
        for model_key in args.models:
            logger.info("Remediation run for %s", model_key)
            try:
                item = _run_one(
                    extractor=extractor,
                    model_key=model_key,
                    iqr_multiplier=args.iqr_multiplier,
                    min_samples=args.min_samples,
                )
                results.append(item)
            except Exception as exc:
                logger.error("  [%s] failed: %s", model_key, exc)
                results.append(
                    {
                        "model": model_key,
                        "status": "error",
                        "message": str(exc),
                    }
                )
    finally:
        conn.close()

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    summary = pd.DataFrame(_to_summary_rows(results))
    summary.to_csv(csv_path, index=False)

    print(f"✓ Detailed report: {json_path}")
    print(f"✓ Summary report : {csv_path}")


if __name__ == "__main__":
    main()
