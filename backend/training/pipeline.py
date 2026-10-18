"""
training/pipeline.py
=====================
Orchestrates the end-to-end RGR model training pipeline:
  1. Extract data via BAGrowthDataExtractor (data/extractors.py)
  2. Train RGRModel (models/rgr_model.py)
  3. Persist coefficients via CoefficientStore (models/coefficient_store.py)

Supports:
  - Parallel training   (n_workers > 1, each worker gets its own connection)
  - Incremental updates (only retrain where new DB data exists since last run)

Uses the existing database connection from DatabaseLinker/main.py.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection
from data.extractors import BAGrowthDataExtractor, SPECIES_CONFIG, FOREST_UNIT_CONFIG, VALID_COMBINATIONS
from models.rgr_model import RGRModel
from models.coefficient_store import CoefficientStore

logger = logging.getLogger(__name__)

_DIVIDER = "=" * 72


# ---------------------------------------------------------------------------
# Worker function (runs in thread; creates its own connection)
# ---------------------------------------------------------------------------

def _train_one(
    species_abbrev: Optional[str],
    forest_unit: Optional[str],
    populate_tcr: bool,
    model_version: str,
    save_to_db: bool,
    since_year: int = 0,
) -> Tuple[str, Optional[Dict]]:
    """
    Train a single model with confidence assessment and quality flags.
    """
    is_fu_model = species_abbrev is None
    if is_fu_model:
        key = f"FU_{forest_unit}"
        cfg = FOREST_UNIT_CONFIG[forest_unit]
    else:
        key = species_abbrev
        cfg = SPECIES_CONFIG[species_abbrev]

    # Lower threshold for experimental models
    min_samples = 15  # Changed from 30 to allow Ab/Or

    # Each thread gets an independent connection
    conn = get_db_connection()
    extractor  = BAGrowthDataExtractor(conn, populate_tcr=populate_tcr)
    coef_store = CoefficientStore(conn)

    try:
        data = extractor.extract_training_data(species_abbrev, forest_unit, since_year)
    except Exception as exc:
        logger.error("  [%s] Extraction failed: %s", key, exc)
        conn.close()
        return key, None

    if len(data) < min_samples:
        logger.warning("  [%s] Skipped – %d rows (need >= %d)", key, len(data), min_samples)
        conn.close()
        return key, None

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    model = RGRModel(min_samples=min_samples)
    try:
        result = model.train(data)
    except ValueError as exc:
        logger.error("  [%s] Training failed: %s", key, exc)
        conn.close()
        return key, None

    # =========================================================================
    # ASSESS QUALITY & CONFIDENCE
    # =========================================================================
    n_samples = result["n_samples"]
    n_plots = int(data["PlotKey"].nunique())
    r_squared = result["r_squared"]

    # Determine confidence level
    if n_samples >= 100:
        confidence = "HIGH"
    elif n_samples >= 50:
        confidence = "MEDIUM"
    elif n_samples >= 30:
        confidence = "LOW"
    else:
        confidence = "EXPERIMENTAL"

    # Check coefficient signs (Penner 2023 expected signs)
    coeffs = model.coefficients
    expected_signs = {
        "BA": "negative",
        "AGSfrac": "positive",
        "TCR": "positive",
        "DQ": "negative",
        "q": "positive",
        "Cut": "negative",
    }

    wrong_signs = []
    for coef_name, expected_sign in expected_signs.items():
        value = coeffs.get(coef_name, 0)
        if expected_sign == "positive" and value < 0:
            wrong_signs.append(f"{coef_name}(-)")
        elif expected_sign == "negative" and value > 0:
            wrong_signs.append(f"{coef_name}(+)")

    # Build quality flags
    quality_flags = []
    warnings = []

    if n_samples < 30:
        quality_flags.append("SAMPLE_SIZE_CRITICAL")
        warnings.append(f"⚠️ CRITICAL: Only {n_samples} samples (minimum 30 recommended, 50+ preferred)")
    elif n_samples < 50:
        quality_flags.append("SAMPLE_SIZE_SMALL")
        warnings.append(f"⚠️ Small sample size ({n_samples}) may affect reliability")

    if r_squared < 0.10:
        quality_flags.append("VERY_POOR_FIT")
        warnings.append(f"⚠️ R² = {r_squared:.3f} - model has very poor predictive power")
    elif r_squared < 0.15:
        quality_flags.append("POOR_FIT")
        warnings.append(f"⚠️ R² = {r_squared:.3f} - model has poor predictive power")
    elif r_squared < 0.20:
        quality_flags.append("LOW_FIT")

    if len(wrong_signs) >= 3:
        quality_flags.append("MANY_WRONG_SIGNS")
        warnings.append(
            f"⚠️ {len(wrong_signs)}/6 coefficients have biologically implausible signs: {', '.join(wrong_signs)}"
        )
    elif len(wrong_signs) > 0:
        quality_flags.append("SOME_WRONG_SIGNS")
        warnings.append(f"⚠️ Wrong coefficient signs: {', '.join(wrong_signs)}")

    # Production readiness
    is_production = (
        confidence in ["HIGH", "MEDIUM", "LOW"]
        and r_squared >= 0.15
        and len(wrong_signs) <= 2
    )

    if not is_production:
        if confidence == "EXPERIMENTAL":
            warnings.append("⚠️ DO NOT use for operational forest management planning")
            warnings.append("ℹ️ Available for research and exploration only")
        quality_flags.append("NOT_PRODUCTION_READY")

    # Warning message for database
    warning_msg = " | ".join(warnings) if warnings else None

    # Log results
    logger.info(
        "  [%s] R²=%.4f  RMSE=%.5f  N=%d  Plots=%d  Confidence=%s  WrongSigns=%d  %s",
        key,
        r_squared,
        result["rmse"],
        n_samples,
        n_plots,
        confidence,
        len(wrong_signs),
        "✅ PRODUCTION" if is_production else "❌ NOT PROD",
    )

    if warnings:
        for warning in warnings:
            logger.warning("  [%s] %s", key, warning)

    # Add metadata to result
    result.update(
        {
            "confidence_level": confidence,
            "is_production_ready": is_production,
            "quality_flags": quality_flags,
            "warnings": warnings,
            "warning_message": warning_msg,
            "wrong_sign_count": len(wrong_signs),
            "wrong_signs": wrong_signs,
            "n_plots": n_plots,
        }
    )

    # =========================================================================
    # PERSIST
    # =========================================================================
    if save_to_db:
        max_year = int(data["EndYear"].max()) if not data.empty else None
        coef_id = coef_store.save_coefficients(
            spec_code=cfg.get("code"),
            forest_unit=forest_unit,
            species_name=cfg["name"],
            coefficients=model.coefficients,
            metrics=model.metrics,
            model_version=model_version,
            max_data_year=max_year,
            notes=f"Trained on {n_samples} intervals, {n_plots} plots",
            confidence_level=confidence,
            is_production_ready=is_production,
            quality_flags=",".join(quality_flags) if quality_flags else None,
            warning_message=warning_msg,
            wrong_sign_count=len(wrong_signs),
            n_plots=n_plots,
        )
        coef_store.save_training_history(coef_id, data)

    conn.close()
    return key, result


class RGRTrainingPipeline:
    """
    Train and persist RGR linear regression models.

    Parameters
    ----------
    connection : pyodbc.Connection, optional
        Reuses an existing connection for single-worker use.
        Parallel workers always create their own connections.
    populate_tcr : bool
        Whether to enrich TCR from the spatial shapefile via FindTCR.py.
    model_version : str
        Version tag stored alongside the coefficients.
    n_workers : int
        Number of parallel threads for training.  Defaults to 4.
        Each worker opens its own database connection, so values > 1
        are safe with pyodbc.
    """

    def __init__(
        self,
        connection=None,
        populate_tcr: bool = True,
        model_version: str = "1.0",
        n_workers: int = 4,
    ) -> None:
        self._conn         = connection or get_db_connection()
        self._populate_tcr = populate_tcr
        self.model_version = model_version
        self.n_workers     = max(1, int(n_workers))

        # Convenience objects backed by the primary connection
        self.extractor  = BAGrowthDataExtractor(self._conn, populate_tcr=populate_tcr)
        self.coef_store = CoefficientStore(self._conn)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_single_model(
        self,
        species_abbrev: Optional[str],
        forest_unit: Optional[str] = None,
        save_to_db: bool = True,
        since_year: int = 0,
    ) -> Optional[Dict]:
        """
        Train one model (species or forest-unit) using the shared connection.

        Species model : train_single_model('Mh')
        FU model      : train_single_model(None, 'TH1')
        """
        is_fu = species_abbrev is None
        if is_fu:
            cfg   = FOREST_UNIT_CONFIG[forest_unit]
            label = f"FU {forest_unit}"
        else:
            cfg   = SPECIES_CONFIG[species_abbrev]
            label = f"{cfg['name']} ({species_abbrev})"
        logger.info("\n%s", _DIVIDER)
        logger.info("Training  %s", label)
        logger.info(_DIVIDER)

        _, result = _train_one(
            species_abbrev, forest_unit,
            populate_tcr = self._populate_tcr,
            model_version= self.model_version,
            save_to_db   = save_to_db,
            since_year   = since_year,
        )
        return result

    def train_all_models(self, save_to_db: bool = True) -> Dict[str, Dict]:
        """
        Full re-train all 13 models:
          8 species models (Mh, Or, By, He, Ab, Mr, Cedar, Be)
          5 forest-unit models (TH1, HDSL1, HDSL2, CE1, HE1)
        Uses n_workers parallel threads when n_workers > 1.
        """
        return self._run_batch(VALID_COMBINATIONS, save_to_db, since_year=0)

    def train_incremental(self, save_to_db: bool = True) -> Dict[str, Dict]:
        """
        Incremental update – only retrain combinations where new field data
        has been collected since the model was last trained.

        For each combination:
          - Loads stored MaxDataYear from tblRGRCoefficients
          - Queries current max available year from the source tables
          - If (available_max > stored_max) OR (no model stored): retrains
          - Passes @SinceYear = stored_max to the SP so only new rows are
            fetched, then merges into a fresh full model

        Note: Even in incremental mode the *model* is always retrained from
        scratch on the full dataset that qualifies after the stored year.  
        True online learning is unsupported by scikit-learn LinearRegression.
        """
        logger.info("\n%s", _DIVIDER)
        logger.info("INCREMENTAL UPDATE – checking %d combinations ...", len(VALID_COMBINATIONS))
        logger.info(_DIVIDER)

        needs_update: List[Tuple[str, str, int]] = []

        for species_abbrev, forest_unit in VALID_COMBINATIONS:
            # FU models have species_abbrev=None → must use FOREST_UNIT_CONFIG
            if species_abbrev is not None:
                cfg = SPECIES_CONFIG[species_abbrev]
                spec_code = cfg["code"]
            else:
                cfg = FOREST_UNIT_CONFIG[forest_unit]
                spec_code = None
            model_info = self.coef_store.get_model_info(
                spec_code, forest_unit, self.model_version
            )
            stored_year   = model_info["max_data_year"] if model_info else 0
            available_year = self.extractor.get_max_available_year(species_abbrev)

            if model_info is None:
                logger.info(
                    "  %-22s  no model stored  → full train",
                    f"{species_abbrev}/{forest_unit}",
                )
                needs_update.append((species_abbrev, forest_unit, 0))
            elif available_year > stored_year:
                logger.info(
                    "  %-22s  new data %d → %d  → retrain",
                    f"{species_abbrev}/{forest_unit}", stored_year, available_year,
                )
                needs_update.append((species_abbrev, forest_unit, stored_year))
            else:
                logger.info(
                    "  %-22s  up-to-date (max=%d)  Ο",
                    f"{species_abbrev}/{forest_unit}", stored_year,
                )

        if not needs_update:
            logger.info("\n✓ All models are up-to-date. Nothing to retrain.")
            return {}

        logger.info(
            "\n→ Retraining %d / %d combination(s) ...",
            len(needs_update), len(VALID_COMBINATIONS),
        )
        combos = [(sp, fu) for sp, fu, _ in needs_update]
        since  = {
            (sp if sp is not None else f"FU_{fu}"): yr
            for sp, fu, yr in needs_update
        }
        return self._run_batch_with_since(combos, save_to_db, since)

    # ------------------------------------------------------------------
    # Internal batch runners
    # ------------------------------------------------------------------

    def _run_batch(
        self,
        combinations: List[Tuple],
        save_to_db: bool,
        since_year: int = 0,
    ) -> Dict[str, Dict]:
        """Run a batch with a fixed since_year for all combinations."""
        since = {
            (sp if sp is not None else f"FU_{fu}"): since_year
            for sp, fu in combinations
        }
        return self._run_batch_with_since(combinations, save_to_db, since)

    def _run_batch_with_since(
        self,
        combinations: List[Tuple],
        save_to_db: bool,
        since: Dict[str, int],
    ) -> Dict[str, Dict]:
        """Core batch runner; dispatches to serial or parallel execution."""
        logger.info("\n%s", _DIVIDER)
        logger.info("TRAINING %d MODEL(S)  [workers=%d]", len(combinations), self.n_workers)
        logger.info(_DIVIDER)

        results: Dict[str, Dict] = {}

        def _key(sp, fu):
            return sp if sp is not None else f"FU_{fu}"

        def _label(sp, fu):
            if sp is not None:
                return f"{SPECIES_CONFIG[sp]['name']} ({sp})"
            return f"FU {fu} ({FOREST_UNIT_CONFIG[fu]['name']})"

        if self.n_workers <= 1:
            for sp, fu in combinations:
                key = _key(sp, fu)
                logger.info("\n%s", _DIVIDER)
                logger.info("Training  %s", _label(sp, fu))
                logger.info(_DIVIDER)
                _, result = _train_one(
                    sp, fu,
                    populate_tcr = self._populate_tcr,
                    model_version= self.model_version,
                    save_to_db   = save_to_db,
                    since_year   = since.get(key, 0),
                )
                if result:
                    results[key] = result
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                for sp, fu in combinations:
                    key = _key(sp, fu)
                    future = pool.submit(
                        _train_one, sp, fu,
                        self._populate_tcr,
                        self.model_version,
                        save_to_db,
                        since.get(key, 0),
                    )
                    futures[future] = key

                for future in as_completed(futures):
                    key, result = future.result()
                    if result:
                        results[key] = result

        # Summary
        logger.info("\n%s", _DIVIDER)
        logger.info(
            "COMPLETE: %d/%d trained.",
            len(results), len(combinations),
        )
        logger.info(_DIVIDER)
        for key, res in sorted(results.items()):
            logger.info(
                "  %-22s  R²=%.4f  RMSE=%.4f  N=%d",
                key,
                res.get("r_squared", 0),
                res.get("rmse", 0),
                res.get("n_samples", 0),
            )
        return results

    def retrain_model(
        self,
        spec_code: int,
        forest_unit: str,
        save_to_db: bool = True,
    ) -> Optional[Dict]:
        """Retrain a specific model by spec_code (integer) and forest_unit."""
        abbrev = next(
            (k for k, v in SPECIES_CONFIG.items() if v["code"] == spec_code),
            None,
        )
        if abbrev is None:
            logger.error("Unknown SpecCode: %d", spec_code)
            return None
        return self.train_single_model(abbrev, forest_unit, save_to_db=save_to_db)
