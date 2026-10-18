"""
db/seed.py
==========
Synthetic demo dataset – 100% generated, no real inventory data.

Seeds one coefficient row (plus history / training-history snapshots) and a
set of synthetic growth intervals for every model the API knows about
(``SPECIES_CONFIG`` + ``FOREST_UNIT_CONFIG`` in ``data/extractors.py``), so
that the demo is internally consistent: every model group listed by
``/api/species-list`` has both stored coefficients and growth data.

Everything is generated from a fixed RNG seed → re-seeding gives identical rows.

Realism knobs (so the model benchmark has something to discriminate):
  * a random intercept per plot (repeated-measures structure – the reason the
    benchmark uses plot-grouped cross-validation),
  * a mild saturating density effect and a cut × stocking interaction on top of
    the linear Penner-style equation, so non-linear learners can earn their keep,
  * heteroscedastic, occasionally heavy-tailed noise (measurement error,
    windthrow, insect damage).
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from db.models import (
    BAGrowthInterval,
    RGRCoefficient,
    RGRCoefficientHistory,
    RGRFrontendCalcData,
    RGRFrontendCalcPlotData,
    RGRPrediction,
    RGRTrainingHistory,
)

DEMO_MODEL_VERSION = "1.0"
DEMO_NOTE = "Synthetic demo model – generated data, not fitted to real inventory plots"


def demo_database_is_empty(session: Session) -> bool:
    return session.execute(select(func.count()).select_from(RGRCoefficient)).scalar_one() == 0


def _model_catalogue() -> List[Dict[str, Any]]:
    """Species + forest-unit models exactly as the API/extractor define them."""
    from data.extractors import FOREST_UNIT_CONFIG, SPECIES_CONFIG  # lazy: heavy geo imports

    catalogue: List[Dict[str, Any]] = []
    for abbrev, cfg in SPECIES_CONFIG.items():
        catalogue.append(
            {"key": abbrev, "spec_code": int(cfg["code"]), "forest_unit": None, "name": cfg["name"]}
        )
    for fu, cfg in FOREST_UNIT_CONFIG.items():
        catalogue.append({"key": f"FU_{fu}", "spec_code": None, "forest_unit": fu, "name": cfg["name"]})
    return catalogue


def _synthetic_coefficients(rng: random.Random) -> Dict[str, float]:
    """Coefficients with the biologically expected signs (Penner 2023)."""
    return {
        "intercept": rng.uniform(0.030, 0.055),
        "BA": -rng.uniform(0.00025, 0.00060),
        "BA2": rng.uniform(0.5e-6, 2.0e-6),
        "AGSfrac": rng.uniform(0.006, 0.016),
        "TCR": rng.uniform(0.0008, 0.0025),
        "DQ": -rng.uniform(0.00010, 0.00030),
        "q": rng.uniform(0.0015, 0.0045),
        "Cut": -rng.uniform(0.002, 0.006),
    }


def _apply_equation(c: Dict[str, float], ba: float, ags: float, tcr: int, dq: float, q: float, cut: int) -> float:
    return (
        c["intercept"]
        + c["BA"] * ba
        + c["BA2"] * ba * ba
        + c["AGSfrac"] * ags
        + c["TCR"] * tcr
        + c["DQ"] * dq
        + c["q"] * q
        + c["Cut"] * cut
    )


# --- realism parameters (documented in documentation/model_benchmarking.md) ---
PLOT_EFFECT_SD = 0.006        # sd of the per-plot random intercept
NOISE_SD_BASE = 0.003         # residual sd at BA → 0
NOISE_SD_PER_BA = 0.00008     # residual sd grows with stand density
HEAVY_TAIL_PROB = 0.06        # share of intervals hit by a disturbance
HEAVY_TAIL_SCALE = 3.0        # noise multiplier for those intervals
SATURATION_SCALE = 12.0       # m²/ha; density effect flattens beyond this
SATURATION_GAIN = 0.010
CUT_STOCKING_INTERACTION = 0.012


def _nonlinear_terms(ba: float, ags: float, cut: int) -> float:
    """Departures from the linear equation that the benchmark is meant to detect."""
    import math

    saturation = SATURATION_GAIN * math.exp(-ba / SATURATION_SCALE)
    interaction = CUT_STOCKING_INTERACTION * cut * (ags - 0.6)
    return saturation + interaction


def _fit_ols(rows: List[BAGrowthInterval]):
    """
    Ordinary least squares on the generated intervals – the same equation the
    production ``RGRModel`` fits – returning (coefficients, metrics).
    """
    import numpy as np

    X = np.array([[1.0, r.ba, r.ba * r.ba, r.agsfrac, r.tcr, r.dq, r.q, r.cut] for r in rows], dtype=float)
    y = np.array([r.rgr for r in rows], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    coefficients = {
        "intercept": float(beta[0]),
        "BA": float(beta[1]),
        "BA2": float(beta[2]),
        "AGSfrac": float(beta[3]),
        "TCR": float(beta[4]),
        "DQ": float(beta[5]),
        "q": float(beta[6]),
        "Cut": float(beta[7]),
    }
    metrics = {
        "r_squared": round(1.0 - float(np.sum(resid ** 2)) / ss_tot, 4) if ss_tot > 0 else 0.0,
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 5),
        "mae": round(float(np.mean(np.abs(resid))), 5),
    }
    return coefficients, metrics


def _confidence(n_samples: int) -> str:
    if n_samples >= 100:
        return "HIGH"
    if n_samples >= 50:
        return "MEDIUM"
    if n_samples >= 30:
        return "LOW"
    return "EXPERIMENTAL"


def _growth_rows(
    rng: random.Random,
    model: Dict[str, Any],
    coefficients: Dict[str, float],
    n_plots: int,
    plot_key_base: int,
    forest_units: List[str],
) -> List[BAGrowthInterval]:
    rows: List[BAGrowthInterval] = []
    for i in range(n_plots):
        plot_key = plot_key_base + i
        plot_effect = rng.gauss(0.0, PLOT_EFFECT_SD)  # unobserved site quality
        plot_fu = model["forest_unit"] or rng.choice(forest_units)
        # Synthetic Ontario-ish UTM 17N coordinates – not real plot locations.
        easting = rng.uniform(520_000, 690_000)
        northing = rng.uniform(4_930_000, 5_120_000)
        latitude = 44.5 + (northing - 4_930_000) / 111_000
        longitude = -80.7 + (easting - 520_000) / 78_000

        ba = rng.uniform(9.0, 42.0)
        # 3 intervals of at most 8 years from a start ≤ 2000 keeps EndYear ≤ 2024.
        start_year = rng.randint(1986, 2000)
        tcr = rng.randint(1, 7)
        n_intervals = rng.choice([2, 3, 3])
        for _ in range(n_intervals):
            interval = rng.randint(4, 8)
            end_year = start_year + interval
            ags = round(rng.uniform(0.30, 0.92), 3)
            dq = round(rng.uniform(12.0, 38.0), 1)
            q = round(rng.uniform(1.05, 2.4), 2)
            cut = 1 if rng.random() < 0.25 else 0
            rgr = _apply_equation(coefficients, ba, ags, tcr, dq, q, cut)
            rgr += _nonlinear_terms(ba, ags, cut)
            rgr += plot_effect
            noise_sd = NOISE_SD_BASE + NOISE_SD_PER_BA * ba
            if rng.random() < HEAVY_TAIL_PROB:
                noise_sd *= HEAVY_TAIL_SCALE
            rgr += rng.gauss(0.0, noise_sd)
            rgr = max(-0.15, min(0.25, rgr))
            tree_count = int(ba * rng.uniform(4.0, 7.0))
            rows.append(
                BAGrowthInterval(
                    spec_code=model["spec_code"],
                    forest_unit=plot_fu,
                    plot_key=plot_key,
                    plot_name=f"DEMO_{model['key']}_{plot_key}",
                    dataset_code="SYNTH",
                    start_year=start_year,
                    end_year=end_year,
                    growth_interval_years=float(interval),
                    ba=round(ba, 2),
                    agsfrac=ags,
                    tcr=tcr,
                    dq=dq,
                    q=q,
                    cut=cut,
                    bagrowth=round(rgr * ba, 4),
                    rgr=round(rgr, 5),
                    latitude=round(latitude, 5),
                    longitude=round(longitude, 5),
                    easting=round(easting, 1),
                    northing=round(northing, 1),
                    tree_count=tree_count,
                    shrinkage_count=rng.randint(0, 4),
                )
            )
            # next interval starts where this one ended, BA grows accordingly
            ba = max(5.0, ba + rgr * ba * interval * rng.uniform(0.6, 1.0))
            start_year = end_year
    return rows


def clear_demo_data(session: Session) -> None:
    """Delete all rows in dependency order (children first)."""
    for model in (
        RGRFrontendCalcPlotData,
        RGRFrontendCalcData,
        RGRPrediction,
        RGRTrainingHistory,
        RGRCoefficientHistory,
        RGRCoefficient,
        BAGrowthInterval,
    ):
        session.execute(delete(model))
    session.flush()


def seed_demo_database(
    session: Session,
    *,
    reset: bool = True,
    rng_seed: int = 42,
    n_plots_species: int = 60,
    n_plots_forest_unit: int = 48,
    model_version: str = DEMO_MODEL_VERSION,
) -> Dict[str, int]:
    """
    Populate the database with the synthetic demo dataset via the ORM.

    Returns ``{"models": n, "growth_rows": n, "training_history": n, "coefficient_history": n}``.
    The caller owns the transaction (commit/rollback).
    """
    rng = random.Random(rng_seed)
    catalogue = _model_catalogue()
    forest_units = [m["forest_unit"] for m in catalogue if m["forest_unit"]]

    if reset:
        clear_demo_data(session)

    growth_rows_total = 0
    for index, model in enumerate(catalogue):
        coefficients = _synthetic_coefficients(rng)
        n_plots = n_plots_species if model["spec_code"] is not None else n_plots_forest_unit
        growth_rows = _growth_rows(
            rng, model, coefficients, n_plots, plot_key_base=1000 + index * 1000, forest_units=forest_units
        )
        session.add_all(growth_rows)
        growth_rows_total += len(growth_rows)

        n_samples = len(growth_rows)
        ba_values = [row.ba for row in growth_rows if row.ba is not None]
        # Store the coefficients the production OLS would actually fit to these
        # rows (not the generating parameters) so stored metrics are honest.
        coefficients, metrics = _fit_ols(growth_rows)
        confidence = _confidence(n_samples)

        coefficient = RGRCoefficient(
            spec_code=model["spec_code"],
            forest_unit=model["forest_unit"],
            species_name=model["name"],
            intercept=coefficients["intercept"],
            coef_ba=coefficients["BA"],
            coef_ba2=coefficients["BA2"],
            coef_agsfrac=coefficients["AGSfrac"],
            coef_tcr=coefficients["TCR"],
            coef_dq=coefficients["DQ"],
            coef_q=coefficients["q"],
            coef_cut=coefficients["Cut"],
            r_squared=metrics["r_squared"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            n_samples=n_samples,
            model_version=model_version,
            max_data_year=max(row.end_year for row in growth_rows),
            notes=DEMO_NOTE,
            confidence_level=confidence,
            is_production_ready=confidence in {"HIGH", "MEDIUM", "LOW"},
            quality_flags=None,
            warning_message=None,
            wrong_sign_count=0,
            n_plots=n_plots,
        )
        session.add(coefficient)
        session.flush()  # assigns CoefficientID

        session.add(
            RGRCoefficientHistory(
                coefficient_id=coefficient.coefficient_id,
                spec_code=coefficient.spec_code,
                forest_unit=coefficient.forest_unit,
                species_name=coefficient.species_name,
                intercept=coefficient.intercept,
                coef_ba=coefficient.coef_ba,
                coef_ba2=coefficient.coef_ba2,
                coef_agsfrac=coefficient.coef_agsfrac,
                coef_tcr=coefficient.coef_tcr,
                coef_dq=coefficient.coef_dq,
                coef_q=coefficient.coef_q,
                coef_cut=coefficient.coef_cut,
                r_squared=coefficient.r_squared,
                rmse=coefficient.rmse,
                mae=coefficient.mae,
                n_samples=coefficient.n_samples,
                model_version=coefficient.model_version,
                max_data_year=coefficient.max_data_year,
                notes=coefficient.notes,
                confidence_level=coefficient.confidence_level,
                is_production_ready=coefficient.is_production_ready,
                quality_flags=coefficient.quality_flags,
                warning_message=coefficient.warning_message,
                wrong_sign_count=coefficient.wrong_sign_count,
                n_plots=coefficient.n_plots,
            )
        )
        session.add(
            RGRTrainingHistory(
                coefficient_id=coefficient.coefficient_id,
                n_plots=n_plots,
                n_growth_intervals=n_samples,
                avg_growth_interval_years=sum(r.growth_interval_years for r in growth_rows) / n_samples,
                min_ba=min(ba_values),
                max_ba=max(ba_values),
                avg_ba=sum(ba_values) / len(ba_values),
                data_quality_flags=json.dumps({"status": "SYNTHETIC_DEMO_OK", "n_plots": n_plots}),
            )
        )

    session.flush()
    return {
        "models": len(catalogue),
        "growth_rows": growth_rows_total,
        "training_history": len(catalogue),
        "coefficient_history": len(catalogue),
    }


def seed_summary(session: Session) -> Dict[str, Optional[int]]:
    """Row counts for the seeded tables (used by the CLI + tests)."""
    return {
        "tblRGRCoefficients": session.execute(select(func.count()).select_from(RGRCoefficient)).scalar_one(),
        "tblBAGrowthData": session.execute(select(func.count()).select_from(BAGrowthInterval)).scalar_one(),
        "tblRGRTrainingHistory": session.execute(
            select(func.count()).select_from(RGRTrainingHistory)
        ).scalar_one(),
        "tblRGRCoefficientHistory": session.execute(
            select(func.count()).select_from(RGRCoefficientHistory)
        ).scalar_one(),
    }
