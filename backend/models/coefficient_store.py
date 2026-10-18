"""
models/coefficient_store.py
============================
Persists fitted RGR coefficients to and retrieves them from the
tblRGRCoefficients, tblRGRCoefficientHistory, tblRGRTrainingHistory,
tblRGRPredictions and tblRGRFrontendCalc* tables.

Implemented with SQLAlchemy 2.0 ORM models (``db/models.py``) so the same
code runs on SQLite, PostgreSQL and SQL Server.  The public method
signatures are unchanged from the pyodbc implementation; ``connection`` is
now a SQLAlchemy ``Session`` (see ``DatabaseLinker/main.py``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection  # noqa: E402  (session factory)
from db.models import (  # noqa: E402
    Base,
    RGRCoefficient,
    RGRCoefficientHistory,
    RGRFrontendCalcData,
    RGRFrontendCalcPlotData,
    RGRPrediction,
    RGRTrainingHistory,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fmt_dt(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _opt_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def _opt_int(value: Any) -> Optional[int]:
    return int(value) if value is not None else None


def _opt_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(int(value))


class CoefficientStore:
    """
    Save / load RGR model coefficients.

    Parameters
    ----------
    connection : sqlalchemy.orm.Session, optional
        An open session.  A new one is created via get_db_connection()
        if None is passed (the caller owns its lifetime either way).
    """

    def __init__(self, connection: Optional[Session] = None) -> None:
        self.session: Session = connection or get_db_connection()
        # Backward-compatible alias used by older call sites.
        self.connection = self.session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_filter(spec_code: Optional[int], forest_unit: Optional[str], model_version: str):
        """NULL-safe (SpecCode, ForestUnit, ModelVersion) selector."""
        conditions = [RGRCoefficient.model_version == model_version]
        if spec_code is None:
            conditions.append(RGRCoefficient.spec_code.is_(None))
        else:
            conditions.append(RGRCoefficient.spec_code == int(spec_code))
        if forest_unit is None:
            conditions.append(RGRCoefficient.forest_unit.is_(None))
        else:
            conditions.append(RGRCoefficient.forest_unit == str(forest_unit))
        return and_(*conditions)

    def _latest(self, spec_code: Optional[int], forest_unit: Optional[str], model_version: str) -> Optional[RGRCoefficient]:
        stmt = (
            select(RGRCoefficient)
            .where(self._model_filter(spec_code, forest_unit, model_version))
            .order_by(RGRCoefficient.trained_date.desc(), RGRCoefficient.coefficient_id.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalars().first()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_coefficients(
        self,
        spec_code: Optional[int],
        forest_unit: Optional[str],
        species_name: str,
        coefficients: Dict[str, float],
        metrics: Dict,
        model_version: str = "1.0",
        max_data_year: Optional[int] = None,
        notes: Optional[str] = None,
        confidence_level: str = "UNKNOWN",
        is_production_ready: bool = False,
        quality_flags: Optional[str] = None,
        warning_message: Optional[str] = None,
        wrong_sign_count: int = 0,
        n_plots: Optional[int] = None,
    ) -> int:
        """
        Save or update a row in tblRGRCoefficients with confidence metadata.

        Retraining updates the existing row in place (so FK-linked history rows
        are preserved) and always appends an immutable snapshot to
        tblRGRCoefficientHistory.

        Returns
        -------
        int  The CoefficientID of the saved row.
        """
        values = {
            "spec_code": _opt_int(spec_code),
            "forest_unit": forest_unit,
            "species_name": species_name,
            "intercept": float(coefficients["intercept"]),
            "coef_ba": float(coefficients["BA"]),
            "coef_ba2": float(coefficients.get("BA2", 0.0) or 0.0),
            "coef_agsfrac": float(coefficients["AGSfrac"]),
            "coef_tcr": float(coefficients["TCR"]),
            "coef_dq": float(coefficients["DQ"]),
            "coef_q": float(coefficients["q"]),
            "coef_cut": float(coefficients["Cut"]),
            "r_squared": _opt_float(metrics.get("r_squared")),
            "rmse": _opt_float(metrics.get("rmse")),
            "mae": _opt_float(metrics.get("mae")),
            "n_samples": _opt_int(metrics.get("n_samples")),
            "model_version": model_version,
            "max_data_year": _opt_int(max_data_year),
            "notes": notes,
            "confidence_level": confidence_level,
            "is_production_ready": bool(is_production_ready),
            "quality_flags": quality_flags,
            "warning_message": warning_message,
            "wrong_sign_count": int(wrong_sign_count),
            "n_plots": _opt_int(n_plots),
        }

        existing = self.session.execute(
            select(RGRCoefficient).where(self._model_filter(spec_code, forest_unit, model_version)).limit(1)
        ).scalars().first()

        if existing is not None:
            for key, value in values.items():
                setattr(existing, key, value)
            existing.trained_date = _utcnow()
            row = existing
        else:
            row = RGRCoefficient(**values, trained_date=_utcnow())
            self.session.add(row)

        self.session.flush()  # assigns CoefficientID
        coef_id = int(row.coefficient_id)

        self.session.add(
            RGRCoefficientHistory(
                coefficient_id=coef_id,
                trained_date=_utcnow(),
                **values,
            )
        )
        self.session.commit()

        logger.info("  ✓ Saved coefficients  CoefficientID=%d  MaxDataYear=%s", coef_id, max_data_year)
        return coef_id

    def save_training_history(
        self,
        coefficient_id: int,
        training_data: pd.DataFrame,
    ) -> None:
        """Insert a row into tblRGRTrainingHistory for audit purposes."""
        tree_total = float(pd.to_numeric(training_data.get("TreeCount"), errors="coerce").fillna(0).sum())
        shrink_total = float(pd.to_numeric(training_data.get("ShrinkageCount"), errors="coerce").fillna(0).sum())

        quality_flags = {
            "n_plots": int(training_data["PlotKey"].nunique()),
            "date_range": f"{int(training_data['StartYear'].min())}-{int(training_data['EndYear'].max())}",
            "avg_ba": float(training_data["BA"].mean()),
            "shrinkage_pct": (shrink_total / tree_total * 100.0) if tree_total > 0 else 0.0,
        }

        self.session.add(
            RGRTrainingHistory(
                coefficient_id=int(coefficient_id),
                training_date=_utcnow(),
                n_plots=quality_flags["n_plots"],
                n_growth_intervals=int(len(training_data)),
                avg_growth_interval_years=float(training_data["GrowthInterval_Years"].mean()),
                min_ba=float(training_data["BA"].min()),
                max_ba=float(training_data["BA"].max()),
                avg_ba=quality_flags["avg_ba"],
                data_quality_flags=json.dumps(quality_flags),
            )
        )
        self.session.commit()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_coefficients(
        self,
        spec_code: Optional[int],
        forest_unit: Optional[str],
        model_version: str = "1.0",
    ) -> Dict:
        """
        Return coefficient dict for (spec_code, forest_unit, version).

        Raises
        ------
        ValueError  if no matching row is found.
        """
        row = self._latest(spec_code, forest_unit, model_version)
        if row is None:
            raise ValueError(
                f"No coefficients found for SpecCode={spec_code}, "
                f"ForestUnit='{forest_unit}', Version='{model_version}'. "
                "Run training first."
            )

        return {
            "intercept": float(row.intercept),
            "BA": float(row.coef_ba),
            "BA2": float(row.coef_ba2) if row.coef_ba2 is not None else 0.0,
            "AGSfrac": float(row.coef_agsfrac),
            "TCR": float(row.coef_tcr),
            "DQ": float(row.coef_dq),
            "q": float(row.coef_q),
            "Cut": float(row.coef_cut),
            "r_squared": _opt_float(row.r_squared),
            "rmse": _opt_float(row.rmse),
            "mae": _opt_float(row.mae),
            "n_samples": _opt_int(row.n_samples),
            "species_name": row.species_name,
        }

    def get_model_info(
        self,
        spec_code: Optional[int],
        forest_unit: Optional[str],
        model_version: str = "1.0",
    ) -> Optional[Dict]:
        """
        Return lightweight model metadata, or None if no stored model.

        Used by the incremental training check to compare MaxDataYear
        against the current max available year in the database.
        """
        row = self._latest(spec_code, forest_unit, model_version)
        if row is None:
            return None
        return {
            "coefficient_id": int(row.coefficient_id),
            "max_data_year": int(row.max_data_year) if row.max_data_year is not None else 0,
            "trained_date": _fmt_dt(row.trained_date) or "",
            "n_samples": int(row.n_samples) if row.n_samples is not None else 0,
        }

    def get_all_coefficients(self) -> pd.DataFrame:
        """Return all stored coefficients as a DataFrame (DB column names as headers)."""
        rows = self.session.execute(
            select(RGRCoefficient).order_by(RGRCoefficient.forest_unit, RGRCoefficient.spec_code)
        ).scalars().all()
        columns = [column.name for column in RGRCoefficient.__table__.columns]
        return pd.DataFrame([row.to_dict() for row in rows], columns=columns)

    def ensure_coefficients_table_exists(self) -> None:
        """
        Create the coefficient tables when they are missing.

        Schema evolution is handled by Alembic (``alembic upgrade head``); this
        is a safety net for ad-hoc databases and is a no-op when tables exist.
        """
        Base.metadata.create_all(self.session.get_bind(), checkfirst=True)

    def get_latest_model_selector(self, model_version: str = "1.0") -> Optional[Dict[str, Any]]:
        """Return latest stored model selector from tblRGRCoefficients for a version."""
        row = self.session.execute(
            select(RGRCoefficient)
            .where(RGRCoefficient.model_version == model_version)
            .order_by(RGRCoefficient.trained_date.desc(), RGRCoefficient.coefficient_id.desc())
            .limit(1)
        ).scalars().first()
        if row is None:
            return None
        return {
            "spec_code": _opt_int(row.spec_code),
            "forest_unit": str(row.forest_unit) if row.forest_unit is not None else None,
            "trained_date": _fmt_dt(row.trained_date),
        }

    # ------------------------------------------------------------------
    # Frontend calculation snapshots
    # ------------------------------------------------------------------

    def save_frontend_calc_data(
        self,
        *,
        spec_code: Optional[int],
        coefficient_id: Optional[int],
        species_abbrev: Optional[str],
        forest_unit: Optional[str],
        model_version: str,
        current_year: int,
        projection_years: int,
        extracted_rows: int,
        filtered_rows: int,
        latest_rows: int,
        used_db_model: bool,
        data_payload: Dict[str, Any],
    ) -> int:
        """Persist frontend growth-calculation input dataset snapshot for auditability."""
        row = RGRFrontendCalcData(
            requested_date=_utcnow(),
            spec_code=_opt_int(spec_code),
            coefficient_id=_opt_int(coefficient_id),
            species_abbrev=species_abbrev,
            forest_unit=forest_unit,
            model_version=model_version,
            current_year=int(current_year),
            projection_years=int(projection_years),
            extracted_rows=int(extracted_rows),
            filtered_rows=int(filtered_rows),
            latest_rows=int(latest_rows),
            used_db_model=bool(used_db_model),
            data_payload=json.dumps(data_payload, default=str),
        )
        self.session.add(row)
        self.session.commit()
        return int(row.frontend_calc_id)

    def save_frontend_calc_plot_rows(
        self,
        frontend_calc_id: int,
        plot_rows: List[Dict[str, Any]],
    ) -> None:
        """Persist one row per plot used by frontend BA growth calculation."""
        if not plot_rows:
            return

        self.session.add_all(
            [
                RGRFrontendCalcPlotData(
                    frontend_calc_id=int(frontend_calc_id),
                    plot_key=str(row.get("PlotKey")) if row.get("PlotKey") is not None else None,
                    start_year=_opt_int(row.get("StartYear")),
                    end_year=_opt_int(row.get("EndYear")),
                    growth_interval_years=_opt_float(row.get("GrowthInterval_Years")),
                    ba=_opt_float(row.get("BA")),
                    agsfrac=_opt_float(row.get("AGSfrac")),
                    tcr=_opt_int(row.get("TCR")),
                    dq=_opt_float(row.get("DQ")),
                    q=_opt_float(row.get("q")),
                    cut=_opt_bool(row.get("Cut")),
                    bagrowth_input=_opt_float(row.get("BAgrowth_Input")),
                    rgr_input=_opt_float(row.get("RGR_Input")),
                    rgr_predicted=_opt_float(row.get("RGR_Predicted")),
                    bagrowth_predicted=_opt_float(row.get("BAgrowth_Predicted")),
                )
                for row in plot_rows
            ]
        )
        self.session.commit()

    # ------------------------------------------------------------------
    # Save prediction (for validation auditing)
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        coefficient_id: int,
        plot_key: Optional[int],
        measure_year: Optional[int],
        ba: float,
        ags_frac: float,
        tcr: int,
        dq: float,
        q: float,
        cut: int,
        rgr_predicted: float,
        ba_growth_predicted: float,
        rgr_actual: Optional[float] = None,
        ba_growth_actual: Optional[float] = None,
    ) -> None:
        """Persist a prediction record for audit / validation."""
        residual = (rgr_actual - rgr_predicted) if rgr_actual is not None else None
        abs_error = abs(residual) if residual is not None else None

        self.session.add(
            RGRPrediction(
                coefficient_id=int(coefficient_id),
                plot_key=_opt_int(plot_key),
                measure_year=_opt_int(measure_year),
                ba_input=float(ba),
                agsfrac_input=float(ags_frac),
                tcr_input=int(tcr),
                dq_input=float(dq),
                q_input=float(q),
                cut_input=bool(int(cut)),
                rgr_predicted=float(rgr_predicted),
                bagrowth_predicted=float(ba_growth_predicted),
                rgr_actual=_opt_float(rgr_actual),
                bagrowth_actual=_opt_float(ba_growth_actual),
                prediction_date=_utcnow(),
                residual=_opt_float(residual),
                abs_error=_opt_float(abs_error),
            )
        )
        self.session.commit()
