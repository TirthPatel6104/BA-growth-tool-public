"""
prediction/predictor.py
========================
Uses stored RGR coefficients to predict:
  - RGR (relative growth rate)
  - BA growth (m²/ha/yr)
  - Multi-year BA projections

Integrates with:
  - CoefficientStore (models/coefficient_store.py) – loads fitted coefficients
  - FindTCR.py                                     – resolves TCR from coordinates
  - get_db_connection()                             – DatabaseLinker/main.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from DatabaseLinker.main import get_db_connection
from models.coefficient_store import CoefficientStore
from data.extractors import _get_tcr  # reuse the spatial TCR helper

logger = logging.getLogger(__name__)


class RGRPredictor:
    """
    Predict RGR and basal area growth using pre-trained coefficients.

    Parameters
    ----------
    connection : pyodbc.Connection, optional
        Existing connection; creates a new one if None.
    model_version : str
        Which coefficient version to load (default "1.0").
    """

    def __init__(self, connection=None, model_version: str = "1.0") -> None:
        conn = connection or get_db_connection()
        self._store         = CoefficientStore(conn)
        self._cache: Dict   = {}          # {(spec_code, forest_unit): coef_dict}
        self._coef_ids: Dict = {}         # {(spec_code, forest_unit): CoefficientID}
        self.model_version  = model_version

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------

    def predict_rgr(
        self,
        spec_code: int,
        forest_unit: str,
        BA: float,
        AGSfrac: float,
        TCR: int,
        DQ: float,
        q: float,
        Cut: int,
    ) -> float:
        """
        Predict RGR for one observation.

        Parameters
        ----------
        spec_code   : int    Species code (e.g. 30 = Hard Maple, 999 = Cedar)
        forest_unit : str    Forest unit code (e.g. 'TH1', 'HDSL2')
        BA          : float  Basal area (m²/ha)
        AGSfrac     : float  Acceptable growing stock fraction (0–1)
        TCR         : int    Timber capability rating (1–7); use 5 if unknown
        DQ          : float  Quadratic mean diameter (cm)
        q           : float  q-factor (diameter class distribution)
        Cut         : int    Cutting indicator (0=no cut, 1=cut)

        Returns
        -------
        float  Predicted RGR value
        """
        coefs = self._load_model(spec_code, forest_unit)
        return self._apply_equation(coefs, BA, AGSfrac, TCR, DQ, q, Cut)

    def predict_ba_growth(
        self,
        spec_code: int,
        forest_unit: str,
        BA: float,
        AGSfrac: float,
        TCR: int,
        DQ: float,
        q: float,
        Cut: int,
    ) -> float:
        """
        Predict annual BA growth (m²/ha/yr) = RGR × BA.
        """
        rgr = self.predict_rgr(spec_code, forest_unit, BA, AGSfrac, TCR, DQ, q, Cut)
        return rgr * BA

    def predict_with_coordinates(
        self,
        spec_code: int,
        forest_unit: str,
        BA: float,
        AGSfrac: float,
        DQ: float,
        q: float,
        Cut: int,
        easting: Optional[float] = None,
        northing: Optional[float] = None,
    ) -> Tuple[float, float, int]:
        """
        Predict RGR and BA growth, resolving TCR from spatial coordinates
        via FindTCR.py when provided.

        Returns
        -------
        (rgr, ba_growth, tcr_used)
        """
        tcr = _get_tcr(easting, northing)
        rgr = self.predict_rgr(spec_code, forest_unit, BA, AGSfrac, tcr, DQ, q, Cut)
        return rgr, rgr * BA, tcr

    # ------------------------------------------------------------------
    # Multi-year projection
    # ------------------------------------------------------------------

    def predict_future_ba(
        self,
        spec_code: int,
        forest_unit: str,
        BA_start: float,
        AGSfrac: float,
        TCR: int,
        DQ: float,
        q: float,
        Cut: int,
        years: int = 10,
    ) -> pd.DataFrame:
        """
        Project basal area forward year by year.

        Returns
        -------
        pd.DataFrame  with columns: Year, BA, BAgrowth, RGR
        """
        rows = [{"Year": 0, "BA": BA_start, "BAgrowth": 0.0, "RGR": 0.0}]
        ba_current = BA_start

        for yr in range(1, years + 1):
            rgr       = self.predict_rgr(spec_code, forest_unit, ba_current,
                                          AGSfrac, TCR, DQ, q, Cut)
            ba_growth = rgr * ba_current
            ba_current += ba_growth
            rows.append({"Year": yr, "BA": ba_current, "BAgrowth": ba_growth, "RGR": rgr})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict RGR and BAgrowth for every row in df.

        df must have columns:
            SpecCode, ForestUnit, BA, AGSfrac, TCR, DQ, q, Cut

        Returns a copy of df with added columns:
            RGR_Predicted, BAgrowth_Predicted
        """
        required_cols = ["SpecCode", "ForestUnit", "BA", "AGSfrac", "TCR", "DQ", "q", "Cut"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        df = df.copy()
        df["RGR_Predicted"]      = 0.0
        df["BAgrowth_Predicted"] = 0.0

        for idx, row in df.iterrows():
            try:
                rgr = self.predict_rgr(
                    int(row["SpecCode"]),
                    str(row["ForestUnit"]),
                    float(row["BA"]),
                    float(row["AGSfrac"]),
                    int(row["TCR"]) if pd.notna(row["TCR"]) else 5,
                    float(row["DQ"]),
                    float(row["q"]),
                    int(row["Cut"]),
                )
                df.at[idx, "RGR_Predicted"]      = rgr
                df.at[idx, "BAgrowth_Predicted"] = rgr * float(row["BA"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Row %s skipped: %s", idx, exc)

        return df

    # ------------------------------------------------------------------
    # Listing available models
    # ------------------------------------------------------------------

    def list_available_models(self) -> pd.DataFrame:
        """Return a summary of all stored models from the database."""
        return self._store.get_all_coefficients()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, spec_code: int, forest_unit: str) -> Dict:
        """Return cached coefficients or load from DB."""
        key = (spec_code, forest_unit)
        if key not in self._cache:
            self._cache[key] = self._store.load_coefficients(
                spec_code, forest_unit, self.model_version
            )
        return self._cache[key]

    @staticmethod
    def _apply_equation(
        coefs: Dict,
        BA: float,
        AGSfrac: float,
        TCR: int,
        DQ: float,
        q: float,
        Cut: int,
    ) -> float:
        """RGR = β0 + β1·BA + β2·BA² + β3·AGSfrac + β4·TCR + β5·DQ + β6·q + β7·Cut"""
        return (
            coefs["intercept"]
            + coefs["BA"]      * BA
            + coefs.get("BA2", 0.0) * (BA ** 2)
            + coefs["AGSfrac"] * AGSfrac
            + coefs["TCR"]     * TCR
            + coefs["DQ"]      * DQ
            + coefs["q"]       * q
            + coefs["Cut"]     * Cut
        )
