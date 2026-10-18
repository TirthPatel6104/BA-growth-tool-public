"""
models/rgr_model.py
===================
Linear regression model for the Relative Growth Rate (RGR) equation:

    RGR = β0 + β1·BA + β2·BA² + β3·AGSfrac + β4·TCR + β5·DQ + β6·q + β7·Cut

where
    BA      – basal area at start of interval (m²/ha)
    AGSfrac – acceptable growing stock fraction (0–1)
    TCR     – timber capability rating (1–7)
    DQ      – quadratic mean diameter (cm)
    q       – q-factor (diameter class distribution ratio)
    Cut     – cutting indicator (0 = no cut, 1 = cut)

Basal area growth is then:
    BAgrowth = RGR × BA  (m²/ha/yr)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = ["BA", "BA2", "AGSfrac", "TCR", "DQ", "q", "Cut"]
TARGET_NAME: str = "RGR"
MIN_SAMPLES: int = 30


class RGRModel:
    """
    Fits and applies the species-specific RGR linear regression model.

    Attributes
    ----------
    coefficients : dict or None
        Fitted coefficients after calling train().
    metrics : dict or None
        Model quality metrics (R², RMSE, MAE, n_samples) after calling train().
    """

    def __init__(self, min_samples: int = MIN_SAMPLES) -> None:
        self._model = LinearRegression()
        self.min_samples = int(min_samples)
        self.coefficients: Optional[Dict] = None
        self.metrics: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Fit the RGR model on the supplied training data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: BA, AGSfrac, TCR, DQ, q, Cut, RGR.
            Missing TCR is filled with 5.
            AGSfrac is not imputed; rows missing AGSfrac are dropped.
            Rows with missing DQ or q are dropped.

        Returns
        -------
        dict  containing all coefficients and quality metrics.

        Raises
        ------
        ValueError  if fewer than MIN_SAMPLES rows remain after cleaning.
        """
        X, y = self._prepare_features(df)

        if len(X) < self.min_samples:
            raise ValueError(
                f"Insufficient training data: {len(X)} rows (need >= {self.min_samples})"
            )

        self._model.fit(X, y)

        self.coefficients = {
            "intercept": float(self._model.intercept_),
            "BA":        float(self._model.coef_[0]),
            "BA2":       float(self._model.coef_[1]),
            "AGSfrac":   float(self._model.coef_[2]),
            "TCR":       float(self._model.coef_[3]),
            "DQ":        float(self._model.coef_[4]),
            "q":         float(self._model.coef_[5]),
            "Cut":       float(self._model.coef_[6]),
        }

        y_pred = self._model.predict(X)
        self.metrics = {
            "r_squared": float(r2_score(y, y_pred)),
            "rmse":      float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae":       float(mean_absolute_error(y, y_pred)),
            "n_samples": len(X),
            "n_plots":   int(df["PlotKey"].nunique()) if "PlotKey" in df.columns else None,
            "avg_interval": float(df["GrowthInterval_Years"].mean())
                            if "GrowthInterval_Years" in df.columns else None,
        }

        logger.info(
            "  ✓ Trained  R²=%.4f  RMSE=%.4f  N=%d",
            self.metrics["r_squared"],
            self.metrics["rmse"],
            self.metrics["n_samples"],
        )
        logger.info("  %s", self.get_equation_string())

        return {**self.coefficients, **self.metrics}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        BA: float,
        AGSfrac: float,
        TCR: int,
        DQ: float,
        q: float,
        Cut: int,
    ) -> float:
        """
        Predict RGR for a single observation.

        Parameters
        ----------
        BA      : float  Basal area (m²/ha)
        AGSfrac : float  AGS fraction (0–1)
        TCR     : int    Timber capability rating (1–7); default 5 if unknown
        DQ      : float  Quadratic mean diameter (cm)
        q       : float  q-factor
        Cut     : int    Cutting indicator (0 or 1)

        Returns
        -------
        float  predicted RGR
        """
        self._check_trained()
        X = pd.DataFrame(
            [[BA, BA * BA, AGSfrac, TCR, DQ, q, Cut]], columns=FEATURE_NAMES
        )
        return float(self._model.predict(X)[0])

    def predict_ba_growth(
        self,
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
        return self.predict(BA, AGSfrac, TCR, DQ, q, Cut) * BA

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RGR_Predicted and BAgrowth_Predicted columns to a DataFrame
        that already contains FEATURE_NAMES columns.
        """
        self._check_trained()
        X = self._fill_defaults(df.copy())[FEATURE_NAMES]
        df = df.copy()
        df["RGR_Predicted"]      = self._model.predict(X).tolist()
        df["BAgrowth_Predicted"] = df["RGR_Predicted"] * df["BA"]
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_equation_string(self) -> str:
        """Return a human-readable equation with fitted coefficients."""
        if self.coefficients is None:
            return "Model not trained yet."
        c = self.coefficients
        return (
            f"RGR = {c['intercept']:+.5f} "
            f"{c['BA']:+.5f}·BA "
            f"{c['BA2']:+.5f}·BA2 "
            f"{c['AGSfrac']:+.5f}·AGSfrac "
            f"{c['TCR']:+.5f}·TCR "
            f"{c['DQ']:+.5f}·DQ "
            f"{c['q']:+.5f}·q "
            f"{c['Cut']:+.5f}·Cut"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data and return (X, y)."""
        df_clean = self._fill_defaults(df.copy())

        # Drop rows where critical predictors are missing
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=["BA", "BA2", "AGSfrac", "TCR", "DQ", "q", "Cut", TARGET_NAME])
        dropped = before - len(df_clean)
        if dropped:
            logger.debug("  Dropped %d rows with missing BA/BA2/AGSfrac/TCR/DQ/q/Cut/RGR.", dropped)

        X = df_clean[FEATURE_NAMES]
        y = df_clean[TARGET_NAME]
        return X, y

    @staticmethod
    def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing-value defaults and type normalization."""
        df["BA"]      = pd.to_numeric(df["BA"], errors="coerce")
        df["BA2"]     = df["BA"] ** 2
        df["AGSfrac"] = pd.to_numeric(df["AGSfrac"], errors="coerce")
        df["TCR"]     = df["TCR"].fillna(5)    # RULE H1: default TCR = 5
        df["TCR"]     = pd.to_numeric(df["TCR"], errors="coerce")
        df["DQ"]      = pd.to_numeric(df["DQ"], errors="coerce")
        df["q"]       = pd.to_numeric(df["q"], errors="coerce")
        df["Cut"]     = df["Cut"].fillna(0)
        df["Cut"]     = pd.to_numeric(df["Cut"], errors="coerce")
        return df

    def _check_trained(self) -> None:
        if self.coefficients is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
