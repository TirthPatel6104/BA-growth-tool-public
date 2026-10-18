"""
tests/backend/test_pipeline.py
================================
Unit tests for the RGR model training pipeline.

Run from the backend directory:
    pytest tests/backend/test_pipeline.py -v

Requires a working database connection (uses existing config) for
integration tests; pure-model tests use synthetic data and run offline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure backend is on path
_BACKEND_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend"
)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from models.rgr_model import RGRModel, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data that matches the schema returned by
    sp_GetBAGrowthData.
    """
    rng = np.random.default_rng(seed)
    BA      = rng.uniform(5, 50,   n)
    AGSfrac = rng.uniform(0.2, 0.9, n)
    TCR     = rng.integers(1, 8,   n)
    DQ      = rng.uniform(10, 40,  n)
    q       = rng.uniform(0.5, 2.5, n)
    Cut     = rng.integers(0, 2,   n)

    # True coefficients
    RGR = (
        0.07
        + -0.002  * BA
        + 0.015   * AGSfrac
        + 0.002   * TCR
        + -0.001  * DQ
        + 0.004   * q
        + -0.01   * Cut
        + rng.normal(0, 0.005, n)
    )
    RGR = np.clip(RGR, -0.5, 0.5)

    return pd.DataFrame({
        "PlotKey":              rng.integers(1000, 9999, n),
        "StartYear":            rng.integers(1980, 2010, n),
        "EndYear":              rng.integers(2010, 2022, n),
        "GrowthInterval_Years": rng.integers(3, 15,   n),
        "BA":       BA,
        "AGSfrac":  AGSfrac,
        "TCR":      TCR.astype(float),
        "DQ":       DQ,
        "q":        q,
        "Cut":      Cut.astype(float),
        "RGR":      RGR,
        "BAgrowth": RGR * BA,
        "TreeCount":       rng.integers(20, 200, n),
        "ShrinkageCount":  rng.integers(0,  5,   n),
    })


# ---------------------------------------------------------------------------
# RGRModel tests (no database required)
# ---------------------------------------------------------------------------

class TestRGRModel:

    def test_train_returns_coefficients(self):
        data  = _make_synthetic_data(100)
        model = RGRModel()
        result = model.train(data)

        assert "intercept" in result
        for feat in FEATURE_NAMES:
            assert feat in result, f"Missing coefficient for {feat}"
        assert result["n_samples"] > 0

    def test_r_squared_reasonable(self):
        data  = _make_synthetic_data(200)
        model = RGRModel()
        result = model.train(data)
        # Synthetic data has low noise → expect decent R²
        assert result["r_squared"] > 0.10, "R² too low for synthetic data"

    def test_predict_single(self):
        model = RGRModel()
        model.train(_make_synthetic_data())
        rgr = model.predict(BA=25.0, AGSfrac=0.6, TCR=4, DQ=28.0, q=1.2, Cut=0)
        assert isinstance(rgr, float)
        assert -0.5 <= rgr <= 0.5

    def test_predict_ba_growth(self):
        model = RGRModel()
        model.train(_make_synthetic_data())
        ba    = 20.0
        growth = model.predict_ba_growth(BA=ba, AGSfrac=0.5, TCR=5, DQ=25.0, q=1.0, Cut=0)
        rgr    = model.predict(BA=ba, AGSfrac=0.5, TCR=5, DQ=25.0, q=1.0, Cut=0)
        assert abs(growth - rgr * ba) < 1e-9

    def test_insufficient_samples_raises(self):
        data = _make_synthetic_data(10)  # fewer than MIN_SAMPLES=30
        model = RGRModel()
        with pytest.raises(ValueError, match="Insufficient"):
            model.train(data)

    def test_missing_tcr_filled_with_default(self):
        data = _make_synthetic_data(80)
        data.loc[data.index[:20], "TCR"] = None  # introduce NULLs
        model  = RGRModel()
        result = model.train(data)
        # Should not raise and n_samples should not drop to zero
        assert result["n_samples"] >= 60

    def test_missing_ags_filled_with_default(self):
        data = _make_synthetic_data(80)
        data.loc[data.index[:10], "AGSfrac"] = None
        model  = RGRModel()
        result = model.train(data)
        assert result["n_samples"] >= 70

    def test_missing_q_drops_rows(self):
        data = _make_synthetic_data(80)
        drop_n = 15
        data.loc[data.index[:drop_n], "q"] = None
        model  = RGRModel()
        result = model.train(data)
        assert result["n_samples"] <= 80 - drop_n

    def test_predict_batch(self):
        model = RGRModel()
        model.train(_make_synthetic_data())
        batch = _make_synthetic_data(10)
        out   = model.predict_batch(batch)
        assert "RGR_Predicted" in out.columns
        assert "BAgrowth_Predicted" in out.columns
        assert len(out) == 10

    def test_untrained_predict_raises(self):
        model = RGRModel()
        with pytest.raises(RuntimeError):
            model.predict(BA=20, AGSfrac=0.5, TCR=5, DQ=25, q=1, Cut=0)

    def test_equation_string(self):
        model = RGRModel()
        model.train(_make_synthetic_data())
        eq = model.get_equation_string()
        assert "RGR =" in eq
        for feat in FEATURE_NAMES:
            assert feat in eq


# ---------------------------------------------------------------------------
# Species / forest-unit configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:

    @pytest.mark.integration
    def test_cedar_has_no_forest_unit_restriction(self):
        from data.extractors import VALID_COMBINATIONS
        cedar_combos = [(s, fu) for s, fu in VALID_COMBINATIONS if s == "Cedar"]
        assert len(cedar_combos) > 0, "Cedar should appear in VALID_COMBINATIONS"
        assert all(fu is None for _, fu in cedar_combos), \
            "Cedar is a species model and should have fu=None (runs across all forest units)"

    @pytest.mark.integration
    def test_all_single_species_valid(self):
        from data.extractors import SPECIES_CONFIG, VALID_COMBINATIONS
        non_cedar = [k for k in SPECIES_CONFIG if k != "Cedar"]
        for sp in non_cedar:
            matched = [fu for s, fu in VALID_COMBINATIONS if s == sp]
            assert len(matched) > 0, f"No forest units found for species {sp}"

    @pytest.mark.integration
    def test_spec_codes_unique(self):
        from data.extractors import SPECIES_CONFIG
        codes = [v["code"] for k, v in SPECIES_CONFIG.items() if k != "Cedar"]
        assert len(codes) == len(set(codes)), "Duplicate SpecCodes found"
