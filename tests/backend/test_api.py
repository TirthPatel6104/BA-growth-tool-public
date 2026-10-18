"""
tests/backend/test_api.py
=========================
FastAPI endpoint tests against the seeded SQLite scratch database.
The ``client`` fixture (conftest.py) overrides ``api.deps.get_db``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

pytestmark = pytest.mark.unit


class TestInfrastructureEndpoints:

    def test_health_reports_database(self, client):
        body = client.get("/health").json()
        assert body["status"] == "UP"
        assert body["database_connection"] == "HEALTHY"
        assert body["database"]["dialect"] == "sqlite"

    def test_test_connection_is_dialect_aware(self, client):
        body = client.get("/test-connection").json()
        assert body["status"] == "success"
        assert body["dialect"] == "sqlite"
        assert body["sql_version"].startswith("SQLite")

    def test_tables_lists_migrated_schema(self, client):
        tables = client.get("/tables").json()["tables"]
        assert "tblRGRCoefficients" in tables
        assert "tblBAGrowthData" in tables
        assert "alembic_version" in tables


class TestModelCatalogue:

    def test_species_list_matches_seeded_models(self, client):
        body = client.get("/api/species-list").json()
        assert "HardMaple" in body["species_groups"]
        assert "FU_TH1" in body["forest_units"]
        assert body["mapping"]["HardMaple"] == "Mh"

    def test_stored_models_endpoint(self, client):
        models = client.get("/models/stored").json()["models"]
        assert len(models) >= 13
        assert {"CoefficientID", "SpecCode", "ForestUnit", "Coef_BA"} <= set(models[0])


class TestCoefficients:

    def test_latest_coefficients_without_filter(self, client):
        body = client.get("/api/model-coefficients").json()
        assert body["source"] == "tblRGRCoefficients"
        assert set(body["coefficients"]) == {"BA", "BA2", "AGSfrac", "TCR", "DQ", "q", "Cut"}
        assert body["model_info"]["coefficient_id"] is not None

    @pytest.mark.parametrize("group", ["HardMaple", "Cedar", "FU_TH1", "fu_hdsl2"])
    def test_coefficients_by_model_group(self, client, group):
        r = client.get("/api/model-coefficients", params={"species_group": group})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_version"] == "1.0"
        assert body["coefficients"]["BA"] < 0  # seeded with the expected sign

    def test_unknown_group_is_400(self, client):
        r = client.get("/api/model-coefficients", params={"species_group": "Unobtainium"})
        assert r.status_code == 400

    def test_unknown_version_is_404(self, client):
        r = client.get("/api/model-coefficients", params={"species_group": "HardMaple", "model_version": "9.9"})
        assert r.status_code == 404


class TestGrowthCalculations:

    def test_summary_stats(self, client):
        body = client.get("/api/bagrowth/summary-stats", params={"species_group": "HardMaple"}).json()
        assert body["species_group"] == "HardMaple"
        assert body["plots_analyzed"] > 0
        assert body["measurements"] >= 30
        assert body["variables"]["BA"]["mean"] > 0

    def test_formula_calculation_uses_stored_model(self, client):
        body = client.get("/api/bagrowth/formula-calculation", params={"species_group": "Cedar"}).json()
        assert body["species_group"] == "Cedar"
        assert body["coefficients"]["intercept"] > 0
        assert body["scenarios"]["with_cut"]["cut_value"] == 1
        assert body["measurements_used"] > 0

    @pytest.mark.parametrize("group", ["HardMaple", "FU_HDSL2"])
    def test_calculate_projection(self, client, group):
        r = client.get(
            "/api/bagrowth/calculate",
            params={"species_group": group, "current_year": 2025, "projection_years": 5, "refresh": True},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_plots"] > 0
        assert len(body["yearly_projection"]) == 6
        assert body["yearly_projection"][0]["year"] == 2025
        assert body["yearly_projection"][-1]["BA_predicted"] >= body["yearly_projection"][0]["BA_predicted"]
        assert body["pipeline_metrics"]["cache_reason"] == "Using stored coefficients"

    def test_calculate_persists_frontend_snapshot(self, client, db_session):
        from db.models import RGRFrontendCalcData, RGRFrontendCalcPlotData

        before = db_session.execute(select(func.count()).select_from(RGRFrontendCalcData)).scalar_one()
        r = client.get(
            "/api/bagrowth/calculate",
            params={"species_group": "RedMaple", "current_year": 2025, "projection_years": 2, "refresh": True},
        )
        assert r.status_code == 200, r.text
        after = db_session.execute(select(func.count()).select_from(RGRFrontendCalcData)).scalar_one()
        assert after == before + 1

        latest = db_session.execute(
            select(RGRFrontendCalcData).order_by(RGRFrontendCalcData.frontend_calc_id.desc())
        ).scalars().first()
        assert latest.species_abbrev == "Mr"
        assert latest.used_db_model is True
        plot_rows = db_session.execute(
            select(func.count()).select_from(RGRFrontendCalcPlotData).where(
                RGRFrontendCalcPlotData.frontend_calc_id == latest.frontend_calc_id
            )
        ).scalar_one()
        assert plot_rows == r.json()["n_plots"]

    def test_batch_predict_rows_without_model_are_skipped_not_failed(self, client):
        r = client.post(
            "/models/predict/batch",
            json={"rows": [{"SpecCode": 30, "ForestUnit": "TH1", "BA": 20, "AGSfrac": 0.6,
                            "TCR": 4, "DQ": 25, "q": 1.3, "Cut": 0}]},
        )
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 1
