"""
tests/backend/test_db_layer.py
==============================
SQLAlchemy / Alembic data-layer tests (SQLite scratch database, no server).
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestDatabaseConfig:

    def test_default_is_sqlite_demo_file(self, monkeypatch):
        from db import config

        for var in ("DATABASE_URL", "BAGT_SQL_CONNECTION_STRING", "BAGT_SQL_SERVER", "BAGT_SQLITE_PATH"):
            monkeypatch.delenv(var, raising=False)
        url = config.get_database_url()
        assert url.startswith("sqlite:///")
        assert url.endswith("/demo_basal_growth.db")
        assert config.get_dialect(url) == "sqlite"

    def test_database_url_wins_and_postgres_is_normalised(self, monkeypatch):
        from db import config

        monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@db.example.org:5432/bagt")
        monkeypatch.setenv("BAGT_SQL_SERVER", "ignored-server")
        url = config.get_database_url()
        assert url == "postgresql+psycopg2://user:pw@db.example.org:5432/bagt"
        assert config.get_dialect(url) == "postgresql"
        assert "pw" not in config.redact_url(url)

    def test_legacy_mssql_keywords_build_pyodbc_url(self, monkeypatch):
        from db import config

        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("BAGT_SQL_CONNECTION_STRING", raising=False)
        monkeypatch.setenv("BAGT_SQL_SERVER", "sqlhost")
        monkeypatch.setenv("BAGT_SQL_DATABASE", "growth")
        monkeypatch.setenv("BAGT_SQL_TRUSTED", "no")
        monkeypatch.setenv("BAGT_SQL_USER", "u")
        monkeypatch.setenv("BAGT_SQL_PASSWORD", "p")
        url = config.get_database_url()
        assert url.startswith("mssql+pyodbc:///?odbc_connect=")
        assert "SERVER%3Dsqlhost" in url
        assert "DATABASE%3Dgrowth" in url
        assert "UID%3Du" in url
        assert config.is_mssql(url)
        assert config.redact_url(url) == "mssql+pyodbc://<odbc_connect redacted>"


# ---------------------------------------------------------------------------
# Alembic migrations
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "tblRGRCoefficients",
    "tblRGRCoefficientHistory",
    "tblRGRTrainingHistory",
    "tblRGRPredictions",
    "tblRGRFrontendCalcData",
    "tblRGRFrontendCalcPlotData",
    "tblBAGrowthData",
}


class TestMigrations:

    def test_upgrade_head_creates_current_schema(self, migrated_db):
        from db.engine import get_engine

        tables = set(inspect(get_engine()).get_table_names())
        assert EXPECTED_TABLES <= tables
        assert "alembic_version" in tables

    def test_database_is_stamped_at_head(self, migrated_db):
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        from db.bootstrap import alembic_config
        from db.engine import get_engine

        head = ScriptDirectory.from_config(alembic_config(migrated_db)).get_current_head()
        with get_engine().connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
        assert current == head

    def test_models_match_migrations(self, migrated_db):
        """Autogenerate must find nothing to do – ORM models and migrations agree."""
        from alembic.autogenerate import compare_metadata
        from alembic.runtime.migration import MigrationContext

        from db.engine import get_engine
        from db.models import Base

        with get_engine().connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            diff = compare_metadata(ctx, Base.metadata)
        assert diff == [], f"ORM models drifted from migrations: {diff}"

    def test_legacy_coefficient_columns_are_present(self, migrated_db):
        """Column names must match sql/create_rgr_tables.sql + add_confidence_columns.sql."""
        from db.engine import get_engine

        columns = {c["name"] for c in inspect(get_engine()).get_columns("tblRGRCoefficients")}
        for legacy in (
            "CoefficientID", "SpecCode", "ForestUnit", "SpeciesName", "Intercept", "Coef_BA",
            "Coef_BA2", "Coef_AGSfrac", "Coef_TCR", "Coef_DQ", "Coef_q", "Coef_Cut", "R_Squared",
            "RMSE", "MAE", "N_Samples", "TrainedDate", "ModelVersion", "MaxDataYear", "Notes",
            "ConfidenceLevel", "IsProductionReady", "QualityFlags", "WarningMessage",
            "WrongSignCount", "N_Plots",
        ):
            assert legacy in columns, f"missing legacy column {legacy}"


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

class TestSeed:

    def test_every_configured_model_has_coefficients_and_growth_data(self, db_session):
        from data.extractors import FOREST_UNIT_CONFIG, SPECIES_CONFIG
        from db.models import BAGrowthInterval, RGRCoefficient

        coefficients = db_session.execute(select(RGRCoefficient)).scalars().all()
        assert len(coefficients) == len(SPECIES_CONFIG) + len(FOREST_UNIT_CONFIG)

        by_species = {c.spec_code for c in coefficients if c.spec_code is not None}
        assert by_species == {int(cfg["code"]) for cfg in SPECIES_CONFIG.values()}
        by_fu = {c.forest_unit for c in coefficients if c.forest_unit is not None}
        assert by_fu == set(FOREST_UNIT_CONFIG)

        for cfg in SPECIES_CONFIG.values():
            n = db_session.execute(
                select(func.count()).select_from(BAGrowthInterval).where(BAGrowthInterval.spec_code == int(cfg["code"]))
            ).scalar_one()
            assert n >= 30, f"species {cfg['name']} has only {n} growth rows"
        for fu in FOREST_UNIT_CONFIG:
            n = db_session.execute(
                select(func.count()).select_from(BAGrowthInterval).where(
                    BAGrowthInterval.spec_code.is_(None), BAGrowthInterval.forest_unit == fu
                )
            ).scalar_one()
            assert n >= 30, f"forest unit {fu} has only {n} growth rows"

    def test_seed_is_deterministic(self, db_session):
        from db.engine import session_scope
        from db.models import RGRCoefficient
        from db.seed import seed_demo_database

        before = db_session.execute(
            select(RGRCoefficient.spec_code, RGRCoefficient.intercept).order_by(RGRCoefficient.coefficient_id)
        ).all()
        with session_scope() as session:
            seed_demo_database(session, reset=True)
        after = db_session.execute(
            select(RGRCoefficient.spec_code, RGRCoefficient.intercept).order_by(RGRCoefficient.coefficient_id)
        ).all()
        assert [tuple(r) for r in before] == [tuple(r) for r in after]

    def test_seed_years_are_in_the_past(self, db_session):
        from db.models import BAGrowthInterval

        max_year = db_session.execute(select(func.max(BAGrowthInterval.end_year))).scalar_one()
        assert max_year <= 2024


# ---------------------------------------------------------------------------
# CoefficientStore (ORM)
# ---------------------------------------------------------------------------

COEFS = {"intercept": 0.05, "BA": -0.0004, "BA2": 0.0, "AGSfrac": 0.012, "TCR": 0.002,
         "DQ": -0.0002, "q": 0.003, "Cut": -0.004}
METRICS = {"r_squared": 0.71, "rmse": 0.0031, "mae": 0.0024, "n_samples": 120}


class TestCoefficientStore:

    def test_save_then_load_roundtrip(self, db_session):
        from models.coefficient_store import CoefficientStore

        store = CoefficientStore(db_session)
        coef_id = store.save_coefficients(
            spec_code=30, forest_unit=None, species_name="Hard Maple",
            coefficients=COEFS, metrics=METRICS, model_version="test-rt",
            max_data_year=2021, confidence_level="HIGH", is_production_ready=True, n_plots=40,
        )
        loaded = store.load_coefficients(30, None, model_version="test-rt")
        assert loaded["intercept"] == pytest.approx(0.05)
        assert loaded["Cut"] == pytest.approx(-0.004)
        assert loaded["n_samples"] == 120
        assert loaded["species_name"] == "Hard Maple"

        info = store.get_model_info(30, None, model_version="test-rt")
        assert info["coefficient_id"] == coef_id
        assert info["max_data_year"] == 2021
        assert info["trained_date"]

    def test_retrain_updates_in_place_and_snapshots_history(self, db_session):
        from db.models import RGRCoefficientHistory
        from models.coefficient_store import CoefficientStore

        store = CoefficientStore(db_session)
        first = store.save_coefficients(None, "TH1", "TH1 (NER)", COEFS, METRICS, model_version="test-upd")
        second = store.save_coefficients(
            None, "TH1", "TH1 (NER)", {**COEFS, "intercept": 0.09}, METRICS, model_version="test-upd"
        )
        assert first == second, "retraining must update the existing row (FK-linked history)"
        assert store.load_coefficients(None, "TH1", model_version="test-upd")["intercept"] == pytest.approx(0.09)

        snapshots = db_session.execute(
            select(func.count()).select_from(RGRCoefficientHistory).where(RGRCoefficientHistory.coefficient_id == first)
        ).scalar_one()
        assert snapshots == 2

    def test_missing_model_raises_value_error(self, db_session):
        from models.coefficient_store import CoefficientStore

        store = CoefficientStore(db_session)
        with pytest.raises(ValueError, match="No coefficients found"):
            store.load_coefficients(12345, None, model_version="1.0")
        assert store.get_model_info(12345, None, model_version="1.0") is None

    def test_get_all_coefficients_uses_db_column_names(self, db_session):
        from models.coefficient_store import CoefficientStore

        df = CoefficientStore(db_session).get_all_coefficients()
        assert len(df) >= 13
        for column in ("CoefficientID", "SpecCode", "ForestUnit", "Coef_BA", "TrainedDate"):
            assert column in df.columns

    def test_training_history_and_audit_rows(self, db_session):
        from data.extractors import BAGrowthDataExtractor
        from db.models import RGRFrontendCalcPlotData, RGRPrediction, RGRTrainingHistory
        from models.coefficient_store import CoefficientStore

        store = CoefficientStore(db_session)
        data = BAGrowthDataExtractor(db_session, populate_tcr=False).extract_training_data("By")
        coef_id = store.save_coefficients(37, None, "Yellow Birch", COEFS, METRICS, model_version="test-hist")
        store.save_training_history(coef_id, data)
        calc_id = store.save_frontend_calc_data(
            spec_code=37, coefficient_id=coef_id, species_abbrev="By", forest_unit=None, model_version="test-hist",
            current_year=2025, projection_years=5, extracted_rows=len(data), filtered_rows=len(data),
            latest_rows=10, used_db_model=True, data_payload={"rows": 1},
        )
        store.save_frontend_calc_plot_rows(calc_id, [{"PlotKey": 1, "BA": 20.0, "Cut": 1, "TCR": 4}])
        store.save_prediction(coef_id, 1, 2020, 20.0, 0.5, 4, 25.0, 1.2, 0, 0.03, 0.6, rgr_actual=0.035)

        assert db_session.execute(
            select(func.count()).select_from(RGRTrainingHistory).where(RGRTrainingHistory.coefficient_id == coef_id)
        ).scalar_one() == 1
        plot_row = db_session.execute(
            select(RGRFrontendCalcPlotData).where(RGRFrontendCalcPlotData.frontend_calc_id == calc_id)
        ).scalars().one()
        assert plot_row.cut is True and plot_row.plot_key == "1"
        prediction = db_session.execute(
            select(RGRPrediction).where(RGRPrediction.coefficient_id == coef_id)
        ).scalars().one()
        assert prediction.residual == pytest.approx(0.005)

    def test_partial_unique_indexes_are_null_safe(self, db_session):
        from db.models import RGRCoefficient

        base = dict(intercept=0.1, coef_ba=-0.1, coef_agsfrac=0.1, coef_tcr=0.1, coef_dq=-0.1, coef_q=0.1, coef_cut=-0.1)
        db_session.add(RGRCoefficient(spec_code=777, forest_unit=None, model_version="uq", **base))
        db_session.add(RGRCoefficient(spec_code=777, forest_unit="TH1", model_version="uq", **base))  # legacy row OK
        db_session.flush()
        db_session.add(RGRCoefficient(spec_code=777, forest_unit=None, model_version="uq", **base))
        with pytest.raises(IntegrityError):
            db_session.flush()
        db_session.rollback()


# ---------------------------------------------------------------------------
# Extractor on the portable growth table
# ---------------------------------------------------------------------------

class TestExtractor:

    def test_species_extraction_returns_stored_procedure_shape(self, db_session):
        from data.extractors import BAGrowthDataExtractor
        from db.models import GROWTH_DATA_COLUMNS

        extractor = BAGrowthDataExtractor(db_session, populate_tcr=False)
        assert extractor.uses_stored_procedures is False
        df = extractor.extract_training_data("Mh")
        assert list(df.columns) == GROWTH_DATA_COLUMNS
        assert len(df) >= 30
        assert (df["SpecCode"] == 30).all()

    def test_forest_unit_extraction_is_stand_level(self, db_session):
        from data.extractors import BAGrowthDataExtractor

        df = BAGrowthDataExtractor(db_session, populate_tcr=False).extract_training_data(None, "HDSL2")
        assert len(df) >= 30
        assert df["SpecCode"].isna().all()
        assert (df["ForestUnit"] == "HDSL2").all()

    def test_since_year_and_max_year(self, db_session):
        from data.extractors import BAGrowthDataExtractor

        extractor = BAGrowthDataExtractor(db_session, populate_tcr=False)
        full = extractor.extract_training_data("Cedar")
        max_year = extractor.get_max_available_year("Cedar")
        assert max_year == int(full["EndYear"].max())
        partial = extractor.extract_training_data("Cedar", since_year=max_year - 3)
        assert 0 < len(partial) < len(full)
        assert (partial["EndYear"] > max_year - 3).all()

    def test_unknown_species_raises(self, db_session):
        from data.extractors import BAGrowthDataExtractor

        with pytest.raises(ValueError, match="Unknown species"):
            BAGrowthDataExtractor(db_session, populate_tcr=False).extract_training_data("Nope")

    def test_extract_all_combinations_covers_every_model(self, db_session):
        from data.extractors import VALID_COMBINATIONS, BAGrowthDataExtractor

        results = BAGrowthDataExtractor(db_session, populate_tcr=False).extract_all_combinations()
        assert len(results) == len(VALID_COMBINATIONS)
