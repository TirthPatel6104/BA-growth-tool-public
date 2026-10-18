"""
db/models.py
============
SQLAlchemy 2.0 ORM models for the RGR coefficient store.

The physical column names deliberately match the historical MSSQL schema
(``sql/create_rgr_tables.sql`` + ``sql/add_confidence_columns.sql``) so that an
existing SQL Server installation keeps working after ``alembic stamp head``.
Python attribute names are snake_case; ``mapped_column("DbName", ...)`` maps
each one to its legacy column.

Tables
------
tblRGRCoefficients          – current fitted coefficients per species / forest unit
tblRGRCoefficientHistory    – immutable snapshot of every training run
tblRGRTrainingHistory       – training-data QA summary per run
tblRGRPredictions           – prediction audit log
tblRGRFrontendCalcData      – frontend growth-calculation request snapshots
tblRGRFrontendCalcPlotData  – per-plot rows used by a frontend calculation
tblBAGrowthData             – portable growth-interval training data (SQLite /
                              Postgres replacement for the MSSQL stored procedures)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Unicode,
    UnicodeText,
    func,
    text,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names so Alembic batch mode (SQLite) can find them.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def _utcnow() -> datetime:
    """Naive UTC timestamp (DATETIME columns have no tz on any of the 3 dialects)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self) -> Dict[str, Any]:
        """Return ``{db_column_name: value}`` – handy for DataFrame / JSON output."""
        mapper = sa_inspect(type(self))
        return {attr.columns[0].name: getattr(self, attr.key) for attr in mapper.column_attrs}


def _partial_where(clause: str) -> Dict[str, Any]:
    """Dialect kwargs for a partial / filtered unique index (all three dialects support them)."""
    return {
        "sqlite_where": text(clause),
        "postgresql_where": text(clause),
        "mssql_where": text(clause),
    }


# ---------------------------------------------------------------------------
# tblRGRCoefficients
# ---------------------------------------------------------------------------
class RGRCoefficient(Base):
    __tablename__ = "tblRGRCoefficients"

    coefficient_id: Mapped[int] = mapped_column(
        "CoefficientID", Integer, primary_key=True, autoincrement=True
    )
    # Species models: SpecCode set, ForestUnit NULL.  FU models: SpecCode NULL, ForestUnit set.
    spec_code: Mapped[Optional[int]] = mapped_column("SpecCode", Integer, nullable=True)
    forest_unit: Mapped[Optional[str]] = mapped_column("ForestUnit", String(10), nullable=True)
    species_name: Mapped[Optional[str]] = mapped_column("SpeciesName", String(50), nullable=True)

    # RGR = Intercept + Coef_BA*BA + Coef_BA2*BA² + Coef_AGSfrac*AGSfrac + Coef_TCR*TCR
    #     + Coef_DQ*DQ + Coef_q*q + Coef_Cut*Cut
    intercept: Mapped[float] = mapped_column("Intercept", Float, nullable=False)
    coef_ba: Mapped[float] = mapped_column("Coef_BA", Float, nullable=False)
    coef_ba2: Mapped[float] = mapped_column(
        "Coef_BA2", Float, nullable=False, default=0.0, server_default=text("0")
    )
    coef_agsfrac: Mapped[float] = mapped_column("Coef_AGSfrac", Float, nullable=False)
    coef_tcr: Mapped[float] = mapped_column("Coef_TCR", Float, nullable=False)
    coef_dq: Mapped[float] = mapped_column("Coef_DQ", Float, nullable=False)
    coef_q: Mapped[float] = mapped_column("Coef_q", Float, nullable=False)
    coef_cut: Mapped[float] = mapped_column("Coef_Cut", Float, nullable=False)

    # Model quality
    r_squared: Mapped[Optional[float]] = mapped_column("R_Squared", Float)
    rmse: Mapped[Optional[float]] = mapped_column("RMSE", Float)
    mae: Mapped[Optional[float]] = mapped_column("MAE", Float)
    n_samples: Mapped[Optional[int]] = mapped_column("N_Samples", Integer)

    # Metadata
    trained_date: Mapped[Optional[datetime]] = mapped_column(
        "TrainedDate", DateTime, server_default=func.now(), default=_utcnow
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        "ModelVersion", String(20), default="1.0", server_default=text("'1.0'")
    )
    max_data_year: Mapped[Optional[int]] = mapped_column("MaxDataYear", Integer)
    notes: Mapped[Optional[str]] = mapped_column("Notes", Unicode(500))

    # Confidence / quality tracking (sql/add_confidence_columns.sql)
    confidence_level: Mapped[Optional[str]] = mapped_column("ConfidenceLevel", String(20))
    is_production_ready: Mapped[Optional[bool]] = mapped_column(
        "IsProductionReady", Boolean, default=False, server_default=text("0")
    )
    quality_flags: Mapped[Optional[str]] = mapped_column("QualityFlags", String(500))
    warning_message: Mapped[Optional[str]] = mapped_column("WarningMessage", Unicode(1000))
    wrong_sign_count: Mapped[Optional[int]] = mapped_column(
        "WrongSignCount", SmallInteger, default=0, server_default=text("0")
    )
    n_plots: Mapped[Optional[int]] = mapped_column("N_Plots", Integer)

    training_history: Mapped[List["RGRTrainingHistory"]] = relationship(back_populates="coefficient")
    history_snapshots: Mapped[List["RGRCoefficientHistory"]] = relationship(back_populates="coefficient")
    predictions: Mapped[List["RGRPrediction"]] = relationship(back_populates="coefficient")
    frontend_calcs: Mapped[List["RGRFrontendCalcData"]] = relationship(back_populates="coefficient")

    def __repr__(self) -> str:  # pragma: no cover – debugging aid
        return (
            f"<RGRCoefficient id={self.coefficient_id} spec={self.spec_code} "
            f"fu={self.forest_unit} v={self.model_version}>"
        )


# NULL-safe uniqueness: species models vs FU models vs legacy (both set) rows.
Index(
    "UQ_Species_Version",
    RGRCoefficient.spec_code,
    RGRCoefficient.model_version,
    unique=True,
    **_partial_where('"ForestUnit" IS NULL'),
)
Index(
    "UQ_ForestUnit_Version",
    RGRCoefficient.forest_unit,
    RGRCoefficient.model_version,
    unique=True,
    **_partial_where('"SpecCode" IS NULL'),
)
Index(
    "UQ_SpecForest_Version",
    RGRCoefficient.spec_code,
    RGRCoefficient.forest_unit,
    RGRCoefficient.model_version,
    unique=True,
    **_partial_where('"SpecCode" IS NOT NULL AND "ForestUnit" IS NOT NULL'),
)


# ---------------------------------------------------------------------------
# tblRGRCoefficientHistory – immutable snapshot per training run
# ---------------------------------------------------------------------------
class RGRCoefficientHistory(Base):
    __tablename__ = "tblRGRCoefficientHistory"

    history_id: Mapped[int] = mapped_column("HistoryID", Integer, primary_key=True, autoincrement=True)
    coefficient_id: Mapped[Optional[int]] = mapped_column(
        "CoefficientID", Integer, ForeignKey("tblRGRCoefficients.CoefficientID"), nullable=True
    )
    spec_code: Mapped[Optional[int]] = mapped_column("SpecCode", Integer)
    forest_unit: Mapped[Optional[str]] = mapped_column("ForestUnit", String(10))
    species_name: Mapped[Optional[str]] = mapped_column("SpeciesName", String(50))
    intercept: Mapped[Optional[float]] = mapped_column("Intercept", Float)
    coef_ba: Mapped[Optional[float]] = mapped_column("Coef_BA", Float)
    coef_ba2: Mapped[Optional[float]] = mapped_column("Coef_BA2", Float)
    coef_agsfrac: Mapped[Optional[float]] = mapped_column("Coef_AGSfrac", Float)
    coef_tcr: Mapped[Optional[float]] = mapped_column("Coef_TCR", Float)
    coef_dq: Mapped[Optional[float]] = mapped_column("Coef_DQ", Float)
    coef_q: Mapped[Optional[float]] = mapped_column("Coef_q", Float)
    coef_cut: Mapped[Optional[float]] = mapped_column("Coef_Cut", Float)
    r_squared: Mapped[Optional[float]] = mapped_column("R_Squared", Float)
    rmse: Mapped[Optional[float]] = mapped_column("RMSE", Float)
    mae: Mapped[Optional[float]] = mapped_column("MAE", Float)
    n_samples: Mapped[Optional[int]] = mapped_column("N_Samples", Integer)
    model_version: Mapped[Optional[str]] = mapped_column("ModelVersion", String(20))
    max_data_year: Mapped[Optional[int]] = mapped_column("MaxDataYear", Integer)
    notes: Mapped[Optional[str]] = mapped_column("Notes", Unicode(500))
    confidence_level: Mapped[Optional[str]] = mapped_column("ConfidenceLevel", String(20))
    is_production_ready: Mapped[Optional[bool]] = mapped_column("IsProductionReady", Boolean)
    quality_flags: Mapped[Optional[str]] = mapped_column("QualityFlags", String(500))
    warning_message: Mapped[Optional[str]] = mapped_column("WarningMessage", Unicode(1000))
    wrong_sign_count: Mapped[Optional[int]] = mapped_column("WrongSignCount", SmallInteger)
    n_plots: Mapped[Optional[int]] = mapped_column("N_Plots", Integer)
    trained_date: Mapped[Optional[datetime]] = mapped_column(
        "TrainedDate", DateTime, server_default=func.now(), default=_utcnow
    )

    coefficient: Mapped[Optional[RGRCoefficient]] = relationship(back_populates="history_snapshots")


Index(
    "IX_RGRCoeffHistory_Model",
    RGRCoefficientHistory.spec_code,
    RGRCoefficientHistory.forest_unit,
    RGRCoefficientHistory.model_version,
    RGRCoefficientHistory.trained_date.desc(),
)


# ---------------------------------------------------------------------------
# tblRGRTrainingHistory – audit trail of the training data used
# ---------------------------------------------------------------------------
class RGRTrainingHistory(Base):
    __tablename__ = "tblRGRTrainingHistory"

    training_id: Mapped[int] = mapped_column("TrainingID", Integer, primary_key=True, autoincrement=True)
    coefficient_id: Mapped[Optional[int]] = mapped_column(
        "CoefficientID", Integer, ForeignKey("tblRGRCoefficients.CoefficientID")
    )
    training_date: Mapped[Optional[datetime]] = mapped_column(
        "TrainingDate", DateTime, server_default=func.now(), default=_utcnow
    )
    n_plots: Mapped[Optional[int]] = mapped_column("N_Plots", Integer)
    n_growth_intervals: Mapped[Optional[int]] = mapped_column("N_GrowthIntervals", Integer)
    avg_growth_interval_years: Mapped[Optional[float]] = mapped_column("AvgGrowthInterval_Years", Float)
    min_ba: Mapped[Optional[float]] = mapped_column("MinBA", Float)
    max_ba: Mapped[Optional[float]] = mapped_column("MaxBA", Float)
    avg_ba: Mapped[Optional[float]] = mapped_column("AvgBA", Float)
    data_quality_flags: Mapped[Optional[str]] = mapped_column("DataQualityFlags", UnicodeText)  # JSON

    coefficient: Mapped[Optional[RGRCoefficient]] = relationship(back_populates="training_history")


# ---------------------------------------------------------------------------
# tblRGRPredictions – prediction audit / validation log
# ---------------------------------------------------------------------------
class RGRPrediction(Base):
    __tablename__ = "tblRGRPredictions"

    prediction_id: Mapped[int] = mapped_column("PredictionID", Integer, primary_key=True, autoincrement=True)
    coefficient_id: Mapped[Optional[int]] = mapped_column(
        "CoefficientID", Integer, ForeignKey("tblRGRCoefficients.CoefficientID")
    )
    plot_key: Mapped[Optional[int]] = mapped_column("PlotKey", Integer)
    measure_year: Mapped[Optional[int]] = mapped_column("MeasureYear", Integer)

    ba_input: Mapped[Optional[float]] = mapped_column("BA_Input", Float)
    agsfrac_input: Mapped[Optional[float]] = mapped_column("AGSfrac_Input", Float)
    tcr_input: Mapped[Optional[int]] = mapped_column("TCR_Input", SmallInteger)
    dq_input: Mapped[Optional[float]] = mapped_column("DQ_Input", Float)
    q_input: Mapped[Optional[float]] = mapped_column("q_Input", Float)
    cut_input: Mapped[Optional[bool]] = mapped_column("Cut_Input", Boolean)

    rgr_predicted: Mapped[Optional[float]] = mapped_column("RGR_Predicted", Float)
    bagrowth_predicted: Mapped[Optional[float]] = mapped_column("BAgrowth_Predicted", Float)
    rgr_actual: Mapped[Optional[float]] = mapped_column("RGR_Actual", Float)
    bagrowth_actual: Mapped[Optional[float]] = mapped_column("BAgrowth_Actual", Float)

    prediction_date: Mapped[Optional[datetime]] = mapped_column(
        "PredictionDate", DateTime, server_default=func.now(), default=_utcnow
    )
    residual: Mapped[Optional[float]] = mapped_column("Residual", Float)
    abs_error: Mapped[Optional[float]] = mapped_column("AbsError", Float)

    coefficient: Mapped[Optional[RGRCoefficient]] = relationship(back_populates="predictions")


# ---------------------------------------------------------------------------
# tblRGRFrontendCalcData – frontend calculation request snapshots
# ---------------------------------------------------------------------------
class RGRFrontendCalcData(Base):
    __tablename__ = "tblRGRFrontendCalcData"

    frontend_calc_id: Mapped[int] = mapped_column(
        "FrontendCalcID", Integer, primary_key=True, autoincrement=True
    )
    requested_date: Mapped[Optional[datetime]] = mapped_column(
        "RequestedDate", DateTime, server_default=func.now(), default=_utcnow
    )
    spec_code: Mapped[Optional[int]] = mapped_column("SpecCode", Integer)
    coefficient_id: Mapped[Optional[int]] = mapped_column(
        "CoefficientID", Integer, ForeignKey("tblRGRCoefficients.CoefficientID"), nullable=True
    )
    species_abbrev: Mapped[Optional[str]] = mapped_column("SpeciesAbbrev", String(20))
    forest_unit: Mapped[Optional[str]] = mapped_column("ForestUnit", String(10))
    model_version: Mapped[Optional[str]] = mapped_column("ModelVersion", String(20))
    current_year: Mapped[int] = mapped_column("CurrentYear", Integer, nullable=False)
    projection_years: Mapped[int] = mapped_column("ProjectionYears", Integer, nullable=False)
    extracted_rows: Mapped[int] = mapped_column("ExtractedRows", Integer, nullable=False)
    filtered_rows: Mapped[int] = mapped_column("FilteredRows", Integer, nullable=False)
    latest_rows: Mapped[int] = mapped_column("LatestRows", Integer, nullable=False)
    used_db_model: Mapped[bool] = mapped_column("UsedDBModel", Boolean, nullable=False)
    data_payload: Mapped[Optional[str]] = mapped_column("DataPayload", UnicodeText)  # JSON

    coefficient: Mapped[Optional[RGRCoefficient]] = relationship(back_populates="frontend_calcs")
    plot_rows: Mapped[List["RGRFrontendCalcPlotData"]] = relationship(
        back_populates="calc", cascade="all, delete-orphan"
    )


Index(
    "IX_FrontendCalcData_ModelDate",
    RGRFrontendCalcData.species_abbrev,
    RGRFrontendCalcData.forest_unit,
    RGRFrontendCalcData.requested_date.desc(),
)
Index("IX_FrontendCalcData_CoefficientID", RGRFrontendCalcData.coefficient_id)


# ---------------------------------------------------------------------------
# tblRGRFrontendCalcPlotData – per-plot rows for one frontend calculation
# ---------------------------------------------------------------------------
class RGRFrontendCalcPlotData(Base):
    __tablename__ = "tblRGRFrontendCalcPlotData"

    frontend_calc_plot_id: Mapped[int] = mapped_column(
        "FrontendCalcPlotID", Integer, primary_key=True, autoincrement=True
    )
    frontend_calc_id: Mapped[int] = mapped_column(
        "FrontendCalcID",
        Integer,
        ForeignKey("tblRGRFrontendCalcData.FrontendCalcID"),
        nullable=False,
    )
    plot_key: Mapped[Optional[str]] = mapped_column("PlotKey", String(50))
    start_year: Mapped[Optional[int]] = mapped_column("StartYear", Integer)
    end_year: Mapped[Optional[int]] = mapped_column("EndYear", Integer)
    growth_interval_years: Mapped[Optional[float]] = mapped_column("GrowthInterval_Years", Float)
    ba: Mapped[Optional[float]] = mapped_column("BA", Float)
    agsfrac: Mapped[Optional[float]] = mapped_column("AGSfrac", Float)
    tcr: Mapped[Optional[int]] = mapped_column("TCR", SmallInteger)
    dq: Mapped[Optional[float]] = mapped_column("DQ", Float)
    q: Mapped[Optional[float]] = mapped_column("q", Float)
    cut: Mapped[Optional[bool]] = mapped_column("Cut", Boolean)
    bagrowth_input: Mapped[Optional[float]] = mapped_column("BAgrowth_Input", Float)
    rgr_input: Mapped[Optional[float]] = mapped_column("RGR_Input", Float)
    rgr_predicted: Mapped[Optional[float]] = mapped_column("RGR_Predicted", Float)
    bagrowth_predicted: Mapped[Optional[float]] = mapped_column("BAgrowth_Predicted", Float)

    calc: Mapped[RGRFrontendCalcData] = relationship(back_populates="plot_rows")


Index("IX_FrontendCalcPlotData_FrontendCalcID", RGRFrontendCalcPlotData.frontend_calc_id)
Index("IX_FrontendCalcPlotData_PlotKey", RGRFrontendCalcPlotData.plot_key)


# ---------------------------------------------------------------------------
# tblBAGrowthData – portable growth-interval training data
# ---------------------------------------------------------------------------
class BAGrowthInterval(Base):
    """
    One basal-area growth interval for a plot.

    On SQL Server the training data is computed on the fly by the
    ``sp_GetBAGrowthData_*`` stored procedures from the raw inventory tables.
    SQLite / PostgreSQL deployments have no inventory tables, so this flat
    table stores the same row shape the procedures return:

    * species rows : ``SpecCode`` set, BA/DQ/q for the target species only
    * stand rows   : ``SpecCode`` NULL, ``ForestUnit`` set, BA/DQ/q across all species
    """

    __tablename__ = "tblBAGrowthData"

    growth_data_id: Mapped[int] = mapped_column(
        "GrowthDataID", Integer, primary_key=True, autoincrement=True
    )
    spec_code: Mapped[Optional[int]] = mapped_column("SpecCode", Integer, index=True)
    forest_unit: Mapped[Optional[str]] = mapped_column("ForestUnit", String(10), index=True)
    plot_key: Mapped[int] = mapped_column("PlotKey", Integer, nullable=False, index=True)
    plot_name: Mapped[Optional[str]] = mapped_column("PlotName", String(50))
    dataset_code: Mapped[Optional[str]] = mapped_column("DatasetCode", String(20))
    start_year: Mapped[int] = mapped_column("StartYear", Integer, nullable=False)
    end_year: Mapped[int] = mapped_column("EndYear", Integer, nullable=False)
    growth_interval_years: Mapped[Optional[float]] = mapped_column("GrowthInterval_Years", Float)

    ba: Mapped[Optional[float]] = mapped_column("BA", Float)
    agsfrac: Mapped[Optional[float]] = mapped_column("AGSfrac", Float)
    tcr: Mapped[Optional[int]] = mapped_column("TCR", SmallInteger)
    dq: Mapped[Optional[float]] = mapped_column("DQ", Float)
    q: Mapped[Optional[float]] = mapped_column("q", Float)
    cut: Mapped[Optional[int]] = mapped_column("Cut", SmallInteger)
    bagrowth: Mapped[Optional[float]] = mapped_column("BAgrowth", Float)
    rgr: Mapped[Optional[float]] = mapped_column("RGR", Float)

    latitude: Mapped[Optional[float]] = mapped_column("Latitude", Float)
    longitude: Mapped[Optional[float]] = mapped_column("Longitude", Float)
    easting: Mapped[Optional[float]] = mapped_column("Easting", Float)
    northing: Mapped[Optional[float]] = mapped_column("Northing", Float)
    tree_count: Mapped[Optional[int]] = mapped_column("TreeCount", Integer)
    shrinkage_count: Mapped[Optional[int]] = mapped_column("ShrinkageCount", Integer)


# Column names returned by the MSSQL stored procedures – used to build DataFrames
# with an identical shape on every dialect (primary key excluded).
GROWTH_DATA_COLUMNS = [str(column.name) for column in BAGrowthInterval.__table__.columns][1:]


__all__ = [
    "Base",
    "RGRCoefficient",
    "RGRCoefficientHistory",
    "RGRTrainingHistory",
    "RGRPrediction",
    "RGRFrontendCalcData",
    "RGRFrontendCalcPlotData",
    "BAGrowthInterval",
    "GROWTH_DATA_COLUMNS",
]
