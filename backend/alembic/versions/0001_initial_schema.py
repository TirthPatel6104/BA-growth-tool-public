"""initial schema

Creates the RGR coefficient store exactly as it existed in the MSSQL-only
version of the tool (``sql/create_rgr_tables.sql`` + ``sql/add_confidence_columns.sql``)
plus the portable ``tblBAGrowthData`` table that replaces the stored
procedures on SQLite / PostgreSQL.

Existing SQL Server databases that were created with the T-SQL scripts already
have this schema – run ``alembic stamp head`` there instead of ``upgrade``.

Revision ID: a1ffabe7dd95
Revises:
Create Date: 2026-09-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1ffabe7dd95"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _partial(clause: str) -> dict:
    """Filtered / partial index kwargs for every supported dialect."""
    return {
        "sqlite_where": sa.text(clause),
        "postgresql_where": sa.text(clause),
        "mssql_where": sa.text(clause),
    }


def upgrade() -> None:
    # ------------------------------------------------------------------
    # tblRGRCoefficients – current fitted coefficients per species / FU
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRCoefficients",
        sa.Column("CoefficientID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("SpecCode", sa.Integer(), nullable=True),
        sa.Column("ForestUnit", sa.String(length=10), nullable=True),
        sa.Column("SpeciesName", sa.String(length=50), nullable=True),
        sa.Column("Intercept", sa.Float(), nullable=False),
        sa.Column("Coef_BA", sa.Float(), nullable=False),
        sa.Column("Coef_BA2", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("Coef_AGSfrac", sa.Float(), nullable=False),
        sa.Column("Coef_TCR", sa.Float(), nullable=False),
        sa.Column("Coef_DQ", sa.Float(), nullable=False),
        sa.Column("Coef_q", sa.Float(), nullable=False),
        sa.Column("Coef_Cut", sa.Float(), nullable=False),
        sa.Column("R_Squared", sa.Float(), nullable=True),
        sa.Column("RMSE", sa.Float(), nullable=True),
        sa.Column("MAE", sa.Float(), nullable=True),
        sa.Column("N_Samples", sa.Integer(), nullable=True),
        sa.Column("TrainedDate", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("ModelVersion", sa.String(length=20), server_default=sa.text("'1.0'"), nullable=True),
        sa.Column("MaxDataYear", sa.Integer(), nullable=True),
        sa.Column("Notes", sa.Unicode(length=500), nullable=True),
        sa.Column("ConfidenceLevel", sa.String(length=20), nullable=True),
        sa.Column("IsProductionReady", sa.Boolean(), server_default=sa.text("0"), nullable=True),
        sa.Column("QualityFlags", sa.String(length=500), nullable=True),
        sa.Column("WarningMessage", sa.Unicode(length=1000), nullable=True),
        sa.Column("WrongSignCount", sa.SmallInteger(), server_default=sa.text("0"), nullable=True),
        sa.Column("N_Plots", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("CoefficientID", name=op.f("pk_tblRGRCoefficients")),
    )
    # NULL-safe uniqueness (species model / FU model / legacy both-set rows)
    op.create_index(
        "UQ_Species_Version",
        "tblRGRCoefficients",
        ["SpecCode", "ModelVersion"],
        unique=True,
        **_partial('"ForestUnit" IS NULL'),
    )
    op.create_index(
        "UQ_ForestUnit_Version",
        "tblRGRCoefficients",
        ["ForestUnit", "ModelVersion"],
        unique=True,
        **_partial('"SpecCode" IS NULL'),
    )
    op.create_index(
        "UQ_SpecForest_Version",
        "tblRGRCoefficients",
        ["SpecCode", "ForestUnit", "ModelVersion"],
        unique=True,
        **_partial('"SpecCode" IS NOT NULL AND "ForestUnit" IS NOT NULL'),
    )

    # ------------------------------------------------------------------
    # tblRGRCoefficientHistory – immutable snapshot per training run
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRCoefficientHistory",
        sa.Column("HistoryID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("CoefficientID", sa.Integer(), nullable=True),
        sa.Column("SpecCode", sa.Integer(), nullable=True),
        sa.Column("ForestUnit", sa.String(length=10), nullable=True),
        sa.Column("SpeciesName", sa.String(length=50), nullable=True),
        sa.Column("Intercept", sa.Float(), nullable=True),
        sa.Column("Coef_BA", sa.Float(), nullable=True),
        sa.Column("Coef_BA2", sa.Float(), nullable=True),
        sa.Column("Coef_AGSfrac", sa.Float(), nullable=True),
        sa.Column("Coef_TCR", sa.Float(), nullable=True),
        sa.Column("Coef_DQ", sa.Float(), nullable=True),
        sa.Column("Coef_q", sa.Float(), nullable=True),
        sa.Column("Coef_Cut", sa.Float(), nullable=True),
        sa.Column("R_Squared", sa.Float(), nullable=True),
        sa.Column("RMSE", sa.Float(), nullable=True),
        sa.Column("MAE", sa.Float(), nullable=True),
        sa.Column("N_Samples", sa.Integer(), nullable=True),
        sa.Column("ModelVersion", sa.String(length=20), nullable=True),
        sa.Column("MaxDataYear", sa.Integer(), nullable=True),
        sa.Column("Notes", sa.Unicode(length=500), nullable=True),
        sa.Column("ConfidenceLevel", sa.String(length=20), nullable=True),
        sa.Column("IsProductionReady", sa.Boolean(), nullable=True),
        sa.Column("QualityFlags", sa.String(length=500), nullable=True),
        sa.Column("WarningMessage", sa.Unicode(length=1000), nullable=True),
        sa.Column("WrongSignCount", sa.SmallInteger(), nullable=True),
        sa.Column("N_Plots", sa.Integer(), nullable=True),
        sa.Column("TrainedDate", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(
            ["CoefficientID"],
            ["tblRGRCoefficients.CoefficientID"],
            name=op.f("fk_tblRGRCoefficientHistory_CoefficientID_tblRGRCoefficients"),
        ),
        sa.PrimaryKeyConstraint("HistoryID", name=op.f("pk_tblRGRCoefficientHistory")),
    )
    op.create_index(
        "IX_RGRCoeffHistory_Model",
        "tblRGRCoefficientHistory",
        ["SpecCode", "ForestUnit", "ModelVersion", sa.literal_column('"TrainedDate" DESC')],
        unique=False,
    )

    # ------------------------------------------------------------------
    # tblRGRTrainingHistory – training-data QA summary per run
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRTrainingHistory",
        sa.Column("TrainingID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("CoefficientID", sa.Integer(), nullable=True),
        sa.Column("TrainingDate", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("N_Plots", sa.Integer(), nullable=True),
        sa.Column("N_GrowthIntervals", sa.Integer(), nullable=True),
        sa.Column("AvgGrowthInterval_Years", sa.Float(), nullable=True),
        sa.Column("MinBA", sa.Float(), nullable=True),
        sa.Column("MaxBA", sa.Float(), nullable=True),
        sa.Column("AvgBA", sa.Float(), nullable=True),
        sa.Column("DataQualityFlags", sa.UnicodeText(), nullable=True),
        sa.ForeignKeyConstraint(
            ["CoefficientID"],
            ["tblRGRCoefficients.CoefficientID"],
            name=op.f("fk_tblRGRTrainingHistory_CoefficientID_tblRGRCoefficients"),
        ),
        sa.PrimaryKeyConstraint("TrainingID", name=op.f("pk_tblRGRTrainingHistory")),
    )

    # ------------------------------------------------------------------
    # tblRGRPredictions – prediction audit / validation log
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRPredictions",
        sa.Column("PredictionID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("CoefficientID", sa.Integer(), nullable=True),
        sa.Column("PlotKey", sa.Integer(), nullable=True),
        sa.Column("MeasureYear", sa.Integer(), nullable=True),
        sa.Column("BA_Input", sa.Float(), nullable=True),
        sa.Column("AGSfrac_Input", sa.Float(), nullable=True),
        sa.Column("TCR_Input", sa.SmallInteger(), nullable=True),
        sa.Column("DQ_Input", sa.Float(), nullable=True),
        sa.Column("q_Input", sa.Float(), nullable=True),
        sa.Column("Cut_Input", sa.Boolean(), nullable=True),
        sa.Column("RGR_Predicted", sa.Float(), nullable=True),
        sa.Column("BAgrowth_Predicted", sa.Float(), nullable=True),
        sa.Column("RGR_Actual", sa.Float(), nullable=True),
        sa.Column("BAgrowth_Actual", sa.Float(), nullable=True),
        sa.Column("PredictionDate", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("Residual", sa.Float(), nullable=True),
        sa.Column("AbsError", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["CoefficientID"],
            ["tblRGRCoefficients.CoefficientID"],
            name=op.f("fk_tblRGRPredictions_CoefficientID_tblRGRCoefficients"),
        ),
        sa.PrimaryKeyConstraint("PredictionID", name=op.f("pk_tblRGRPredictions")),
    )

    # ------------------------------------------------------------------
    # tblRGRFrontendCalcData – frontend calculation request snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRFrontendCalcData",
        sa.Column("FrontendCalcID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("RequestedDate", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("SpecCode", sa.Integer(), nullable=True),
        sa.Column("CoefficientID", sa.Integer(), nullable=True),
        sa.Column("SpeciesAbbrev", sa.String(length=20), nullable=True),
        sa.Column("ForestUnit", sa.String(length=10), nullable=True),
        sa.Column("ModelVersion", sa.String(length=20), nullable=True),
        sa.Column("CurrentYear", sa.Integer(), nullable=False),
        sa.Column("ProjectionYears", sa.Integer(), nullable=False),
        sa.Column("ExtractedRows", sa.Integer(), nullable=False),
        sa.Column("FilteredRows", sa.Integer(), nullable=False),
        sa.Column("LatestRows", sa.Integer(), nullable=False),
        sa.Column("UsedDBModel", sa.Boolean(), nullable=False),
        sa.Column("DataPayload", sa.UnicodeText(), nullable=True),
        sa.ForeignKeyConstraint(
            ["CoefficientID"],
            ["tblRGRCoefficients.CoefficientID"],
            name=op.f("fk_tblRGRFrontendCalcData_CoefficientID_tblRGRCoefficients"),
        ),
        sa.PrimaryKeyConstraint("FrontendCalcID", name=op.f("pk_tblRGRFrontendCalcData")),
    )
    op.create_index(
        "IX_FrontendCalcData_CoefficientID", "tblRGRFrontendCalcData", ["CoefficientID"], unique=False
    )
    op.create_index(
        "IX_FrontendCalcData_ModelDate",
        "tblRGRFrontendCalcData",
        ["SpeciesAbbrev", "ForestUnit", sa.literal_column('"RequestedDate" DESC')],
        unique=False,
    )

    # ------------------------------------------------------------------
    # tblRGRFrontendCalcPlotData – per-plot rows for one frontend calculation
    # ------------------------------------------------------------------
    op.create_table(
        "tblRGRFrontendCalcPlotData",
        sa.Column("FrontendCalcPlotID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("FrontendCalcID", sa.Integer(), nullable=False),
        sa.Column("PlotKey", sa.String(length=50), nullable=True),
        sa.Column("StartYear", sa.Integer(), nullable=True),
        sa.Column("EndYear", sa.Integer(), nullable=True),
        sa.Column("GrowthInterval_Years", sa.Float(), nullable=True),
        sa.Column("BA", sa.Float(), nullable=True),
        sa.Column("AGSfrac", sa.Float(), nullable=True),
        sa.Column("TCR", sa.SmallInteger(), nullable=True),
        sa.Column("DQ", sa.Float(), nullable=True),
        sa.Column("q", sa.Float(), nullable=True),
        sa.Column("Cut", sa.Boolean(), nullable=True),
        sa.Column("BAgrowth_Input", sa.Float(), nullable=True),
        sa.Column("RGR_Input", sa.Float(), nullable=True),
        sa.Column("RGR_Predicted", sa.Float(), nullable=True),
        sa.Column("BAgrowth_Predicted", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["FrontendCalcID"],
            ["tblRGRFrontendCalcData.FrontendCalcID"],
            name=op.f("fk_tblRGRFrontendCalcPlotData_FrontendCalcID_tblRGRFrontendCalcData"),
        ),
        sa.PrimaryKeyConstraint("FrontendCalcPlotID", name=op.f("pk_tblRGRFrontendCalcPlotData")),
    )
    op.create_index(
        "IX_FrontendCalcPlotData_FrontendCalcID",
        "tblRGRFrontendCalcPlotData",
        ["FrontendCalcID"],
        unique=False,
    )
    op.create_index(
        "IX_FrontendCalcPlotData_PlotKey", "tblRGRFrontendCalcPlotData", ["PlotKey"], unique=False
    )

    # ------------------------------------------------------------------
    # tblBAGrowthData – portable growth-interval training data
    # (replaces the MSSQL sp_GetBAGrowthData_* procedures on SQLite / Postgres)
    # ------------------------------------------------------------------
    op.create_table(
        "tblBAGrowthData",
        sa.Column("GrowthDataID", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("SpecCode", sa.Integer(), nullable=True),
        sa.Column("ForestUnit", sa.String(length=10), nullable=True),
        sa.Column("PlotKey", sa.Integer(), nullable=False),
        sa.Column("PlotName", sa.String(length=50), nullable=True),
        sa.Column("DatasetCode", sa.String(length=20), nullable=True),
        sa.Column("StartYear", sa.Integer(), nullable=False),
        sa.Column("EndYear", sa.Integer(), nullable=False),
        sa.Column("GrowthInterval_Years", sa.Float(), nullable=True),
        sa.Column("BA", sa.Float(), nullable=True),
        sa.Column("AGSfrac", sa.Float(), nullable=True),
        sa.Column("TCR", sa.SmallInteger(), nullable=True),
        sa.Column("DQ", sa.Float(), nullable=True),
        sa.Column("q", sa.Float(), nullable=True),
        sa.Column("Cut", sa.SmallInteger(), nullable=True),
        sa.Column("BAgrowth", sa.Float(), nullable=True),
        sa.Column("RGR", sa.Float(), nullable=True),
        sa.Column("Latitude", sa.Float(), nullable=True),
        sa.Column("Longitude", sa.Float(), nullable=True),
        sa.Column("Easting", sa.Float(), nullable=True),
        sa.Column("Northing", sa.Float(), nullable=True),
        sa.Column("TreeCount", sa.Integer(), nullable=True),
        sa.Column("ShrinkageCount", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("GrowthDataID", name=op.f("pk_tblBAGrowthData")),
    )
    op.create_index(op.f("ix_tblBAGrowthData_SpecCode"), "tblBAGrowthData", ["SpecCode"], unique=False)
    op.create_index(op.f("ix_tblBAGrowthData_ForestUnit"), "tblBAGrowthData", ["ForestUnit"], unique=False)
    op.create_index(op.f("ix_tblBAGrowthData_PlotKey"), "tblBAGrowthData", ["PlotKey"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_tblBAGrowthData_PlotKey"), table_name="tblBAGrowthData")
    op.drop_index(op.f("ix_tblBAGrowthData_ForestUnit"), table_name="tblBAGrowthData")
    op.drop_index(op.f("ix_tblBAGrowthData_SpecCode"), table_name="tblBAGrowthData")
    op.drop_table("tblBAGrowthData")

    op.drop_index("IX_FrontendCalcPlotData_PlotKey", table_name="tblRGRFrontendCalcPlotData")
    op.drop_index("IX_FrontendCalcPlotData_FrontendCalcID", table_name="tblRGRFrontendCalcPlotData")
    op.drop_table("tblRGRFrontendCalcPlotData")

    op.drop_index("IX_FrontendCalcData_ModelDate", table_name="tblRGRFrontendCalcData")
    op.drop_index("IX_FrontendCalcData_CoefficientID", table_name="tblRGRFrontendCalcData")
    op.drop_table("tblRGRFrontendCalcData")

    op.drop_table("tblRGRPredictions")
    op.drop_table("tblRGRTrainingHistory")

    op.drop_index("IX_RGRCoeffHistory_Model", table_name="tblRGRCoefficientHistory")
    op.drop_table("tblRGRCoefficientHistory")

    op.drop_index("UQ_SpecForest_Version", table_name="tblRGRCoefficients")
    op.drop_index("UQ_ForestUnit_Version", table_name="tblRGRCoefficients")
    op.drop_index("UQ_Species_Version", table_name="tblRGRCoefficients")
    op.drop_table("tblRGRCoefficients")
