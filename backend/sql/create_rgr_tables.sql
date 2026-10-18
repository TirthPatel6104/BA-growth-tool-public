/*
================================================================================
RGR Coefficient Storage Tables
Run this script once against the target database to create the required tables.
================================================================================
*/

-- Table 1: Fitted RGR linear regression coefficients per species/forest unit
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRCoefficients'
)
BEGIN
    CREATE TABLE tblRGRCoefficients (
        CoefficientID   INT IDENTITY(1,1) PRIMARY KEY,
        -- Species models: SpecCode set, ForestUnit NULL (pools all forest units)
        -- FU models     : SpecCode NULL, ForestUnit set (stand-level)
        -- Legacy rows   : both set (old per-FU species model)
        SpecCode        INT          NULL,
        ForestUnit      VARCHAR(10)  NULL,
        SpeciesName     VARCHAR(50),

        -- RGR = Intercept + Coef_BA*BA + Coef_AGSfrac*AGSfrac
        --     + Coef_TCR*TCR + Coef_DQ*DQ + Coef_q*q + Coef_Cut*Cut
        Intercept       FLOAT        NOT NULL,
        Coef_BA         FLOAT        NOT NULL,
        Coef_BA2        FLOAT        NOT NULL DEFAULT 0,
        Coef_AGSfrac    FLOAT        NOT NULL,
        Coef_TCR        FLOAT        NOT NULL,
        Coef_DQ         FLOAT        NOT NULL,
        Coef_q          FLOAT        NOT NULL,
        Coef_Cut        FLOAT        NOT NULL,

        -- Model quality
        R_Squared       FLOAT,
        RMSE            FLOAT,
        MAE             FLOAT,
        N_Samples       INT,

        -- Metadata
        TrainedDate     DATETIME     DEFAULT GETDATE(),
        ModelVersion    VARCHAR(20)  DEFAULT '1.0',
        MaxDataYear     INT,          -- max EndYear of training data (for incremental check)
        Notes           NVARCHAR(500),

        -- NOTE: uniqueness enforced by filtered indexes below (NULLs need special handling)
    );
    PRINT 'Created tblRGRCoefficients';
END
ELSE
    PRINT 'tblRGRCoefficients already exists, skipping.';
GO

-- Filtered unique indexes: NULL-safe uniqueness for species vs FU models
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='UQ_Species_Version' AND object_id=OBJECT_ID('tblRGRCoefficients'))
    CREATE UNIQUE INDEX UQ_Species_Version   ON tblRGRCoefficients (SpecCode,    ModelVersion) WHERE ForestUnit IS NULL;
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='UQ_ForestUnit_Version' AND object_id=OBJECT_ID('tblRGRCoefficients'))
    CREATE UNIQUE INDEX UQ_ForestUnit_Version ON tblRGRCoefficients (ForestUnit, ModelVersion) WHERE SpecCode    IS NULL;
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='UQ_SpecForest_Version' AND object_id=OBJECT_ID('tblRGRCoefficients'))
    CREATE UNIQUE INDEX UQ_SpecForest_Version ON tblRGRCoefficients (SpecCode, ForestUnit, ModelVersion) WHERE SpecCode IS NOT NULL AND ForestUnit IS NOT NULL;
GO

-- Table 2: Training run history (audit trail)
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRTrainingHistory'
)
BEGIN
    CREATE TABLE tblRGRTrainingHistory (
        TrainingID              INT IDENTITY(1,1) PRIMARY KEY,
        CoefficientID           INT FOREIGN KEY REFERENCES tblRGRCoefficients(CoefficientID),
        TrainingDate            DATETIME     DEFAULT GETDATE(),
        N_Plots                 INT,
        N_GrowthIntervals       INT,
        AvgGrowthInterval_Years FLOAT,
        MinBA                   FLOAT,
        MaxBA                   FLOAT,
        AvgBA                   FLOAT,
        DataQualityFlags        NVARCHAR(MAX)  -- JSON with QA metrics
    );
    PRINT 'Created tblRGRTrainingHistory';
END
ELSE
    PRINT 'tblRGRTrainingHistory already exists, skipping.';
GO

-- Table 3: Predictions for validation and audit
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRPredictions'
)
BEGIN
    CREATE TABLE tblRGRPredictions (
        PredictionID        INT IDENTITY(1,1) PRIMARY KEY,
        CoefficientID       INT FOREIGN KEY REFERENCES tblRGRCoefficients(CoefficientID),
        PlotKey             INT,
        MeasureYear         INT,

        -- Input values
        BA_Input            FLOAT,
        AGSfrac_Input       FLOAT,
        TCR_Input           TINYINT,
        DQ_Input            FLOAT,
        q_Input             FLOAT,
        Cut_Input           BIT,

        -- Predicted
        RGR_Predicted       FLOAT,
        BAgrowth_Predicted  FLOAT,

        -- Actual (for validation)
        RGR_Actual          FLOAT,
        BAgrowth_Actual     FLOAT,

        -- Diagnostics
        PredictionDate      DATETIME DEFAULT GETDATE(),
        Residual            FLOAT,
        AbsError            FLOAT
    );
    PRINT 'Created tblRGRPredictions';
END
ELSE
    PRINT 'tblRGRPredictions already exists, skipping.';
GO

-- Table 4: Immutable coefficient snapshots per retraining run
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRCoefficientHistory'
)
BEGIN
    CREATE TABLE tblRGRCoefficientHistory (
        HistoryID            INT IDENTITY(1,1) PRIMARY KEY,
        CoefficientID        INT NULL FOREIGN KEY REFERENCES tblRGRCoefficients(CoefficientID),
        SpecCode             INT NULL,
        ForestUnit           VARCHAR(10) NULL,
        SpeciesName          VARCHAR(50) NULL,
        Intercept            FLOAT NULL,
        Coef_BA              FLOAT NULL,
        Coef_BA2             FLOAT NULL,
        Coef_AGSfrac         FLOAT NULL,
        Coef_TCR             FLOAT NULL,
        Coef_DQ              FLOAT NULL,
        Coef_q               FLOAT NULL,
        Coef_Cut             FLOAT NULL,
        R_Squared            FLOAT NULL,
        RMSE                 FLOAT NULL,
        MAE                  FLOAT NULL,
        N_Samples            INT NULL,
        ModelVersion         VARCHAR(20) NULL,
        MaxDataYear          INT NULL,
        Notes                NVARCHAR(500) NULL,
        ConfidenceLevel      VARCHAR(20) NULL,
        IsProductionReady    BIT NULL,
        QualityFlags         VARCHAR(500) NULL,
        WarningMessage       NVARCHAR(1000) NULL,
        WrongSignCount       TINYINT NULL,
        N_Plots              INT NULL,
        TrainedDate          DATETIME DEFAULT GETDATE()
    );

    CREATE INDEX IX_RGRCoeffHistory_Model
        ON tblRGRCoefficientHistory(SpecCode, ForestUnit, ModelVersion, TrainedDate DESC);

    PRINT 'Created tblRGRCoefficientHistory';
END
ELSE
    PRINT 'tblRGRCoefficientHistory already exists, skipping.';
GO

-- Table 5: Persist frontend growth-calculation fetched data snapshots
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRFrontendCalcData'
)
BEGIN
    CREATE TABLE tblRGRFrontendCalcData (
        FrontendCalcID       INT IDENTITY(1,1) PRIMARY KEY,
        RequestedDate        DATETIME DEFAULT GETDATE(),
        SpecCode             INT NULL,
        CoefficientID        INT NULL FOREIGN KEY REFERENCES tblRGRCoefficients(CoefficientID),
        SpeciesAbbrev        VARCHAR(20) NULL,
        ForestUnit           VARCHAR(10) NULL,
        ModelVersion         VARCHAR(20) NULL,
        CurrentYear          INT NOT NULL,
        ProjectionYears      INT NOT NULL,
        ExtractedRows        INT NOT NULL,
        FilteredRows         INT NOT NULL,
        LatestRows           INT NOT NULL,
        UsedDBModel          BIT NOT NULL,
        DataPayload          NVARCHAR(MAX) NULL
    );

    CREATE INDEX IX_FrontendCalcData_ModelDate
        ON tblRGRFrontendCalcData(SpeciesAbbrev, ForestUnit, RequestedDate DESC);

    CREATE INDEX IX_FrontendCalcData_CoefficientID
        ON tblRGRFrontendCalcData(CoefficientID);

    PRINT 'Created tblRGRFrontendCalcData';
END
ELSE
    PRINT 'tblRGRFrontendCalcData already exists, skipping.';
GO

-- Table 6: Persist per-plot rows used in frontend growth calculations
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRFrontendCalcPlotData'
)
BEGIN
    CREATE TABLE tblRGRFrontendCalcPlotData (
        FrontendCalcPlotID    INT IDENTITY(1,1) PRIMARY KEY,
        FrontendCalcID        INT NOT NULL FOREIGN KEY REFERENCES tblRGRFrontendCalcData(FrontendCalcID),
        PlotKey               VARCHAR(50) NULL,
        StartYear             INT NULL,
        EndYear               INT NULL,
        GrowthInterval_Years  FLOAT NULL,
        BA                    FLOAT NULL,
        AGSfrac               FLOAT NULL,
        TCR                   TINYINT NULL,
        DQ                    FLOAT NULL,
        q                     FLOAT NULL,
        Cut                   BIT NULL,
        BAgrowth_Input        FLOAT NULL,
        RGR_Input             FLOAT NULL,
        RGR_Predicted         FLOAT NULL,
        BAgrowth_Predicted    FLOAT NULL
    );

    CREATE INDEX IX_FrontendCalcPlotData_FrontendCalcID
        ON tblRGRFrontendCalcPlotData(FrontendCalcID);

    CREATE INDEX IX_FrontendCalcPlotData_PlotKey
        ON tblRGRFrontendCalcPlotData(PlotKey);

    PRINT 'Created tblRGRFrontendCalcPlotData';
END
ELSE
    PRINT 'tblRGRFrontendCalcPlotData already exists, skipping.';
GO

PRINT 'Setup complete.';
