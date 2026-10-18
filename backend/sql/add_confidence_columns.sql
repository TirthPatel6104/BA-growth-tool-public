/*
================================================================================
Add Confidence & Quality Tracking to tblRGRCoefficients
================================================================================
Run once:
    python scripts/update_schema.py
*/

-- Add BA2 coefficient column to main coefficients table
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'tblRGRCoefficients' AND COLUMN_NAME = 'Coef_BA2'
)
BEGIN
    ALTER TABLE tblRGRCoefficients
    ADD Coef_BA2 FLOAT NOT NULL CONSTRAINT DF_tblRGRCoefficients_Coef_BA2 DEFAULT 0;

    PRINT '✓ Added Coef_BA2 to tblRGRCoefficients';
END
ELSE
    PRINT 'Coef_BA2 already exists in tblRGRCoefficients, skipping.';
GO

-- Add confidence tracking columns
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'tblRGRCoefficients' AND COLUMN_NAME = 'ConfidenceLevel'
)
BEGIN
    ALTER TABLE tblRGRCoefficients
    ADD ConfidenceLevel      VARCHAR(20)     NULL,      -- HIGH/MEDIUM/LOW/EXPERIMENTAL
        IsProductionReady    BIT             DEFAULT 0,
        QualityFlags         VARCHAR(500)    NULL,      -- Comma-separated flags
        WarningMessage       NVARCHAR(1000)  NULL,      -- User-facing warning
        WrongSignCount       TINYINT         DEFAULT 0, -- How many coefficients have wrong signs
        N_Plots              INT             NULL;      -- Number of unique plots

    PRINT '✓ Added confidence tracking columns';
END
ELSE
    PRINT 'Confidence columns already exist, skipping.';
GO

-- Create immutable retrain history snapshots for coefficients
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

    PRINT '✓ Created tblRGRCoefficientHistory';
END
ELSE
    PRINT 'tblRGRCoefficientHistory already exists, skipping.';
GO

-- Add BA2 coefficient column to coefficient history table when already present
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRCoefficientHistory'
)
AND NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'tblRGRCoefficientHistory' AND COLUMN_NAME = 'Coef_BA2'
)
BEGIN
    ALTER TABLE tblRGRCoefficientHistory
    ADD Coef_BA2 FLOAT NULL;

    PRINT '✓ Added Coef_BA2 to tblRGRCoefficientHistory';
END
ELSE
    PRINT 'Coef_BA2 already exists in tblRGRCoefficientHistory, skipping.';
GO

-- Store frontend growth-calculation fetched data snapshots
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'tblRGRFrontendCalcData'
)
BEGIN
    CREATE TABLE tblRGRFrontendCalcData (
        FrontendCalcID       INT IDENTITY(1,1) PRIMARY KEY,
        RequestedDate        DATETIME DEFAULT GETDATE(),
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

    PRINT '✓ Created tblRGRFrontendCalcData';
END
ELSE
    PRINT 'tblRGRFrontendCalcData already exists, skipping.';
GO

-- Add direct model linkage columns to frontend calc header table
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'tblRGRFrontendCalcData' AND COLUMN_NAME = 'SpecCode'
)
BEGIN
    ALTER TABLE tblRGRFrontendCalcData
    ADD SpecCode INT NULL;
    PRINT '✓ Added SpecCode to tblRGRFrontendCalcData';
END
ELSE
    PRINT 'SpecCode already exists in tblRGRFrontendCalcData, skipping.';
GO

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'tblRGRFrontendCalcData' AND COLUMN_NAME = 'CoefficientID'
)
BEGIN
    ALTER TABLE tblRGRFrontendCalcData
    ADD CoefficientID INT NULL;

    ALTER TABLE tblRGRFrontendCalcData
    ADD CONSTRAINT FK_tblRGRFrontendCalcData_CoefficientID
        FOREIGN KEY (CoefficientID) REFERENCES tblRGRCoefficients(CoefficientID);

    CREATE INDEX IX_FrontendCalcData_CoefficientID
        ON tblRGRFrontendCalcData(CoefficientID);

    PRINT '✓ Added CoefficientID FK to tblRGRFrontendCalcData';
END
ELSE
    PRINT 'CoefficientID already exists in tblRGRFrontendCalcData, skipping.';
GO

-- Store per-plot frontend growth-calculation rows used for prediction
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

    PRINT '✓ Created tblRGRFrontendCalcPlotData';
END
ELSE
    PRINT 'tblRGRFrontendCalcPlotData already exists, skipping.';
GO