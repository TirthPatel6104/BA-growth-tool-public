/*
================================================================================
Schema migration – adds MaxDataYear column to tblRGRCoefficients.
Run once against the target database on any install created before this column was added.

  python scripts/update_schema.py
================================================================================
*/

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME  = 'tblRGRCoefficients'
      AND COLUMN_NAME = 'MaxDataYear'
)
BEGIN
    ALTER TABLE tblRGRCoefficients ADD MaxDataYear INT NULL;
    PRINT '✓ Added MaxDataYear column to tblRGRCoefficients';
END
ELSE
    PRINT 'MaxDataYear already exists, skipping.';
GO
