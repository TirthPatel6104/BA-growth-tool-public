# `backend/sql/`

Schema scripts for the tool's **own** tables plus the contract for optional,
site-specific stored procedures.

| File | Purpose |
| :--- | :--- |
| `create_rgr_tables.sql` | Legacy T-SQL for `tblRGRCoefficients`, `tblRGRTrainingHistory`, … (Alembic now owns this schema) |
| `alter_rgr_tables.sql`, `add_confidence_columns.sql` | Legacy incremental changes, superseded by `alembic/versions/` |
| `sp_*.sql` | **Not committed.** Site-specific stored procedures (git-ignored) |

## Why the stored procedures are not in the repository

In a real deployment the growth intervals are computed from a forest
inventory database whose table and column names belong to the organisation
that owns it. Shipping those procedures would publish that schema, so the
public release only ships the *portable* path: every dialect reads the
`tblBAGrowthData` table (`db/models.py::BAGrowthInterval`), which the demo
seeds with synthetic rows.

## Contract for a site-specific procedure

Set `BAGT_USE_STORED_PROCEDURES=true` on SQL Server and the extractor
(`data/extractors.py`) will call, per model group:

```sql
EXEC sp_GetBAGrowthData_Species    @SpecCode = <int>, @SinceYear = <int>
EXEC sp_GetBAGrowthData_Cedar      @SinceYear = <int>
EXEC sp_GetBAGrowthData_ForestUnit @ForestUnit = <nvarchar>, @RequireTolerantSpecies = <bit>, @SinceYear = <int>
```

Each must return one row per plot × growth interval with exactly these columns
(the same as `tblBAGrowthData`):

```
SpecCode, ForestUnit, PlotKey, PlotName, DatasetCode,
StartYear, EndYear, GrowthInterval_Years,
BA, AGSfrac, TCR, DQ, q, Cut,
BAgrowth, RGR,
Latitude, Longitude, Easting, Northing,
TreeCount, ShrinkageCount
```

Optional extra columns `Datum` and `Zone` (integer UTM zone / datum codes)
enable datum-aware re-projection during the TCR shapefile lookup.

`@SinceYear = 0` means "everything"; otherwise only intervals with
`EndYear > @SinceYear` should be returned (used by incremental training).

Deploy with `python scripts/deploy_stored_procedures.py`, which executes every
`sql/sp_*.sql` file it finds.
