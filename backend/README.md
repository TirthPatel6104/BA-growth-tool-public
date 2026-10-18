# Backend Structure

FastAPI service for the Basal Area Growth Tool. The data layer runs on
**SQLAlchemy 2.0** with **Alembic** migrations and works unchanged on SQLite
(zero-config demo), PostgreSQL and, optionally, Microsoft SQL Server.

## Core Application

- `main.py` – FastAPI app (start-up hook migrates/seeds the demo DB; training and benchmark job registry)
- `app.py` – compatibility wrapper that re-exports `main:app`
- `requirements.txt` – backend dependencies
  - `requirements-ml.txt` – optional benchmark back-ends (statsmodels, XGBoost, LightGBM)
  - `requirements-mssql.txt` – optional `pyodbc` driver
- `alembic.ini`, `alembic/` – schema migrations (`alembic/versions/0001_initial_schema.py`)
- `.env.example` – every supported environment variable, documented

## Domain Modules

- `db/` – **data layer**
  - `config.py` – resolves the database URL from the environment
  - `engine.py` – engine / session factory (`get_session`, `session_scope`)
  - `models.py` – ORM models (`tblRGRCoefficients`, `tblBAGrowthData`, …)
  - `bootstrap.py` – `init_db()`: Alembic upgrade + demo seeding on start-up
  - `seed.py` – synthetic demo dataset generator (deterministic; plot random effects, non-linear terms, heavy-tailed noise)
- `api/` – FastAPI dependency providers (`deps.py`) and DTO schemas
- `models/` – production RGR model (`rgr_model.py`) and the ORM-backed `CoefficientStore`
- `training/` – training pipeline orchestration (confidence levels, sign checks, incremental retraining)
- `prediction/` – prediction services
- `data/` – extractors (portable `tblBAGrowthData`; optional site-specific stored procedures on SQL Server) and TCR enrichment
- `ml/` – **model benchmark toolkit**
  - `features.py` – design matrices (`base`, `extended`, `spatial` feature sets)
  - `registry.py` – 17 candidate algorithms (linear, additive, kernel, ensemble, neural, mixed effects) with optional back-ends
  - `evaluation.py` – metrics (R², RMSE, MAE, p95 |AE|, bias, …) and plot-grouped / temporal / naive CV
  - `conformal.py` – split-conformal and quantile-boosting prediction intervals with coverage checks
  - `stats.py` – paired t-test, Wilcoxon, bootstrap CIs, ranking
  - `benchmark.py` – orchestration → leaderboard dict; `reports.py` – JSON + Markdown writers
- `scripts/` – CLI entry points
  - `seed_demo_db.py` – migrate + (re)seed the synthetic demo database
  - `train_all_models.py` – fit the 13 production OLS models
  - `benchmark_models.py` – compare candidate models (`--all`, `--group Mh`, `--fast`, `--cv temporal`, `--features extended`, `--list-models`)
  - `deploy_stored_procedures.py`, `setup_database_tables.py`, `update_schema.py` – SQL Server helpers
  - `export_openapi.py` – writes `documentation/openapi.json`
- `sql/` – legacy T-SQL schema scripts and the contract for site-specific stored procedures (`sql/README.md`)
- `DatabaseLinker/` – backward-compatible `get_db_connection()` shim (returns a SQLAlchemy `Session`)
- `FindTCR.py` – stand-alone TCR polygon-lookup utility (CSV in, CSV out; no database schema)

## Artifacts (generated outputs, git-ignored except the synthetic demo files)

- `artifacts/models/model_coefficients.json` – fallback coefficients (synthetic demo)
- `artifacts/reports/training_results_synthetic_demo.json` – output of `train_all_models.py` on the demo seed
- `artifacts/benchmarks/benchmark_synthetic_demo.json`, `LEADERBOARD.md` – output of `benchmark_models.py --all` on the demo seed

---

## Database configuration

The URL is resolved by `db/config.py` in this order (first match wins):

| Priority | Variable(s) | Result |
| :--- | :--- | :--- |
| 1 | `DATABASE_URL` | any SQLAlchemy URL (`postgresql://…`, `mssql+pyodbc://…`, `sqlite:///…`) |
| 2 | `BAGT_SQL_CONNECTION_STRING` | raw ODBC string → SQL Server |
| 3 | `BAGT_SQL_SERVER` (+ `BAGT_SQL_DATABASE`, `BAGT_SQL_USER`, …) | legacy keyword settings → SQL Server |
| 4 | *(nothing)* | SQLite file `backend/demo_basal_growth.db` (`BAGT_SQLITE_PATH` overrides the path) |

Start-up behaviour (`db/bootstrap.py`):

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `BAGT_AUTO_MIGRATE` | `true` for SQLite, `false` otherwise | run `alembic upgrade head` when the API starts |
| `BAGT_AUTO_SEED` | `true` for SQLite, `false` otherwise | insert the synthetic demo dataset if the coefficient table is empty |
| `BAGT_SQL_ECHO` | `false` | log every SQL statement |
| `BAGT_USE_STORED_PROCEDURES` | `false` | SQL Server only: read training data from your own `sql/sp_*.sql` procedures |

### Zero-config demo (SQLite)

```bash
pip install -r requirements.txt -r requirements-ml.txt
python app.py            # creates + migrates + seeds demo_basal_growth.db on first start
```

`python scripts/seed_demo_db.py` does the same explicitly and can re-seed
(`--no-reset` keeps existing rows, `--db-path` / `--database-url` target another database).

### PostgreSQL

```bash
export DATABASE_URL=postgresql://bagt:bagt@localhost:5432/bagt
alembic upgrade head
python scripts/seed_demo_db.py       # optional: synthetic demo data
python app.py
```

### Microsoft SQL Server (optional)

```bash
pip install -r requirements.txt -r requirements-mssql.txt
export DATABASE_URL="mssql+pyodbc://user:pw@SERVER/db?driver=ODBC+Driver+17+for+SQL+Server"
alembic upgrade head
# optional: your own stored procedures (never committed – see sql/README.md)
python scripts/deploy_stored_procedures.py && export BAGT_USE_STORED_PROCEDURES=true
```

### Migrations

```bash
alembic upgrade head                              # apply
alembic current                                   # show revision
alembic revision --autogenerate -m "add column"   # after editing db/models.py
alembic downgrade -1
```

## Model benchmark

```bash
python scripts/benchmark_models.py --list-models
python scripts/benchmark_models.py --all --no-tcr                      # every group, every available model
python scripts/benchmark_models.py --group Mh --group FU_TH1 --fast    # quick
python scripts/benchmark_models.py --all --features spatial --models knn,random_forest,ols   # leakage demo
```

Or via the API: `POST /api/benchmarks/run` → poll `/models/training-status/{job_id}` →
`GET /api/benchmarks`. Methodology: `documentation/model_benchmarking.md`.

## Tests

```bash
pytest tests/backend/ -v
```

`tests/backend/conftest.py` creates a throw-away SQLite database with the real
Alembic migrations and the synthetic seed – no database server is needed.
Set `BAGT_TEST_DATABASE_URL` to run the same suite against PostgreSQL.
`tests/backend/test_ml_benchmark.py` covers the benchmark toolkit end to end,
including the API job.
