# Basal Area Growth Tool

**Forest growth-and-yield analytics: fit, benchmark and serve stand-level basal-area growth models.**

![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)
![React 19](https://img.shields.io/badge/frontend-React_19_+_Vite_7-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/typescript-5.9-3178C6?logo=typescript&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn_·_XGBoost_·_LightGBM_·_statsmodels-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-89_pytest_·_7_Playwright-brightgreen)
![License](https://img.shields.io/badge/license-source--available-lightgrey)

A full-stack platform that pairs the classical relative-growth-rate (RGR)
regression used in operational growth-and-yield work with a leakage-aware
model-comparison toolkit. It answers a question most growth tools skip:
**is the production equation actually the best model for this data?**

> [!IMPORTANT]
> **This is the public, synthetic-data edition of a private project.**
> Everything in this repository runs on a dataset generated from a fixed
> random seed by `backend/db/seed.py`. There are **no real inventory plots,
> coordinates, fitted coefficients, benchmark results, hostnames, credentials
> or organisation-specific database schema** anywhere in the tree or its
> history. Site-specific stored procedures are deliberately excluded; the
> contract they must satisfy is documented in `backend/sql/README.md`.

---

## Screenshots

| Home · Data Entry | BA Growth Calculator |
| :---: | :---: |
| ![Home tab](docs/screenshots/home.png) | ![BA growth calculator](docs/screenshots/ba-growth-calculator.png) |

<details>
<summary><strong>Model Benchmarks tab</strong> (light, dark and mobile)</summary>

| Light | Dark |
| :---: | :---: |
| ![Model benchmarks, light theme](docs/screenshots/model-benchmarks.png) | ![Model benchmarks, dark theme](docs/screenshots/model-benchmarks-dark.png) |

<p align="center"><img src="docs/screenshots/model-benchmarks-mobile.png" alt="Model benchmarks on a phone-width viewport" width="260"></p>

</details>

---

## Contents

- [What it does](#what-it-does)
- [The growth model](#the-growth-model)
- [The model benchmark](#the-model-benchmark)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Project layout](#project-layout)
- [Testing and CI](#testing-and-ci)
- [What is and is not in this public edition](#what-is-and-is-not-in-this-public-edition)
- [Roadmap, contributing, security](#roadmap-contributing-security)
- [License](#license)

---

## What it does

- **Growth calculator.** Annual and periodic basal-area increment from stand
  density, competition and site parameters, projected 5 to 25 years ahead
  with a 95 % band.
- **13 model groups.** Eight species models (hard maple, red oak, yellow
  birch, hemlock, black ash, red maple, cedar, beech) and five stand-level
  forest-unit models, each with its own stored coefficients and quality
  metrics.
- **Model benchmark.** Seventeen candidate algorithms scored with
  plot-grouped cross-validation, paired significance tests against the
  production equation, prediction-interval calibration, permutation
  importance and an explicit leakage check. Runs from the UI, the API or the
  command line.
- **Reporting.** R², RMSE, MAE, p95 absolute error, bias, sample sizes,
  confidence levels and coefficient-sign plausibility, exportable as JSON
  and Markdown leaderboards.
- **Portable data layer.** SQLAlchemy 2.0 and Alembic. Zero-config SQLite
  for the demo, PostgreSQL through `DATABASE_URL`, optional SQL Server.
- **Background jobs.** Training and benchmarking return `202 Accepted` and a
  job id you can poll.
- **Bilingual UI.** English and French, light and dark themes, responsive
  down to phone width, with an automated accessibility scan in the E2E suite.

---

## The growth model

The production engine estimates the relative growth rate of stand basal area
over a remeasurement interval with a fixed-form multiple linear regression:

$$RGR = \beta_0 + \beta_1 BA + \beta_2 BA^2 + \beta_3 AGS_{frac} + \beta_4 TCR + \beta_5 DQ + \beta_6 q + \beta_7 Cut$$

| Predictor | Meaning | Units | Expected sign |
| :--- | :--- | :---: | :---: |
| $BA$ | Initial stand basal area | m²/ha | − |
| $BA^2$ | Density-saturation term | (m²/ha)² | |
| $AGS_{frac}$ | Proportion of acceptable growing stock | 0–1 | + |
| $TCR$ | Timber capability rating of the site | 1–7 | + |
| $DQ$ | Quadratic mean diameter at breast height | cm | − |
| $q$ | Diameter-distribution quotient | – | + |
| $Cut$ | Partial-harvest flag | 0/1 | − |

Projection iterates annually: $BA_{t+1} = BA_t + RGR \cdot BA_t$.

Every fitted linear model is checked against the expected biological sign of
each coefficient and flagged when it disagrees. The expected signs follow the
published tolerant-hardwood growth literature; no proprietary coefficients
are included.

---

## The model benchmark

`backend/ml` compares the production OLS equation with sixteen alternatives
on every model group and produces an out-of-sample leaderboard.

| Family | Models |
| :--- | :--- |
| Linear | OLS (baseline), ridge, lasso, elastic net, Huber, Bayesian ridge |
| Additive | Cubic-spline GAM-style, degree-2 polynomial + ridge |
| Kernel | k-nearest neighbours, RBF support-vector regression |
| Ensemble | Random forest, extra-trees, histogram gradient boosting, XGBoost, LightGBM |
| Neural | Two-layer MLP |
| Mixed | Linear mixed model with a random intercept per plot (statsmodels) |

**How models are judged**

- **Plot-grouped K-fold** (default). A plot's remeasurements never straddle
  train and test, so scores reflect prediction on unmeasured plots.
- **Temporal forward-chaining.** Fit on earlier measurement years, test on
  later ones.
- **Naive K-fold as a leakage control.** The gap to grouped CV is reported
  per model. With the `spatial` feature set, where coordinates identify
  plots, naive CV over-states R² by a wide margin.
- **Metrics.** R², adjusted R², RMSE, MAE, median / p90 / p95 / p99 absolute
  error, bias, MAPE, bootstrap confidence intervals.
- **Statistics versus OLS.** Paired t-test on per-fold RMSE and Wilcoxon on
  per-sample absolute errors.
- **Prediction intervals.** Split-conformal (any model, finite-sample
  guarantee) versus native quantile gradient boosting, scored by empirical
  coverage and width.
- **Explainability.** Permutation importance of the winning model on
  held-out plots, plus sign plausibility for every linear model.

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt
python scripts/benchmark_models.py --all                        # 13 groups × 17 models
python scripts/benchmark_models.py --group Mh --fast --features extended --cv temporal
python scripts/benchmark_models.py --list-models
```

Results land in `backend/artifacts/benchmarks/` as timestamped JSON plus a
`LEADERBOARD.md`, and are served to the **Model Benchmarks** tab by
`GET /api/benchmarks`. The committed `benchmark_synthetic_demo.json` was
produced on the synthetic seed. The methodology and a discussion of those
results are in [documentation/model_benchmarking.md](documentation/model_benchmarking.md).

---

## Architecture

```mermaid
graph TD
    A[React 19 + TypeScript frontend] -->|REST| B[FastAPI backend]
    B --> C[Predictor engine]
    B --> D[Coefficient store · SQLAlchemy ORM]
    B --> H[Model benchmark · backend/ml]
    C --> E[RGR linear model]
    H --> I[Model registry: linear · additive · kernel · ensemble · neural · mixed]
    H --> J[Grouped / temporal / naive CV · conformal intervals · importance]
    D --> F[(SQLite / PostgreSQL / SQL Server via Alembic)]
    C --> G[(tblBAGrowthData · synthetic intervals)]
    H --> G
    H --> K[artifacts/benchmarks · JSON + Markdown leaderboard]
```

**Frontend.** React 19, TypeScript, Vite 7, Tailwind CSS 4, Radix UI,
Recharts, Lucide. Playwright and axe for end-to-end and accessibility tests.

**Backend.** Python 3.11, FastAPI, Pydantic 2, SQLAlchemy 2.0, Alembic,
pandas, NumPy, SciPy, scikit-learn. Optional statsmodels, XGBoost and
LightGBM for the benchmark, and GeoPandas / Shapely / pyproj for the optional
TCR polygon lookup.

C4 diagrams, sequence flows and the ER model are in
[documentation/system_design.md](documentation/system_design.md).

---

## Quick start

**Prerequisites:** Python 3.11+, Node.js 20+, npm 9+.

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
python app.py
```

On first start the API creates `backend/demo_basal_growth.db`, applies the
Alembic migrations and seeds the synthetic dataset (13 model groups, roughly
2 000 growth intervals). Interactive API docs are at `http://localhost:8000/docs`.

Optional follow-ups:

```bash
python scripts/seed_demo_db.py                       # rebuild the demo database
python scripts/train_all_models.py --all --no-tcr    # refit the 13 OLS models
python scripts/benchmark_models.py --all --no-tcr    # full model comparison
```

### Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

### Docker

```bash
docker compose up --build
```

The backend image seeds its own SQLite demo database at build time. Point it
at PostgreSQL instead by uncommenting `DATABASE_URL` in `docker-compose.yml`.

---

## Configuration

All settings are environment variables read from `backend/.env` (see
`backend/.env.example` for the full, commented list). With nothing set, the
API runs on a local SQLite file.

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `DATABASE_URL` | Any SQLAlchemy URL (PostgreSQL, SQL Server, …) | SQLite demo file |
| `BAGT_SQLITE_PATH` | Location of the SQLite demo file | `./demo_basal_growth.db` |
| `BAGT_AUTO_MIGRATE` | Run `alembic upgrade head` on start-up | `true` for SQLite |
| `BAGT_AUTO_SEED` | Seed the synthetic dataset when the coefficient table is empty | `true` for SQLite |
| `BAGT_CORS_ORIGINS` | Extra allowed origins, comma-separated | Vite dev origins |
| `BAGT_USE_STORED_PROCEDURES` | Read training data from site-specific SQL Server procedures | `false` |
| `BAGT_TCR_SHAPEFILE` | Polygon shapefile for the optional TCR lookup | unset |
| `VITE_API_BASE_URL` | Frontend: backend base URL | `http://localhost:8000` |

PostgreSQL:

```bash
export DATABASE_URL=postgresql://bagt:bagt@localhost:5432/bagt
alembic upgrade head && python scripts/seed_demo_db.py && python app.py
```

SQL Server (optional): `pip install -r requirements-mssql.txt`, set a
`mssql+pyodbc://` URL or the `BAGT_SQL_*` variables, run `alembic upgrade head`.
Training data is read from the portable `tblBAGrowthData` table unless you
provide your own stored procedures. Their result-set contract is in
`backend/sql/README.md`.

---

## API

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness and database status |
| `GET` | `/api/species-list` | Model groups (species and forest units) |
| `GET` | `/api/model-coefficients?model_group=` | Fitted coefficients and quality metrics |
| `GET` | `/api/bagrowth/summary-stats?species_group=` | Dataset summary for a group |
| `GET` | `/api/bagrowth/calculate?species_group=&current_year=` | Per-plot predictions and yearly projection |
| `POST` | `/models/train` | Start a training job (`202`) |
| `GET` | `/models/training-status/{job_id}` | Poll a training or benchmark job |
| `POST` | `/models/predict`, `/models/predict/batch`, `/models/auto-predict` | Direct predictions |
| `GET` | `/api/benchmarks/models` | Algorithm registry, CV strategies, feature sets |
| `POST` | `/api/benchmarks/run` | Start a benchmark job (`202`) |
| `GET` | `/api/benchmarks` | Latest leaderboard, all groups |
| `GET` | `/api/benchmarks/{model_group}` | One group's section of the latest report |

The full OpenAPI schema is committed at `documentation/openapi.json` and
regenerated with `python scripts/export_openapi.py`.

---

## Project layout

```
.
├── backend/
│   ├── main.py                 FastAPI app, start-up migrate/seed, job registry
│   ├── api/                    Route handlers and Pydantic schemas
│   ├── db/                     SQLAlchemy models, engine, config, synthetic seed
│   ├── alembic/                Schema migrations
│   ├── data/                   Training-data extractors (table or stored-procedure path)
│   ├── models/                 RGR linear model and coefficient store
│   ├── prediction/             Predictor engine
│   ├── training/               Training pipeline with quality and sign checks
│   ├── ml/                     Benchmark: registry, features, CV, stats, conformal, reports
│   ├── scripts/                CLI entry points (seed, train, benchmark, export OpenAPI)
│   ├── sql/                    Legacy T-SQL schema plus the stored-procedure contract
│   ├── data/synthetic_demo_growth_data.csv   Synthetic sample export
│   └── artifacts/              Synthetic demo coefficients, report and leaderboard
├── frontend/
│   ├── src/components/         Calculator, coefficients, benchmarks, reports, charts
│   ├── src/hooks/              API hooks
│   └── tests/                  Playwright end-to-end and accessibility tests
├── tests/backend/              pytest suite (offline, scratch SQLite via Alembic)
├── documentation/              Architecture, system design, benchmarking, user guide, OpenAPI
├── docs/screenshots/           Images used in this README
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Testing and CI

```bash
pytest tests/backend/ -v --cov=backend        # 89 tests, offline
cd frontend && npm run lint && npm run build  # ESLint + TypeScript
cd frontend && npx playwright test            # 7 E2E tests incl. an axe accessibility scan
```

GitHub Actions runs three jobs on every push and pull request: flake8 and
mypy, the unit tests with coverage plus an Alembic migration sanity check and
a benchmark smoke run on the synthetic database, and the frontend lint and
build.

---

## What is and is not in this public edition

| Included | Excluded |
| :--- | :--- |
| Complete application code, frontend and backend | Real inventory plot records and coordinates |
| Deterministic synthetic dataset generator | Coefficients fitted to real data |
| Coefficients and benchmark report fitted **to the synthetic data** | Benchmark results on real data |
| Portable `tblBAGrowthData` schema via Alembic | Site-specific stored procedures and the inventory schema behind them |
| Documented stored-procedure contract | Hostnames, connection strings, credentials |

Model artifacts and reports are git-ignored except the clearly labelled
synthetic demo files, and all `.env` files are ignored.

---

## Roadmap, contributing, security

- [ROADMAP.md](ROADMAP.md): hierarchical Bayesian pooling across species,
  monotone-constrained boosting, heteroscedastic / CRPS evaluation,
  multi-step trajectory error, experiment tracking, model promotion gates,
  authentication, Redis-backed jobs, observability.
- [CONTRIBUTING.md](CONTRIBUTING.md): workflow and coding standards.
- [SECURITY.md](SECURITY.md): please report vulnerabilities privately.

---

## License

Source-available, all rights reserved. You may view and inspect the code for
evaluation; copying, modification or redistribution requires written
permission. See [LICENSE](LICENSE).
