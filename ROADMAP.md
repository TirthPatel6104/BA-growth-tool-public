# Roadmap

What exists today, what is in progress, and what would make this a stronger
end-to-end forest-analytics system. Items are ordered within each section by
expected payoff per hour of work.

## Done

- Portable data layer (SQLAlchemy 2.0 + Alembic, SQLite/PostgreSQL/SQL Server)
- Production RGR model (OLS on the seven Penner-style predictors), training
  pipeline with confidence/sign-plausibility flags, background training jobs
- **Model benchmark** (`backend/ml`): 17 candidate algorithms, plot-grouped /
  temporal / naive cross-validation, RMSE · R² · MAE · p95 absolute error ·
  bias, paired significance tests vs OLS, bootstrap CIs, split-conformal and
  quantile-boosting prediction intervals, permutation importance, leakage
  check, JSON + Markdown leaderboard, API + UI tab
- Synthetic demo generator with plot random effects, non-linear terms and
  heavy-tailed noise (no real inventory data anywhere in the repository)

## Modelling

1. **Promote a winner into production.** The predictor still uses the OLS
   coefficient table. Add a `model_backend` column to `tblRGRCoefficients`
   (or a new `tblModelArtifacts` with a serialised estimator + feature-set
   name) so `/api/bagrowth/calculate` can serve the benchmark winner, and
   gate promotion on "beats OLS at p < 0.05 on grouped CV *and* temporal CV".
2. **Hierarchical Bayesian RGR model** (PyMC / NumPyro): species-level
   coefficients drawn from a shared prior. Pools strength for data-poor groups
   (Black Ash, Red Oak) instead of the current all-or-nothing `min_samples`
   rule, and yields full posterior predictive intervals.
3. **Mixed model with random slopes** (BA and Cut varying by plot / forest
   unit) via `statsmodels` or `pymer4`; compare marginal vs conditional R².
4. **Heteroscedastic modelling**: residual sd clearly grows with BA in the
   demo data. Fit `loss="gamma"` / `"poisson"`-style GLMs on BA growth, or a
   two-headed network (mean + log-variance), and score with CRPS, not RMSE.
5. **Monotone constraints**: XGBoost/LightGBM/HistGradientBoosting all
   support `monotonic_cst`. Enforce the biological signs (BA ↓, AGS ↑, Cut ↓)
   so tree models stay plausible under extrapolation.
6. **Hyper-parameter search** with Optuna inside the grouped CV loop (nested
   CV) so tuned models are not scored on the folds they were tuned on.
7. **Multi-step projection error**: the UI iterates the one-step model for
   5–25 years. Evaluate *trajectory* error (simulate from the first
   measurement of each plot, score against its later measurements) rather than
   only one-interval RGR.
8. **Survival / mortality sub-model**: `ShrinkageCount` is already extracted;
   a zero-inflated or hurdle component for mortality would explain the
   negative-RGR tail that no regression here captures.
9. **Spatial cross-validation** (`BlockCV`-style clusters on Easting/Northing)
   as a fourth strategy; the `spatial` feature set already shows how location
   leaks plot identity.
10. **Explainability**: SHAP values for the winning model per prediction
    (with the TreeExplainer for boosted models) surfaced in the Calculator tab.

## Data & MLOps

- Experiment tracking with **MLflow** (log every benchmark run, params,
  metrics, leaderboard artifact); DVC for the synthetic datasets.
- **Model registry + versioned artifacts** (`artifacts/models/<group>/<version>/`)
  with a `/api/models/{group}/versions` endpoint and rollback.
- **Drift monitoring**: PSI / KS tests on incoming plot features vs. the
  training distribution; alert when `tblBAGrowthData` shifts.
- **Scheduled retraining** (GitHub Actions cron or Prefect flow) that runs
  the incremental pipeline, the benchmark, and opens a PR with the new
  LEADERBOARD.md.
- **Great Expectations / pandera** schema checks on `tblBAGrowthData`
  (ranges for BA, AGSfrac ∈ [0,1], TCR ∈ 1–7, EndYear > StartYear).

## Platform

- **Auth**: OAuth2 / JWT (FastAPI `Security`) with roles `viewer`,
  `analyst` (run benchmarks), `admin` (train / promote).
- **Async jobs**: replace the in-memory thread registry with Celery/RQ or
  `arq` on Redis so jobs survive restarts and can be cancelled.
- **Observability**: structured JSON logs, Prometheus metrics
  (`/metrics`), OpenTelemetry traces; request-id header already exists.
- **Caching**: Redis for the response cache instead of the process-local dict.
- **Containers**: multi-stage Dockerfiles exist; add a `docker-compose.prod.yml`
  with PostgreSQL + Redis + Nginx, health checks, and a Makefile.
- **Deploy**: Fly.io / Render / Azure Container Apps with the SQLite demo;
  Terraform module for the PostgreSQL variant.
- **Frontend**: TanStack Query for data fetching, Zod-validated API client
  generated from `documentation/openapi.json`, Vitest component tests,
  Storybook for the chart components, Playwright visual regression in CI.
- **Accessibility & i18n**: axe already runs in Playwright; move strings to
  i18next resource files instead of per-component dictionaries.

## Research questions worth a write-up

- Does a plot random intercept change *coefficients* materially, or only
  standard errors? (Compare `ols` vs `mixed_effects` coefficient tables in the
  leaderboard.)
- How much skill is lost going from grouped CV to temporal CV? That gap is
  the honest estimate of degradation when coefficients fitted on 1990s plots
  are applied to 2020s inventories.
- Are conformal intervals around a simple OLS *narrower at the same coverage*
  than native quantile-boosting intervals on small groups? (In the synthetic
  demo they usually are.)
