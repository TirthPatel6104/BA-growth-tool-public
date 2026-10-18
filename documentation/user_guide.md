# User Guide

The web application has seven tabs. All of them work against the synthetic
demo database that the backend creates on first start.

| Tab | What it does |
| :--- | :--- |
| **Data Entry** | Checks the API/database connection and lets you enter plot measurements for the stand-level calculators. |
| **Calculations** | Fixed-coefficient worked examples of the basal-area growth equation for entered plots. |
| **BA Growth Calculation** | Selects a model group (species or forest unit), pulls its stored coefficients and growth intervals, and projects basal area 5–25 years ahead with a 95 % band. |
| **Model Coefficients** | The fitted RGR equation per model group with R², RMSE, MAE, sample size, confidence level and sign-plausibility flags. |
| **Model Benchmarks** | Compares 17 candidate algorithms with the production equation (see below). |
| **Visualization** | Trajectory and distribution charts for the selected group. |
| **Reports** | Latest training report and exportable summaries. |

## Model Benchmarks tab

1. If no report exists the tab says so. Tick **Run (fast models)** for a
   quick run (OLS, ridge, Huber, spline GAM, random forest, boosting, mixed
   model) or untick it for all 17 algorithms, then press **Run**. Progress is
   polled every 1.5 s; a full run on the demo data takes a few minutes.
2. Pick a **model group** in the selector. The header shows how many growth
   intervals and plots were used and confirms that no plot appears in both the
   training and test folds.
3. Read the **leaderboard**. Rows are ranked by out-of-fold RMSE (ties broken
   by the 95th-percentile absolute error). *vs OLS* is the RMSE improvement
   relative to the production equation; *p (paired t)* tests whether that
   difference is distinguishable from fold-to-fold noise (`*` marks p < 0.05).
   *Wrong signs* counts coefficients whose sign contradicts forestry
   expectations (only for linear models).
4. The **RMSE bar chart** shows ± one standard deviation across folds with the
   OLS baseline as a dashed line; the **R² vs p95 scatter** places the best
   models bottom-right.
5. **Leakage check** reports how much a naive K-fold split over-states R²
   compared with the plot-grouped split, per model.
6. **Prediction intervals** compares split-conformal intervals (around the best
   model and around OLS) with native quantile-boosting intervals at the
   configured level (90 % by default). *Coverage* should be close to the
   target; narrower *mean width* is better at equal coverage.
7. **Permutation importance** shows which inputs the winning model relies on
   when predicting held-out plots.

The same report is available as JSON (`GET /api/benchmarks`) and Markdown
(`backend/artifacts/benchmarks/LEADERBOARD.md`), and the CLI
`python scripts/benchmark_models.py --help` exposes every option (feature
sets, CV strategy, fold count, interval level).
