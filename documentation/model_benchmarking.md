# Model Benchmarking: Methodology and Results

This document explains how `backend/ml` decides whether any candidate
algorithm beats the production relative-growth-rate (RGR) equation, why the
evaluation is designed the way it is, and what the synthetic demo run shows.
Results on real inventory data are intentionally not published here.

## 1. The question

The production model is a fixed-form ordinary least squares fit:

```
RGR = β0 + β1·BA + β2·BA² + β3·AGSfrac + β4·TCR + β5·DQ + β6·q + β7·Cut
```

It is transparent, cheap, and its coefficients can be checked against
forestry expectations. The benchmark asks three things:

1. Does a more flexible learner predict *unmeasured plots* better?
2. If so, is the improvement larger than fold-to-fold noise?
3. How wide are honest prediction intervals, and are they calibrated?

## 2. Data structure dictates the validation design

Permanent sample plots are **remeasured**: each plot contributes several
growth intervals, and intervals from the same plot share unobserved site
quality. Three consequences shape the design:

| Issue | Design response |
| :--- | :--- |
| Rows are not independent | **Plot-grouped K-fold** (`GroupKFold` on `PlotKey`) is the primary split. A plot's intervals are either all in training or all in test. |
| Coefficients fitted on old measurements are applied to new inventories | **Temporal forward-chaining** on `EndYear` is reported as a secondary split. |
| Naive K-fold silently over-states skill | It is still run, as a **leakage control**, and the R² inflation is reported per model. |

The inflation is small with the seven equation features alone (they do not
identify a plot) and large with the `spatial` feature set (coordinates do),
which the unit test `test_leakage_inflation_is_positive_when_features_identify_plots`
demonstrates. That is the practical lesson: adding location-like features
*without* grouped CV produces leaderboards that look better than they are.

## 3. Candidate models

| Family | Models | Why included |
| :--- | :--- | :--- |
| Linear | `ols` (baseline), `ridge`, `lasso`, `elastic_net`, `huber`, `bayesian_ridge` | Same functional form; tests whether shrinkage or robustness helps on small groups |
| Additive | `spline_gam`, `poly2_ridge` | Smooth non-linearities and pairwise interactions without abandoning interpretability |
| Kernel | `knn`, `svr` | Local / kernel estimates of the growth surface |
| Ensemble | `random_forest`, `extra_trees`, `hist_gbm`, `xgboost`, `lightgbm` | Flexible interaction learners; the usual tabular-data winners |
| Neural | `mlp` | Sanity check that a small network is not a free win on 100–200 rows |
| Mixed | `mixed_effects` | Random intercept per plot: the statistically appropriate model for repeated measures |

Scale-sensitive learners are wrapped in `StandardScaler`; the MLP and SVR also
standardise the target (RGR ≈ 0.03 ± 0.01) so no model is handicapped by
loss scale. Hyper-parameters are fixed, defensible defaults rather than
tuned, so the comparison is between *model classes*; nested tuning is on the
roadmap.

## 4. Metrics

All metrics are computed on out-of-fold predictions and averaged over folds
(standard deviation reported):

- **RMSE** (primary ranking key), **MAE**, **median absolute error**
- **p90 / p95 / p99 absolute error** – tail risk; p95 is the tie-breaker because a growth model that is occasionally very wrong is operationally worse than one that is uniformly slightly wrong
- **R²** and adjusted R², **bias** (mean residual), MAPE (guarded for near-zero RGR)
- **Bootstrap 95 % CIs** on RMSE and p95 from the pooled out-of-fold residuals

## 5. Is a difference real?

For every model versus `ols`, on identical folds:

- **Paired t-test** on per-fold RMSE (few pairs, parametric)
- **Wilcoxon signed-rank** on per-sample absolute errors (many pairs, non-parametric)

A model is only interesting when its RMSE improvement is positive *and* the
paired test is significant; in practice on 120–170 rows most differences
between the linear variants are noise, which is itself a useful conclusion.

## 6. Prediction intervals

- **Split conformal** around any point model: plots are split into
  fitting and calibration sets; the (1-α) quantile of calibration absolute
  residuals (finite-sample corrected) becomes a symmetric half-width. Valid
  under exchangeability, and the group-aware split keeps that assumption
  honest.
- **Quantile gradient boosting**: two `HistGradientBoostingRegressor`s at
  α/2 and 1-α/2 give input-dependent widths (e.g. wider after a cut) but no
  finite-sample guarantee.

Both are scored on held-out plots by empirical **coverage** (target 90 %)
and **mean width**.

## 7. Plausibility and explainability

- Linear models report coefficients on the original scale and are checked
  against expected signs (BA −, AGSfrac +, TCR +, DQ −, q +, Cut −).
- The winning model's **permutation importance** (increase in RMSE when a
  feature is shuffled on held-out plots) is reported.

## 8. Running it

```bash
cd backend
pip install -r requirements.txt -r requirements-ml.txt
python scripts/benchmark_models.py --all --no-tcr                # everything
python scripts/benchmark_models.py --group Mh --fast             # quick
python scripts/benchmark_models.py --all --cv temporal           # rank by forward-chaining CV
python scripts/benchmark_models.py --all --features extended     # engineered features
```

API: `POST /api/benchmarks/run` → poll `/models/training-status/{job_id}` →
`GET /api/benchmarks`. UI: the **Model Benchmarks** tab.

## 9. Results on the synthetic demo data

The committed `backend/artifacts/benchmarks/benchmark_synthetic_demo.json`
was produced by `benchmark_models.py --all --no-tcr` on the deterministic
synthetic seed (`db/seed.py`, 60 plots per species group, 48 per forest unit,
plot random intercepts, a saturating density term, a cut × stocking
interaction and heteroscedastic, occasionally heavy-tailed noise).

Run: 2026-10-12T00:02:46 UTC, 13 model groups, 17 algorithms, 377 s on a laptop CPU.

### 9.1 Overall ranking (mean rank across the 13 groups, plot-grouped 5-fold CV)

| Model | Mean rank | Groups won | Median RMSE change vs OLS |
| :--- | ---: | ---: | ---: |
| bayesian_ridge | 4.15 | 3 | -0.0% |
| ridge | 4.54 | 1 | +0.1% |
| ols | 4.92 | 1 | baseline |
| huber | 5.00 | 2 | +0.1% |
| mixed_effects | 5.38 | 1 | -0.0% |
| elastic_net | 6.31 | 0 | -0.6% |
| lasso | 6.85 | 0 | -0.6% |
| spline_gam | 7.85 | 2 | -1.4% |
| svr | 9.08 | 2 | -4.0% |
| knn | 9.54 | 0 | -2.3% |
| extra_trees | 10.62 | 0 | -3.2% |
| poly2_ridge | 10.92 | 1 | -3.6% |
| random_forest | 11.15 | 0 | -4.2% |
| mlp | 11.85 | 0 | -6.0% |
| hist_gbm | 14.54 | 0 | -8.3% |
| xgboost | 15.08 | 0 | -10.2% |
| lightgbm | 15.23 | 0 | -9.6% |

### 9.2 Per-group winners

| Group | n | plots | Best model | Best RMSE | OLS RMSE | Best R² | OLS R² | Temporal OLS R² | p (best vs OLS) | Significant? |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Mh | 164 | 60 | svr | 0.00846 | 0.00867 | 0.303 | 0.281 | 0.207 | 0.476 | no |
| Or | 159 | 60 | poly2_ridge | 0.00952 | 0.01001 | 0.219 | 0.142 | 0.308 | 0.049 | yes |
| By | 163 | 60 | ridge | 0.00840 | 0.00848 | 0.140 | 0.120 | -0.067 | 0.349 | no |
| He | 158 | 60 | bayesian_ridge | 0.00905 | 0.00917 | 0.247 | 0.225 | 0.176 | 0.411 | no |
| Ab | 162 | 60 | spline_gam | 0.00973 | 0.00987 | 0.303 | 0.244 | 0.399 | 0.741 | no |
| Mr | 162 | 60 | mixed_effects | 0.00826 | 0.00834 | 0.421 | 0.411 | 0.317 | 0.195 | no |
| Cedar | 157 | 60 | spline_gam | 0.00838 | 0.00871 | 0.302 | 0.217 | 0.334 | 0.486 | no |
| Be | 162 | 60 | ols | 0.00931 | 0.00931 | 0.141 | 0.141 | 0.099 | – | no |
| FU_TH1 | 130 | 48 | huber | 0.00928 | 0.00945 | 0.394 | 0.371 | 0.273 | 0.153 | no |
| FU_HDSL | 133 | 48 | bayesian_ridge | 0.01058 | 0.01115 | 0.220 | 0.112 | 0.253 | 0.429 | no |
| FU_HDSL2 | 124 | 48 | bayesian_ridge | 0.00921 | 0.00928 | 0.173 | 0.156 | -0.258 | 0.592 | no |
| FU_CE1 | 134 | 48 | svr | 0.00813 | 0.00838 | 0.288 | 0.249 | 0.221 | 0.341 | no |
| FU_HE1 | 120 | 48 | huber | 0.00802 | 0.00814 | 0.186 | 0.161 | 0.136 | 0.172 | no |

In 1 of 13 groups the winner beats OLS at p < 0.05 on the paired fold test.

### 9.3 Leakage check (naive K-fold minus plot-grouped R², median across models)

Median inflation per group ranges from -0.076 to +0.168 (overall median +0.033). Positive values mean a naive split over-states skill.

### 9.4 Prediction intervals (90 % target, held-out plots)

| Method | Mean coverage | Mean width | Groups calibrated (±5 pts) |
| :--- | ---: | ---: | ---: |
| split_conformal_best | 0.914 | 0.03264 | 12/13 |
| split_conformal_ols | 0.906 | 0.03313 | 12/12 |
| quantile_gbm | 0.678 | 0.01980 | 0/13 |

### 9.5 Permutation importance of the winning models

BA (0.00122), TCR (0.00080), BA2 (0.00070), AGSfrac (0.00039), Cut (0.00035), DQ (0.00025), q (0.00007)

### 9.6 Sign plausibility of linear models

ols: 1 wrong signs over 13 groups, huber: 2 wrong signs over 13 groups, ridge: 0 wrong signs over 13 groups, elastic_net: 0 wrong signs over 13 groups, lasso: 0 wrong signs over 13 groups, bayesian_ridge: 0 wrong signs over 13 groups, mixed_effects: 2 wrong signs over 13 groups

### 9.7 What the demo run says

- **The production equation is hard to beat on this data.** Five of the top six models are the same linear form with different shrinkage or loss; the leader (`bayesian_ridge`) changes RMSE by well under 1 % relative to OLS, and only one group shows a difference that survives the paired test (Red Oak, degree-2 polynomial, p = 0.049). With ~130–165 intervals per group, a 17-model leaderboard will always crown *something*, so the significance column matters more than the rank.
- **Gradient boosting and the MLP lose at this sample size.** `hist_gbm`, `xgboost` and `lightgbm` are 8–10 % worse than OLS on held-out plots; they fit the plot-level noise. The random forest and extra-trees are closer but still behind. This is the expected regime for small, noisy ecological tables and is the reason the roadmap puts monotone constraints and nested tuning ahead of more flexible architectures.
- **Where non-linearity exists, the additive models find it.** `spline_gam` wins Black Ash and Cedar and `poly2_ridge` wins Red Oak, the groups where the synthetic saturating-density and cut × stocking terms are strongest relative to noise. A GAM-style model is the natural next step for the production equation because it keeps per-variable interpretability.
- **Temporal validation is the stricter test.** OLS R² under forward-chaining is lower than under grouped CV for most groups and negative for Yellow Birch and HDSL2: coefficients fitted on the earlier measurement years do not transfer cleanly to later ones. Any coefficient update process should report this number, not just the grouped one.
- **Conformal intervals are calibrated; quantile boosting is not.** Split-conformal bands hit 90–91 % coverage in 12 of 13 groups. The quantile boosting bands are narrower but cover only ~68 %; on this little data they are optimistic, so the deployment default should be conformal intervals around whichever point model is promoted.
- **Leakage is modest with the equation features and material once plots are identifiable.** The median naive-vs-grouped R² gap is +0.03 here (up to +0.17 in individual groups); the unit tests show it exceeding +0.05 as soon as coordinates are added. Plot-grouped CV is therefore the only acceptable default for this data.
- **Sign plausibility holds.** Ridge, lasso, elastic net and Bayesian ridge have zero implausible coefficient signs across 13 groups; OLS has one. Shrinkage buys sign stability at no accuracy cost, which is a reasonable argument for `ridge`/`bayesian_ridge` as the production fit.

These conclusions are about the *synthetic* generator. Re-run `benchmark_models.py --all` against a real extract to obtain the real leaderboard; keep that report out of version control (`backend/artifacts/` is git-ignored except the demo files).
