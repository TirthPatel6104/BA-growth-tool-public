"""
tests/backend/test_ml_benchmark.py
==================================
Unit tests for the model-benchmarking toolkit (``backend/ml``).

Pure-numpy fixtures run offline; the API tests use the seeded SQLite scratch
database from ``conftest.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml import conformal, stats  # noqa: E402
from ml.benchmark import BenchmarkConfig, benchmark_all, benchmark_group  # noqa: E402
from ml.evaluation import (  # noqa: E402
    compute_metrics,
    cross_validate,
    group_overlap,
    make_splits,
    sign_check,
)
from ml.features import BASE_FEATURES, EXTENDED_FEATURES, build_dataset  # noqa: E402
from ml.registry import (  # noqa: E402
    MODEL_SPECS,
    MixedEffectsRegressor,
    available_models,
    linear_coefficients,
    list_models,
)
from ml.reports import to_markdown, write_json, load_json  # noqa: E402

pytestmark = pytest.mark.unit

TRUE_COEF = {"intercept": 0.05, "BA": -0.0015, "BA2": 1e-6, "AGSfrac": 0.012,
             "TCR": 0.002, "DQ": -0.0002, "q": 0.003, "Cut": -0.008}


def _repeated_measures(n_plots: int = 60, per_plot: int = 4, seed: int = 7, plot_sd: float = 0.006) -> pd.DataFrame:
    """Synthetic plots with a random intercept – the structure grouped CV must respect."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_plots):
        u = rng.normal(0, plot_sd)
        ba = rng.uniform(10, 40)
        year = int(rng.integers(1990, 2000))
        lat, lon = rng.uniform(44.0, 46.5), rng.uniform(-81.0, -77.5)
        for _ in range(per_plot):
            ags, tcr, dq, q, cut = rng.uniform(0.3, 0.9), int(rng.integers(1, 8)), rng.uniform(12, 35), rng.uniform(1.1, 2.3), int(rng.random() < 0.25)
            rgr = (TRUE_COEF["intercept"] + TRUE_COEF["BA"] * ba + TRUE_COEF["BA2"] * ba * ba + TRUE_COEF["AGSfrac"] * ags
                   + TRUE_COEF["TCR"] * tcr + TRUE_COEF["DQ"] * dq + TRUE_COEF["q"] * q + TRUE_COEF["Cut"] * cut
                   + u + rng.normal(0, 0.003))
            interval = int(rng.integers(4, 8))
            rows.append({"PlotKey": 1000 + p, "StartYear": year, "EndYear": year + interval,
                         "GrowthInterval_Years": interval, "BA": ba, "AGSfrac": ags, "TCR": tcr,
                         "DQ": dq, "q": q, "Cut": cut, "RGR": rgr, "BAgrowth": rgr * ba,
                         "Latitude": lat, "Longitude": lon})
            year += interval
            ba = ba * (1 + rgr * interval)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _repeated_measures()


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_base_dataset_shape(self, frame):
        ds = build_dataset(frame, "base")
        assert ds.feature_names == BASE_FEATURES
        assert ds.n_samples == len(frame)
        assert ds.n_groups == 60
        assert np.allclose(ds.X["BA2"], ds.X["BA"] ** 2)

    def test_extended_dataset_adds_engineered_columns(self, frame):
        ds = build_dataset(frame, "extended")
        assert ds.feature_names == EXTENDED_FEATURES
        assert (ds.X["Interval"] > 0).all()

    def test_defaults_and_drops_match_production_rules(self, frame):
        messy = frame.copy()
        messy.loc[0, "TCR"] = np.nan      # defaulted to 5
        messy.loc[1, "Cut"] = np.nan      # defaulted to 0
        messy.loc[2, "AGSfrac"] = np.nan  # row dropped
        ds = build_dataset(messy)
        assert ds.n_samples == len(frame) - 1
        assert ds.X.loc[0, "TCR"] == 5
        assert ds.X.loc[1, "Cut"] == 0

    def test_spatial_dataset_requires_coordinates(self, frame):
        ds = build_dataset(frame, "spatial")
        assert ds.feature_names[-2:] == ["Latitude", "Longitude"]
        with pytest.raises(ValueError, match="Latitude"):
            build_dataset(frame.drop(columns=["Latitude"]), "spatial")

    def test_unknown_feature_set(self, frame):
        with pytest.raises(ValueError):
            build_dataset(frame, "nope")


# ---------------------------------------------------------------------------
# metrics & splits
# ---------------------------------------------------------------------------

class TestEvaluation:

    def test_metrics_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y, n_features=2)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == 0 and m["p95_ae"] == 0 and m["bias"] == 0

    def test_metrics_known_values(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        pred = np.array([1.0, -1.0, 2.0, -2.0])
        m = compute_metrics(y, pred)
        assert m["rmse"] == pytest.approx(np.sqrt(2.5))
        assert m["mae"] == pytest.approx(1.5)
        assert m["bias"] == pytest.approx(0.0)
        assert m["max_ae"] == 2.0

    def test_group_splits_never_leak_plots(self, frame):
        ds = build_dataset(frame)
        splits = make_splits(ds, "group", 5, seed=1)
        assert len(splits) == 5
        assert group_overlap(ds, splits) == 0
        covered = np.concatenate([t for _, t in splits])
        assert sorted(covered) == list(range(ds.n_samples))

    def test_kfold_splits_do_leak_plots(self, frame):
        ds = build_dataset(frame)
        assert group_overlap(ds, make_splits(ds, "kfold", 5, seed=1)) > 0

    def test_temporal_splits_are_forward_only(self, frame):
        ds = build_dataset(frame)
        for train_idx, test_idx in make_splits(ds, "temporal", 4, seed=1):
            assert ds.years.iloc[train_idx].max() < ds.years.iloc[test_idx].min()

    def test_cross_validate_ols_recovers_signal(self, frame):
        ds = build_dataset(frame)
        res = cross_validate(MODEL_SPECS["ols"], ds, make_splits(ds, "group", 5, 1), "group", 1)
        assert len(res.folds) == 5
        assert res.metric_summary()["mean"]["r2"] > 0.7
        assert not np.isnan(res.oof_pred).any()

    def test_sign_check(self):
        good = {"BA": -1, "AGSfrac": 1, "TCR": 1, "DQ": -1, "q": 1, "Cut": -1}
        assert sign_check(good)["wrong_sign_count"] == 0
        bad = dict(good, BA=+1, q=-1)
        assert sorted(sign_check(bad)["wrong_signs"]) == ["BA(+)", "q(-)"]
        assert sign_check(None) is None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_catalogue_lists_every_spec(self):
        names = {m["name"] for m in list_models()}
        assert names == set(MODEL_SPECS)
        assert all({"family", "description", "available"} <= set(m) for m in list_models())

    def test_available_models_skips_missing_backends(self, monkeypatch):
        monkeypatch.setattr("ml.registry._has", lambda module: False)
        names = [s.name for s in available_models(["ols", "xgboost", "mixed_effects"])]
        assert names == ["ols"]

    def test_unknown_model_name(self):
        with pytest.raises(ValueError):
            available_models(["not_a_model"])

    def test_linear_coefficients_unscale_pipeline(self, frame):
        ds = build_dataset(frame)
        X, y = ds.X.to_numpy(), ds.y.to_numpy()
        ridge = MODEL_SPECS["ridge"].build(ds.feature_names, 0).fit(X, y)
        ols = MODEL_SPECS["ols"].build(ds.feature_names, 0).fit(X, y)
        c_ridge = linear_coefficients(ridge, ds.feature_names)
        c_ols = linear_coefficients(ols, ds.feature_names)
        assert set(c_ridge) == {"intercept", *ds.feature_names}
        # tiny alpha ⇒ ridge ≈ OLS on the original scale
        assert c_ridge["Cut"] == pytest.approx(c_ols["Cut"], rel=0.2)
        # predictions from the unscaled coefficients reproduce the pipeline
        manual = c_ridge["intercept"] + X @ np.array([c_ridge[f] for f in ds.feature_names])
        assert np.allclose(manual, ridge.predict(X), atol=1e-8)

    @pytest.mark.skipif(not MODEL_SPECS["mixed_effects"].available, reason="statsmodels not installed")
    def test_mixed_effects_estimates_plot_variance(self, frame):
        ds = build_dataset(frame)
        est = MixedEffectsRegressor().fit(ds.X.to_numpy(), ds.y.to_numpy(), groups=ds.groups.to_numpy())
        assert est.random_effect_sd_ > 0.002          # true plot sd is 0.006
        coefs = linear_coefficients(est, ds.feature_names)
        assert coefs["Cut"] < 0 and coefs["AGSfrac"] > 0
        assert est.predict(ds.X.to_numpy()[:3]).shape == (3,)

    def test_mixed_effects_without_groups_falls_back_to_ols(self, frame):
        ds = build_dataset(frame)
        est = MixedEffectsRegressor().fit(ds.X.to_numpy(), ds.y.to_numpy())
        assert est.random_effect_sd_ == 0.0


# ---------------------------------------------------------------------------
# intervals & stats
# ---------------------------------------------------------------------------

class TestIntervalsAndStats:

    def test_split_conformal_coverage_near_target(self, frame):
        ds = build_dataset(frame)
        out = conformal.split_conformal_evaluate(MODEL_SPECS["ols"], ds, alpha=0.10, n_splits=5, seed=3)
        assert out["n_folds"] == 5
        assert 0.80 <= out["coverage"] <= 1.0
        assert out["mean_width"] > 0

    def test_quantile_gbm_returns_valid_intervals(self, frame):
        ds = build_dataset(frame)
        out = conformal.quantile_gbm_evaluate(ds, alpha=0.10, n_splits=3, seed=3)
        assert 0.0 <= out["coverage"] <= 1.0 and out["mean_width"] >= 0

    def test_conformal_half_width_positive(self, frame):
        ds = build_dataset(frame)
        assert conformal.conformal_half_width(MODEL_SPECS["ols"], ds) > 0

    def test_paired_tests_identical_inputs(self):
        assert stats.paired_fold_test([1, 2, 3], [1, 2, 3])["p_value"] == 1.0
        y = np.zeros(50)
        oof = np.ones(50)
        assert stats.wilcoxon_oof(oof, oof, y)["p_value"] == 1.0

    def test_bootstrap_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        oof = y + rng.normal(scale=0.5, size=200)
        ci = stats.bootstrap_ci(y, oof, n_boot=200)
        rmse = np.sqrt(np.mean((y - oof) ** 2))
        assert ci["rmse"][0] <= rmse <= ci["rmse"][1]

    def test_rank_models(self):
        rows = [{"model": "a", "metrics_mean": {"rmse": 2, "p95_ae": 1}},
                {"model": "b", "metrics_mean": {"rmse": 1, "p95_ae": 5}},
                {"model": "c", "metrics_mean": {"rmse": 1, "p95_ae": 2}}]
        ranked = stats.rank_models(rows)
        assert [r["model"] for r in ranked] == ["c", "b", "a"]
        assert ranked[0]["rank"] == 1


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

class TestBenchmark:

    @pytest.fixture(scope="class")
    def report(self, frame):
        cfg = BenchmarkConfig(models=["ols", "ridge", "random_forest", "hist_gbm"], n_splits=4,
                              leakage_check=True, temporal_check=True, intervals=True, importance=True)
        return benchmark_group(frame, "TEST", cfg)

    def test_report_structure(self, report):
        assert report["best_model"] in report["models"]
        assert report["ranking"][0] == report["best_model"]
        assert report["cv"]["plot_overlap_between_train_and_test"] == 0
        assert set(report["models"]) == {"ols", "ridge", "random_forest", "hist_gbm"}
        assert report["models"]["ols"]["vs_baseline"] is None
        assert "paired_t_on_fold_rmse" in report["models"]["ridge"]["vs_baseline"]

    def test_linear_models_report_coefficients_and_signs(self, report):
        ols = report["models"]["ols"]
        assert set(ols["coefficients"]) == {"intercept", *BASE_FEATURES}
        assert ols["sign_check"]["wrong_sign_count"] <= 1
        assert "coefficients" not in report["models"]["random_forest"]

    def test_leakage_check_reports_every_model(self, report):
        leak = report["leakage_check"]
        assert set(leak["per_model"]) == set(report["models"])
        assert leak["mean_r2_inflation"] is not None
        # With the seven equation features alone the plot effect is not identifiable,
        # so naive K-fold cannot cheat: inflation should be small either way.
        assert abs(leak["mean_r2_inflation"]) < 0.15

    def test_leakage_inflation_is_positive_when_features_identify_plots(self, frame):
        # Coordinates pin down the plot ⇒ a flexible learner memorises each plot's
        # random intercept from the *same plot's* other intervals under naive K-fold.
        cfg = BenchmarkConfig(models=["knn", "random_forest"], feature_set="spatial", n_splits=4,
                              leakage_check=True, temporal_check=False, intervals=False, importance=False)
        rep = benchmark_group(_repeated_measures(plot_sd=0.012), "SPATIAL", cfg)
        assert rep["leakage_check"]["mean_r2_inflation"] > 0.05
        for v in rep["leakage_check"]["per_model"].values():
            assert v["r2_naive_kfold"] > v["r2_grouped"]

    def test_intervals_and_importance_present(self, report):
        assert "split_conformal_best" in report["intervals"]
        assert "quantile_gbm" in report["intervals"]
        feats = [f["feature"] for f in report["importance"]["features"]]
        assert set(feats) == set(BASE_FEATURES)

    def test_other_strategies_recorded(self, report):
        assert {"kfold", "temporal"} <= set(report["models"]["ols"]["other_strategies"])

    def test_too_few_rows(self, frame):
        rep = benchmark_group(frame.head(10), "TINY", BenchmarkConfig(models=["ols"]))
        assert "error" in rep and rep["models"] == {}

    def test_benchmark_all_leaderboard_and_reports(self, frame, tmp_path):
        cfg = BenchmarkConfig(models=["ols", "huber"], n_splits=3, leakage_check=False,
                              temporal_check=False, intervals=False, importance=False)
        report = benchmark_all({"A": frame, "B": frame.iloc[::2]}, cfg)
        assert report["n_groups"] == 2
        assert {r["group"] for r in report["leaderboard"]} == {"A", "B"}
        assert sum(report["summary"]["wins_by_model"].values()) == 2
        assert report["summary"]["overall_best"] in {"ols", "huber"}
        path = write_json(report, out_dir=str(tmp_path))
        assert load_json(path)["n_groups"] == 2
        md = to_markdown(report)
        assert "# Model Benchmark Leaderboard" in md and "| ols |" in md


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class TestBenchmarkAPI:

    def test_registry_endpoint(self, client):
        body = client.get("/api/benchmarks/models").json()
        assert any(m["name"] == "ols" for m in body["models"])
        assert body["cv_strategies"] == ["group", "temporal", "kfold"]

    def test_latest_returns_404_when_no_report(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("main._BENCHMARK_DIR", str(tmp_path))
        assert client.get("/api/benchmarks").status_code == 404

    def test_run_job_end_to_end(self, client, monkeypatch, tmp_path):
        import time

        monkeypatch.setattr("main._BENCHMARK_DIR", str(tmp_path))
        payload = {"groups": ["Mh"], "models": ["ols", "ridge"], "n_splits": 3,
                   "leakage_check": False, "temporal_check": False, "intervals": False, "importance": False}
        res = client.post("/api/benchmarks/run", json=payload)
        assert res.status_code == 202
        job_id = res.json()["job_id"]
        for _ in range(120):
            job = client.get(f"/models/training-status/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.5)
        assert job["status"] == "completed", job.get("errors")
        assert job["kind"] == "benchmark"
        latest = client.get("/api/benchmarks").json()
        assert "Mh" in latest["report"]["groups"]
        group = client.get("/api/benchmarks/Mh").json()
        assert group["best_model"] in {"ols", "ridge"}
        assert client.get("/api/benchmarks/NotAGroup").status_code == 404

    def test_run_rejects_unknown_model(self, client):
        res = client.post("/api/benchmarks/run", json={"groups": ["Mh"], "models": ["nope"]})
        assert res.status_code == 400
