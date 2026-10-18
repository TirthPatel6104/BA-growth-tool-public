"""
ml/registry.py
==============
Catalogue of candidate regression algorithms for the RGR benchmark.

Every entry is a ``ModelSpec`` whose ``build()`` returns a *fresh* scikit-learn
compatible estimator.  Optional back-ends (XGBoost, LightGBM, statsmodels) are
imported lazily; when a library is missing the spec is reported as
``available=False`` and skipped instead of crashing the benchmark.

Families
--------
linear       – OLS (production baseline), Ridge, Lasso, ElasticNet, Huber, Bayesian ridge
additive     – spline (GAM-style) and quadratic polynomial expansions with ridge shrinkage
kernel       – k-nearest neighbours, support-vector regression
ensemble     – random forest, extra-trees, histogram gradient boosting, XGBoost, LightGBM
neural       – small multilayer perceptron
mixed        – linear mixed model with a random intercept per plot (statsmodels)

All scale-sensitive estimators are wrapped in a ``Pipeline`` with
``StandardScaler`` so the benchmark compares algorithms, not preprocessing luck.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    RidgeCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from .features import continuous_feature_indices

logger = logging.getLogger(__name__)

_ALPHAS = np.logspace(-6, 2, 25)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# ---------------------------------------------------------------------------
# Mixed-effects wrapper (statsmodels) – random intercept per plot
# ---------------------------------------------------------------------------

class MixedEffectsRegressor(BaseEstimator, RegressorMixin):
    """
    Linear mixed model  RGR = Xβ + u_plot + ε,  u_plot ~ N(0, σ²_u).

    Permanent-sample-plot data are *repeated measures*: the same plot
    contributes several growth intervals, so OLS residuals are correlated within
    plot and its standard errors are too optimistic.  A random intercept absorbs
    the plot-level effect.  Prediction for an *unseen* plot uses the fixed
    effects only (the population-average curve), which is exactly what grouped
    cross-validation tests.

    ``fit`` accepts ``groups`` (plot identifiers); without them it degrades to OLS.
    """

    needs_groups = True

    def __init__(self, reml: bool = True, maxiter: int = 200):
        self.reml = reml
        self.maxiter = maxiter

    def fit(self, X, y, groups=None):  # noqa: D401 – sklearn signature
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        exog = sm.add_constant(Xs, has_constant="add")

        if groups is None or len(set(map(str, groups))) < 2:
            self.ols_fallback_ = LinearRegression().fit(Xs, y)
            self.params_ = np.concatenate([[self.ols_fallback_.intercept_], self.ols_fallback_.coef_])
            self.random_effect_sd_ = 0.0
            self.converged_ = True
            return self

        codes = pd.factorize(pd.Series(groups).astype(str))[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence chatter on tiny/singular folds
                model = MixedLM(y, exog, groups=codes)
                result = model.fit(reml=self.reml, maxiter=self.maxiter, method=["lbfgs", "bfgs"])
            self.params_ = np.asarray(result.fe_params, dtype=float)
            cov_re = np.asarray(result.cov_re, dtype=float)
            self.random_effect_sd_ = float(np.sqrt(max(cov_re.ravel()[0], 0.0)))
            self.converged_ = bool(getattr(result, "converged", True))
            self.residual_sd_ = float(np.sqrt(result.scale))
        except Exception as exc:  # noqa: BLE001 – numerical failure → OLS
            logger.info("MixedLM failed (%s); falling back to OLS for this fit", exc)
            fallback = LinearRegression().fit(Xs, y)
            self.params_ = np.concatenate([[fallback.intercept_], fallback.coef_])
            self.random_effect_sd_ = 0.0
            self.converged_ = False
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.params_[0] + Xs @ self.params_[1:]

    @property
    def coef_(self) -> np.ndarray:  # on the original scale, for sign checks
        return self.params_[1:] / self.scaler_.scale_

    @property
    def intercept_(self) -> float:
        return float(self.params_[0] - np.sum(self.params_[1:] * self.scaler_.mean_ / self.scaler_.scale_))


# ---------------------------------------------------------------------------
# Spec + catalogue
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    family: str
    description: str
    build: Callable[[List[str], int], BaseEstimator]
    requires: Optional[str] = None          # importable module needed
    linear: bool = False                    # exposes interpretable coefficients
    tags: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.requires is None or _has(self.requires)


def _scaled(estimator) -> Pipeline:
    return make_pipeline(StandardScaler(), estimator)


def _scaled_target(estimator) -> TransformedTargetRegressor:
    """Standardise RGR (≈0.03 ± 0.01) so loss-scale-sensitive learners are not handicapped."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def _spline_gam(feature_names: List[str], seed: int) -> Pipeline:
    cont = continuous_feature_indices(feature_names)
    passthrough = [i for i in range(len(feature_names)) if i not in cont]
    expand = ColumnTransformer(
        [
            ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False), cont),
            ("binary", "passthrough", passthrough),
        ]
    )
    return Pipeline([("expand", expand), ("scale", StandardScaler()), ("ridge", RidgeCV(alphas=_ALPHAS))])


def _poly_ridge(feature_names: List[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", RidgeCV(alphas=_ALPHAS)),
        ]
    )


def _xgboost(feature_names: List[str], seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _lightgbm(feature_names: List[str], seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


MODEL_SPECS: Dict[str, ModelSpec] = {
    "ols": ModelSpec(
        "ols", "linear",
        "Ordinary least squares – the production RGR equation (baseline).",
        lambda f, s: LinearRegression(), linear=True, tags=["baseline"],
    ),
    "ridge": ModelSpec(
        "ridge", "linear",
        "L2-regularised linear model; alpha chosen by internal leave-one-out CV.",
        lambda f, s: _scaled(RidgeCV(alphas=_ALPHAS)), linear=True,
    ),
    "lasso": ModelSpec(
        "lasso", "linear",
        "L1-regularised linear model – performs automatic feature selection.",
        lambda f, s: _scaled(LassoCV(cv=5, random_state=s, max_iter=20000)), linear=True,
    ),
    "elastic_net": ModelSpec(
        "elastic_net", "linear",
        "Mixed L1/L2 penalty; stable when predictors are correlated (BA and BA²).",
        lambda f, s: _scaled(ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, random_state=s, max_iter=20000)),
        linear=True,
    ),
    "huber": ModelSpec(
        "huber", "linear",
        "Robust linear regression – down-weights outlying growth intervals.",
        lambda f, s: _scaled(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)), linear=True,
    ),
    "bayesian_ridge": ModelSpec(
        "bayesian_ridge", "linear",
        "Bayesian linear regression with evidence-maximised priors; gives predictive variance.",
        lambda f, s: _scaled(BayesianRidge()), linear=True,
    ),
    "spline_gam": ModelSpec(
        "spline_gam", "additive",
        "GAM-style additive model: cubic B-splines per predictor + ridge shrinkage.",
        _spline_gam,
    ),
    "poly2_ridge": ModelSpec(
        "poly2_ridge", "additive",
        "All degree-2 interactions and squares with ridge shrinkage.",
        _poly_ridge,
    ),
    "knn": ModelSpec(
        "knn", "kernel",
        "k-nearest neighbours (k=15, distance weighted) in standardised feature space.",
        lambda f, s: _scaled(KNeighborsRegressor(n_neighbors=15, weights="distance")),
    ),
    "svr": ModelSpec(
        "svr", "kernel",
        "RBF support-vector regression with epsilon-insensitive loss (target standardised).",
        lambda f, s: _scaled_target(_scaled(SVR(C=1.0, epsilon=0.1, gamma="scale"))),
    ),
    "random_forest": ModelSpec(
        "random_forest", "ensemble",
        "Bagged decision trees (300 trees, leaf ≥ 5 samples).",
        lambda f, s: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=1),
    ),
    "extra_trees": ModelSpec(
        "extra_trees", "ensemble",
        "Extremely randomised trees – lower variance than RF on small data.",
        lambda f, s: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=1),
    ),
    "hist_gbm": ModelSpec(
        "hist_gbm", "ensemble",
        "scikit-learn histogram gradient boosting with early stopping.",
        lambda f, s: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=s,
        ),
    ),
    "xgboost": ModelSpec(
        "xgboost", "ensemble",
        "XGBoost gradient boosting (depth 3, 400 rounds, subsampled).",
        _xgboost, requires="xgboost",
    ),
    "lightgbm": ModelSpec(
        "lightgbm", "ensemble",
        "LightGBM leaf-wise gradient boosting (15 leaves, 400 rounds).",
        _lightgbm, requires="lightgbm",
    ),
    "mlp": ModelSpec(
        "mlp", "neural",
        "Two-layer perceptron (64→32, ReLU) with early stopping (inputs and target standardised).",
        lambda f, s: _scaled_target(_scaled(MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
            max_iter=2000, early_stopping=True, n_iter_no_change=40, random_state=s,
        ))),
    ),
    "mixed_effects": ModelSpec(
        "mixed_effects", "mixed",
        "Linear mixed model with a random intercept per plot (repeated-measures aware).",
        lambda f, s: MixedEffectsRegressor(), requires="statsmodels", linear=True,
    ),
}

DEFAULT_MODELS: List[str] = [
    "ols", "ridge", "lasso", "elastic_net", "huber", "bayesian_ridge",
    "spline_gam", "poly2_ridge", "knn", "svr",
    "random_forest", "extra_trees", "hist_gbm", "xgboost", "lightgbm",
    "mlp", "mixed_effects",
]

FAST_MODELS: List[str] = ["ols", "ridge", "huber", "spline_gam", "random_forest", "hist_gbm", "mixed_effects"]


def list_models() -> List[Dict[str, object]]:
    """Catalogue for the API / CLI, including availability of optional back-ends."""
    return [
        {
            "name": spec.name,
            "family": spec.family,
            "description": spec.description,
            "linear": spec.linear,
            "requires": spec.requires,
            "available": spec.available,
            "tags": spec.tags,
        }
        for spec in MODEL_SPECS.values()
    ]


def available_models(names: Optional[List[str]] = None) -> List[ModelSpec]:
    """Resolve requested names (default: all) to available specs, logging skips."""
    wanted = names or DEFAULT_MODELS
    specs: List[ModelSpec] = []
    for name in wanted:
        spec = MODEL_SPECS.get(name)
        if spec is None:
            raise ValueError(f"Unknown model '{name}'. Valid: {sorted(MODEL_SPECS)}")
        if not spec.available:
            logger.warning("Skipping %s – optional dependency '%s' not installed", name, spec.requires)
            continue
        specs.append(spec)
    return specs


def needs_groups(estimator) -> bool:
    return bool(getattr(estimator, "needs_groups", False))


def linear_coefficients(estimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """
    Extract coefficients on the ORIGINAL feature scale from a linear estimator
    (bare, or ``StandardScaler → model`` pipeline).  Returns None otherwise.
    """
    model = estimator
    scaler = None
    if isinstance(estimator, Pipeline):
        steps = dict(estimator.named_steps)
        scaler = steps.get("standardscaler")
        model = estimator.steps[-1][1]
        if len(estimator.steps) != 2 or scaler is None:
            return None
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coef is None or intercept is None:
        return None
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.shape[0] != len(feature_names):
        return None
    if scaler is not None and not isinstance(model, MixedEffectsRegressor):
        coef = coef / scaler.scale_
        intercept = float(intercept) - float(np.sum(coef * scaler.mean_))
    return {"intercept": float(intercept), **{n: float(c) for n, c in zip(feature_names, coef)}}
