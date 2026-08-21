"""Model classes for PPR point total forecasting.

All models expose a minimal sklearn-compatible interface::

    model.fit(X_train, y_train)   # X is a DataFrame from feature_matrix()
    model.predict(X_test)         # returns a numpy array of point-total forecasts

Every model class declares a ``feature_spec`` attribute (FeatureSpec) so the
evaluation harness knows which tier of features each model needs.  The tier-aware
models (Ridge, XGBoost, HierarchicalBayes) also accept a ``max_tier`` constructor
argument so the same class can be evaluated at every tier in the ablation sweep.
"""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

_PYMC_AVAILABLE = find_spec("pymc") is not None

# Columns that identify rows but are not features
_ID_COLS = frozenset({"player_id", "player_name", "position", "season", "target"})


class NotFittedError(RuntimeError):
    pass


class FeatureSpec(NamedTuple):
    max_tier: int  # highest tier this model consumes
    required: tuple[str, ...]  # columns that must be present at fit() time


def _require_features(X: pd.DataFrame, spec: FeatureSpec) -> None:
    missing = [column for column in spec.required if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")


# ---------------------------------------------------------------------------
# Tier-0 baselines
# ---------------------------------------------------------------------------


class RandomWalkModel(BaseEstimator):
    """Baseline: forecast = last year's PPR total (random walk)."""

    name = "random_walk"
    feature_spec = FeatureSpec(max_tier=0, required=("points_ppr_lag1",))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomWalkModel:
        _require_features(X, self.feature_spec)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["points_ppr_lag1"].fillna(0).values


class PositionMeanModel(BaseEstimator):
    """Second baseline: predict the position-conditional mean from training data."""

    name = "position_mean"
    feature_spec = FeatureSpec(max_tier=0, required=("pos_code",))

    def __init__(self) -> None:
        self._means: dict[int, float] = {}
        self._global_mean: float = 0.0
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PositionMeanModel:
        _require_features(X, self.feature_spec)
        self._global_mean = float(y.mean())
        for code in X["pos_code"].unique():
            mask = X["pos_code"] == code
            self._means[int(code)] = float(y[mask].mean())
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")
        return X["pos_code"].map(self._means).fillna(self._global_mean).values


class ExponentialSmoothingModel(BaseEstimator):
    """Single exponential smoothing with learnable alpha.

    The forecast is a weighted blend of lag1, lag2, lag3 where weights decay
    geometrically: alpha, alpha*(1-alpha), alpha*(1-alpha)^2.
    Alpha is chosen to minimise MSE on the training set.
    """

    name = "exp_smoothing"
    feature_spec = FeatureSpec(
        max_tier=0,
        required=("points_ppr_lag1", "points_ppr_lag2"),
    )

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._fitted_alpha: float = alpha
        self._fitted = False

    def _predict_with_alpha(self, X: pd.DataFrame, alpha: float) -> np.ndarray:
        a, b, c = alpha, alpha * (1 - alpha), alpha * (1 - alpha) ** 2
        l1 = X["points_ppr_lag1"].fillna(0)
        l2 = X["points_ppr_lag2"].fillna(0)
        l3 = (
            X["points_ppr_lag3"].fillna(0)
            if "points_ppr_lag3" in X.columns
            else pd.Series(0.0, index=X.index)
        )

        w1 = a * X["points_ppr_lag1"].notna().astype(float)
        w2 = b * X["points_ppr_lag2"].notna().astype(float)
        w3 = c * (
            X["points_ppr_lag3"].notna().astype(float) if "points_ppr_lag3" in X.columns else 0.0
        )
        total_w = (w1 + w2 + w3).replace(0, float("nan"))

        return ((l1 * w1 + l2 * w2 + l3 * w3) / total_w).fillna(l1).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ExponentialSmoothingModel:
        _require_features(X, self.feature_spec)

        def mse(alpha: float) -> float:
            preds = self._predict_with_alpha(X, alpha)
            return float(np.mean((preds - y.values) ** 2))

        result = minimize_scalar(mse, bounds=(0.05, 0.95), method="bounded")
        self._fitted_alpha = float(result.x)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")
        return self._predict_with_alpha(X, self._fitted_alpha)

    @property
    def fitted_alpha(self) -> float:
        return self._fitted_alpha


class RegressionToMeanModel(BaseEstimator):
    """Per-position regression-to-mean: forecast = mu_lag1 + beta*(lag1 - mu_lag1).

    Bug fixed vs original: OLS line passes through (mean(lag1), mean(y)), not
    (mean(y), mean(y)).  We now store the per-position mean of lag1 as the
    centering constant (not the mean of y), which matches the OLS geometry.
    """

    name = "regression_to_mean"
    feature_spec = FeatureSpec(max_tier=0, required=("points_ppr_lag1", "pos_code"))

    def __init__(self) -> None:
        self._lag1_mean: dict[int, float] = {}  # centering constant = E[lag1 | pos]
        self._y_mean: dict[int, float] = {}  # predicted value when lag1 == E[lag1]
        self._pos_beta: dict[int, float] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegressionToMeanModel:
        _require_features(X, self.feature_spec)
        for code in X["pos_code"].unique():
            mask = X["pos_code"] == code
            lag1 = X.loc[mask, "points_ppr_lag1"].fillna(0)
            yc = y[mask]
            mu_x = float(lag1.mean())  # center of lag1 (correct OLS anchor)
            mu_y = float(yc.mean())  # E[y | pos]
            self._lag1_mean[int(code)] = mu_x
            self._y_mean[int(code)] = mu_y
            x_centered = lag1 - mu_x
            if x_centered.std() > 0:
                beta = float(np.cov(x_centered, yc)[0, 1] / x_centered.var())
                beta = float(np.clip(beta, 0.0, 1.5))
            else:
                beta = 1.0
            self._pos_beta[int(code)] = beta
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")
        out = np.zeros(len(X))
        lag1 = X["points_ppr_lag1"].fillna(0).values
        codes = X["pos_code"].values
        global_mu_x = float(np.mean(list(self._lag1_mean.values()))) if self._lag1_mean else 0.0
        global_mu_y = float(np.mean(list(self._y_mean.values()))) if self._y_mean else 0.0
        for i, code in enumerate(codes):
            mu_x = self._lag1_mean.get(int(code), global_mu_x)
            mu_y = self._y_mean.get(int(code), global_mu_y)
            beta = self._pos_beta.get(int(code), 1.0)
            out[i] = mu_y + beta * (lag1[i] - mu_x)
        return np.clip(out, 0, None)


# ---------------------------------------------------------------------------
# Tier-aware parametric models
# ---------------------------------------------------------------------------


class RidgeModel(BaseEstimator):
    """Ridge regression with position encoding and StandardScaler.

    Fixes vs original:
    - Uses fitted OneHotEncoder(drop='first') instead of pd.get_dummies so
      train/test always drop the same reference category.
    - Accepts max_tier to slice the wide feature matrix.
    """

    name = "ridge"
    _BASE_REQUIRED = ("points_ppr_lag1", "pos_code")

    # In-fold tuning grid consumed by evaluation.tune_hyperparameters (Part 5.5):
    # alpha is otherwise a global hardcoded constant that a wide feature matrix
    # can easily overfit or underfit depending on tier.
    PARAM_GRID = {"alpha": [0.1, 1.0, 10.0, 50.0, 200.0]}

    def __init__(self, alpha: float = 10.0, max_tier: int = 4) -> None:
        self.alpha = alpha
        self.max_tier = max_tier
        self._feature_cols: list[str] = []
        self._pipeline: Pipeline | None = None
        self._encoder: OneHotEncoder | None = None
        self._fitted = False

    @property
    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec(max_tier=self.max_tier, required=self._BASE_REQUIRED)

    def get_params(self, deep: bool = True) -> dict:
        return {"alpha": self.alpha, "max_tier": self.max_tier}

    def set_params(self, **params: Any) -> RidgeModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _select_features(self, X: pd.DataFrame) -> list[str]:
        from .feature_registry import REGISTRY

        allowed = {
            col
            for col, spec in REGISTRY.items()
            if spec.tier <= self.max_tier and col not in _ID_COLS and col != "target"
        }
        return [c for c in X.columns if c in allowed and X[c].dtype.kind in "fiub"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeModel:
        _require_features(X, self.feature_spec)
        self._feature_cols = self._select_features(X)
        if not self._feature_cols:
            # Fallback if registry not available yet — keep backwards compat
            self._feature_cols = [
                c for c in X.columns if c not in _ID_COLS and X[c].dtype.kind in "fiub"
            ]

        # Fit position encoder once on training data
        self._encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
        pos_train = X[["pos_code"]].fillna(-1).astype(str)
        self._encoder.fit(pos_train)

        Xf = self._encode(X)
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha)),
            ]
        )
        self._pipeline.fit(Xf, y)
        self._fitted = True
        return self

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._encoder is None:
            raise NotFittedError("Call fit() before predict()")
        numeric = X[[c for c in self._feature_cols if c != "pos_code"]].fillna(0)
        pos_encoded = pd.DataFrame(
            self._encoder.transform(X[["pos_code"]].fillna(-1).astype(str)),
            columns=self._encoder.get_feature_names_out(["pos_code"]),
            index=X.index,
        )
        return pd.concat([numeric, pos_encoded], axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._pipeline is None:
            raise NotFittedError("Call fit() before predict()")
        Xf = self._encode(X)
        return self._pipeline.predict(Xf).clip(min=0)


class XGBoostModel(BaseEstimator):
    """XGBoost gradient boosting with tier-aware feature selection.

    Fixes vs original:
    - predict() uses X.reindex(columns=self._cols).fillna(0) (not list-comp)
    - Uses allowlist from feature registry rather than a denylist
    - Attributes initialised in __init__ so AttributeError on unfit model is clean
    """

    name = "xgboost"

    # In-fold tuning grid consumed by evaluation.tune_hyperparameters (Part 5.5).
    PARAM_GRID = {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_lambda": [1.0, 5.0, 15.0],
    }

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 5.0,
        max_tier: int = 4,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.max_tier = max_tier
        self._model: Any = None
        self._cols: list[str] = []
        self._fitted = False

    @property
    def feature_spec(self) -> FeatureSpec:
        return FeatureSpec(max_tier=self.max_tier, required=("points_ppr_lag1",))

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "max_tier": self.max_tier,
        }

    def set_params(self, **params: Any) -> XGBoostModel:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _allowlist_cols(self, X: pd.DataFrame) -> list[str]:
        try:
            from .feature_registry import REGISTRY

            allowed = {
                col
                for col, spec in REGISTRY.items()
                if spec.tier <= self.max_tier and col not in _ID_COLS and col != "target"
            }
            cols = [c for c in X.columns if c in allowed and X[c].dtype.kind in "fiub"]
        except ImportError:
            cols = [c for c in X.columns if c not in _ID_COLS and X[c].dtype.kind in "fiub"]
        return cols

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        _require_features(X, self.feature_spec)
        self._cols = self._allowlist_cols(X)
        Xf = X.reindex(columns=self._cols).fillna(0)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model.fit(Xf, y, verbose=False)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise NotFittedError("Call fit() before predict()")
        Xf = X.reindex(columns=self._cols).fillna(0)
        return self._model.predict(Xf).clip(min=0)

    def feature_importance(self) -> pd.Series:
        if not self._fitted:
            raise NotFittedError("Call fit() before feature_importance()")
        return pd.Series(self._model.feature_importances_, index=self._cols).sort_values(
            ascending=False
        )


class HierarchicalBayesModel(BaseEstimator):
    """Hierarchical Bayesian model with position-level priors (PyMC).

    Fixes vs original:
    1. _prepare() is now pure — never writes to self.  fit() and predict()
       both call it and use the returned values directly.
    2. lag1 is centered before entering the likelihood so alpha and beta
       are no longer strongly correlated.
    3. Non-centered parameterization for position-level effects.
    4. Default inference switched to ADVI (pm.fit) to escape the
       HalfNormal funnel; MAP still available via use_map=True.
    5. Attributes initialised in __init__.
    """

    name = "hierarchical_bayes"

    def __init__(
        self,
        use_map: bool = False,
        draws: int = 1000,
        tune: int = 500,
        advi_iterations: int = 30_000,
        max_tier: int = 1,
    ) -> None:
        if not _PYMC_AVAILABLE:
            raise ImportError("pymc is not installed. pip install pymc")
        self.use_map = use_map
        self.draws = draws
        self.tune = tune
        self.advi_iterations = advi_iterations
        self.max_tier = max_tier
        # Initialise all fit-time attributes so AttributeError on unfit model is clean
        self._map: dict[str, Any] = {}
        self._positions: list[str] = []
        self._pos_map: dict[int, int] = {}
        self._lag1_train_mean: float = 0.0
        self._fitted = False

    @property
    def feature_spec(self) -> FeatureSpec:  # type: ignore[override]
        return FeatureSpec(
            max_tier=self.max_tier, required=("points_ppr_lag1", "ppg_lag1", "pos_code")
        )

    def _prepare(self, X: pd.DataFrame) -> tuple:
        """Pure helper — returns derived arrays, writes nothing to self."""
        pos_codes = sorted(X["pos_code"].unique())
        positions = [str(c) for c in pos_codes]
        pos_map = {c: i for i, c in enumerate(pos_codes)}
        lag1 = X["points_ppr_lag1"].fillna(0).values.astype(float)
        ppg1 = X["ppg_lag1"].fillna(0).values.astype(float)
        pos_idx = X["pos_code"].map(pos_map).fillna(0).values.astype(int)
        return lag1, ppg1, pos_idx, len(pos_codes), pos_map, positions

    def _build_model(
        self,
        lag1_c: np.ndarray,  # already centered
        ppg1: np.ndarray,
        pos_idx: np.ndarray,
        positions: list[str],
        y: np.ndarray | None = None,
    ):
        import pymc as pm

        n_pos = len(positions)
        coords = {"position": positions}
        with pm.Model(coords=coords) as model:
            # Hyperpriors
            alpha_mu = pm.Normal("alpha_mu", mu=150.0, sigma=60.0)
            alpha_sd = pm.HalfNormal("alpha_sd", sigma=30.0)
            beta_mu = pm.Normal("beta_mu", mu=0.75, sigma=0.15)
            beta_sd = pm.HalfNormal("beta_sd", sigma=0.15)

            # Non-centered position effects
            alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, shape=n_pos)
            beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=n_pos)
            alpha = pm.Deterministic("alpha", alpha_mu + alpha_sd * alpha_offset)
            beta = pm.Deterministic("beta", beta_mu + beta_sd * beta_offset)

            gamma = pm.Normal("gamma", mu=0.0, sigma=5.0)
            sigma = pm.HalfNormal("sigma", sigma=60.0)

            # lag1_c is already centred so alpha is the expected output at lag1=E[lag1]
            mu = alpha[pos_idx] + beta[pos_idx] * lag1_c + gamma * ppg1

            if y is not None:
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HierarchicalBayesModel:
        import pymc as pm

        _require_features(X, self.feature_spec)

        lag1, ppg1, pos_idx, n_pos, pos_map, positions = self._prepare(X)
        lag1_mean = float(lag1.mean())
        lag1_c = lag1 - lag1_mean
        y_arr = y.values.astype(float)

        # Store for predict()
        self._positions = positions
        self._pos_map = pos_map
        self._lag1_train_mean = lag1_mean

        model = self._build_model(lag1_c, ppg1, pos_idx, positions, y=y_arr)

        with model:
            if self.use_map:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._map = pm.find_MAP(progressbar=False)
            else:
                approx = pm.fit(
                    n=self.advi_iterations,
                    method="advi",
                    progressbar=False,
                    random_seed=42,
                )
                trace = approx.sample(self.draws, random_seed=42)
                post = trace.posterior
                self._map = {
                    "alpha": post["alpha"].mean(("chain", "draw")).values,
                    "beta": post["beta"].mean(("chain", "draw")).values,
                    "gamma": float(post["gamma"].mean()),
                    "sigma": float(post["sigma"].mean()),
                }

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")

        lag1, ppg1, pos_idx_raw, _, test_pos_map, _ = self._prepare(X)
        lag1_c = lag1 - self._lag1_train_mean

        # Map test pos_codes to training position indices
        # (test may have seen fewer positions than train)
        pos_idx = (
            X["pos_code"]
            .map(self._pos_map)  # map to train index
            .fillna(0)
            .values.astype(int)
        )
        pos_idx = np.clip(pos_idx, 0, len(self._positions) - 1)

        alpha_vals = np.asarray(self._map.get("alpha", np.zeros(len(self._positions))))
        beta_vals = np.asarray(self._map.get("beta", np.ones(len(self._positions)) * 0.75))
        gamma_val = float(self._map.get("gamma", 0.0))

        preds = alpha_vals[pos_idx] + beta_vals[pos_idx] * lag1_c + gamma_val * ppg1
        return np.clip(preds, 0, None)

    def predict_sd(self, X: pd.DataFrame) -> np.ndarray:
        """Homoscedastic predictive sd (posterior residual sigma) for CRPS scoring.

        This is only the observation-noise component of predictive uncertainty
        (it ignores parameter uncertainty in alpha/beta), so it understates true
        posterior predictive spread; it is still a legitimate, cheap-to-compute
        distribution for the evaluation harness's CRPS metric.
        """
        if not self._fitted:
            raise NotFittedError("Call fit() before predict_sd()")
        sigma = float(self._map.get("sigma", 60.0))
        return np.full(len(X), sigma)

    def position_params(self) -> pd.DataFrame:
        """Return fitted alpha/beta per position for inspection."""
        if not self._fitted:
            raise NotFittedError("Call fit() before position_params()")
        return pd.DataFrame(
            {
                "position": self._positions,
                "alpha": self._map.get("alpha", []),
                "beta": self._map.get("beta", []),
            }
        )


# ---------------------------------------------------------------------------
# Market baseline (tier 1)
# ---------------------------------------------------------------------------


class MarketConsensusModel(BaseEstimator):
    """Market baseline: map preseason FantasyPros ECR rank to PPR point forecast.

    ECR (Expert Consensus Ranking) is the preseason consensus draft ranking.
    Fit a position-conditional log-linear regression of actual PPR on ECR rank
    from training data; apply it at test time.  A model that doesn't beat this
    baseline is an expensive way to reproduce consensus rankings.

    Requires tier-1 feature ``ecr_rank`` in the feature matrix.
    Falls back to random-walk prediction when ECR is missing.
    """

    name = "market_consensus"
    feature_spec = FeatureSpec(max_tier=1, required=("ecr_rank", "pos_code"))

    def __init__(self) -> None:
        self._pos_models: dict[int, tuple[float, float]] = {}  # (intercept, slope) per pos
        self._global_model: tuple[float, float] = (150.0, -0.5)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MarketConsensusModel:
        _require_features(X, self.feature_spec)
        if "ecr_rank" not in X.columns:
            # No ECR data available in this training window — degenerate to position mean
            self._fitted = True
            return self

        has_ecr = X["ecr_rank"].notna()
        Xr = X[has_ecr]
        yr = y[has_ecr]

        if len(Xr) >= 50:
            log_rank = np.log1p(Xr["ecr_rank"].clip(lower=1).values)
            coeffs = np.polyfit(log_rank, yr.values, 1)
            self._global_model = (float(coeffs[1]), float(coeffs[0]))

        for code in X["pos_code"].unique():
            mask = has_ecr & (X["pos_code"] == code)
            if mask.sum() < 20:
                continue
            xc = np.log1p(X.loc[mask, "ecr_rank"].clip(lower=1).values)
            yc = y[mask].values
            c = np.polyfit(xc, yc, 1)
            self._pos_models[int(code)] = (float(c[1]), float(c[0]))

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")
        if "ecr_projection" in X.columns:
            projection = pd.to_numeric(X["ecr_projection"], errors="coerce")
        else:
            projection = pd.Series(np.nan, index=X.index)

        out = np.zeros(len(X))
        for i, (_idx, row) in enumerate(X.iterrows()):
            ecr = row.get("ecr_rank", float("nan"))
            if pd.notna(projection.iloc[i]):
                out[i] = max(0.0, float(projection.iloc[i]))
                continue
            if pd.isna(ecr):
                out[i] = row.get("points_ppr_lag1", 0.0) or 0.0
                continue
            log_ecr = float(np.log1p(max(ecr, 1)))
            code = int(row.get("pos_code", -1))
            intercept, slope = self._pos_models.get(code, self._global_model)
            out[i] = max(0.0, intercept + slope * log_ecr)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def MODELS(max_tier: int = 4) -> list:
    """Return one fresh instance of every available model.

    Tier-aware models (Ridge, XGBoost, HierarchicalBayes) are configured with
    ``max_tier``.  Tier-0 baselines are always returned unchanged.

    Args:
        max_tier: Passed to Ridge, XGBoostModel, HierarchicalBayesModel.
                  Set to 0 to get all models consuming only base features.
    """
    models: list = [
        RandomWalkModel(),
        PositionMeanModel(),
        ExponentialSmoothingModel(),
        RegressionToMeanModel(),
        RidgeModel(alpha=10.0, max_tier=max_tier),
    ]

    if _XGB_AVAILABLE:
        models.append(XGBoostModel(max_tier=max_tier))
    else:
        warnings.warn(
            "xgboost not installed; XGBoostModel excluded from MODELS(). pip install xgboost",
            stacklevel=2,
        )

    if _PYMC_AVAILABLE:
        models.append(HierarchicalBayesModel(use_map=False, max_tier=min(max_tier, 1)))
    else:
        warnings.warn(
            "pymc not installed; HierarchicalBayesModel excluded from MODELS(). pip install pymc",
            stacklevel=2,
        )

    if max_tier >= 1:
        models.append(MarketConsensusModel())

    return models
