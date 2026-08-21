"""Model classes for PPR point total forecasting.

All models expose a minimal sklearn-compatible interface::

    model.fit(X_train, y_train)   # X is a DataFrame from features.feature_matrix()
    model.predict(X_test)         # returns a numpy array of point-total forecasts

The ``name`` attribute is used in reporting.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import pymc as pm
    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False


class RandomWalkModel:
    """Baseline: forecast = last year's PPR total (random walk)."""

    name = "random_walk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomWalkModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["points_ppr_lag1"].fillna(0).values


class PositionMeanModel:
    """Second baseline: predict the position-conditional mean from training data."""

    name = "position_mean"

    def __init__(self) -> None:
        self._means: dict[int, float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PositionMeanModel":
        self._global_mean = float(y.mean())
        for code in X["pos_code"].unique():
            mask = X["pos_code"] == code
            self._means[int(code)] = float(y[mask].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["pos_code"].map(self._means).fillna(self._global_mean).values


class ExponentialSmoothingModel:
    """Single exponential smoothing with learnable alpha.

    The forecast is a weighted blend of lag1, lag2, lag3 where weights decay
    geometrically: alpha, alpha*(1-alpha), alpha*(1-alpha)^2.
    Alpha is chosen to minimise MSE on the training set.
    """

    name = "exp_smoothing"

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._fitted_alpha: float = alpha

    def _predict_with_alpha(self, X: pd.DataFrame, alpha: float) -> np.ndarray:
        a, b, c = alpha, alpha * (1 - alpha), alpha * (1 - alpha) ** 2
        l1 = X["points_ppr_lag1"].fillna(0)
        l2 = X["points_ppr_lag2"].fillna(0)
        l3 = X["points_ppr_lag3"].fillna(0) if "points_ppr_lag3" in X.columns else pd.Series(0, index=X.index)

        # normalise weights by how many lags are actually available
        w1 = a * X["points_ppr_lag1"].notna().astype(float)
        w2 = b * X["points_ppr_lag2"].notna().astype(float)
        w3 = c * (X["points_ppr_lag3"].notna().astype(float) if "points_ppr_lag3" in X.columns else 0)
        total_w = (w1 + w2 + w3).replace(0, float("nan"))

        return ((l1 * w1 + l2 * w2 + l3 * w3) / total_w).fillna(l1).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExponentialSmoothingModel":
        def mse(alpha: float) -> float:
            preds = self._predict_with_alpha(X, alpha)
            return float(np.mean((preds - y.values) ** 2))

        result = minimize_scalar(mse, bounds=(0.05, 0.95), method="bounded")
        self._fitted_alpha = float(result.x)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_with_alpha(X, self._fitted_alpha)

    @property
    def fitted_alpha(self) -> float:
        return self._fitted_alpha


class RegressionToMeanModel:
    """Per-position regression-to-mean: forecast = pos_mean + beta*(lag1 - pos_mean).

    Beta is fit separately per position. Equivalent to shrinkage toward the
    position mean; beta=1 is random walk, beta=0 is pure position mean.
    """

    name = "regression_to_mean"

    def __init__(self) -> None:
        self._pos_mean: dict[int, float] = {}
        self._pos_beta: dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressionToMeanModel":
        for code in X["pos_code"].unique():
            mask = X["pos_code"] == code
            xc = X.loc[mask, "points_ppr_lag1"].fillna(0)
            yc = y[mask]
            mu = float(yc.mean())
            self._pos_mean[int(code)] = mu
            # OLS of y on lag1 within position
            x_centered = xc - mu
            if x_centered.std() > 0:
                beta = float(np.cov(x_centered, yc - mu)[0, 1] / x_centered.var())
                beta = float(np.clip(beta, 0.0, 1.5))
            else:
                beta = 1.0
            self._pos_beta[int(code)] = beta
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X))
        lag1 = X["points_ppr_lag1"].fillna(0).values
        for i, code in enumerate(X["pos_code"].values):
            mu = self._pos_mean.get(int(code), float(np.mean(lag1)))
            beta = self._pos_beta.get(int(code), 1.0)
            out[i] = mu + beta * (lag1[i] - mu)
        return out


class RidgeModel:
    """Ridge regression with position dummies and StandardScaler."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self._feature_cols: list[str] = []
        self._pipeline: Pipeline | None = None

    def _make_cols(self, X: pd.DataFrame) -> list[str]:
        base = [
            "points_ppr_lag1", "points_ppr_lag2", "ppg_lag1", "ppg_lag2",
            "games_lag1", "games_lag2", "trend_1", "trend_2",
            "career_season", "exp_smooth",
        ]
        return [c for c in base if c in X.columns]

    def _add_pos_dummies(self, X: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(X["pos_code"].astype(str), prefix="pos", drop_first=True)
        return pd.concat([X[self._feature_cols].fillna(0), dummies], axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self._feature_cols = self._make_cols(X)
        Xd = self._add_pos_dummies(X)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(Xd, y)
        self._dummy_cols = list(Xd.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xd = self._add_pos_dummies(X)
        # align columns (train might have had dummies test doesn't)
        for c in self._dummy_cols:
            if c not in Xd.columns:
                Xd[c] = 0
        Xd = Xd[self._dummy_cols]
        return self._pipeline.predict(Xd).clip(min=0)  # type: ignore[union-attr]


class XGBoostModel:
    """XGBoost gradient boosting with the full feature set."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 5.0,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model: Any = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._cols = [c for c in X.columns if c not in ("player_id", "player_name", "position", "season")]
        Xf = X[self._cols].fillna(0)
        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(Xf, y, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[[c for c in self._cols if c in X.columns]].fillna(0)
        return self._model.predict(Xf).clip(min=0)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_,
            index=self._cols,
        ).sort_values(ascending=False)


class HierarchicalBayesModel:
    """Hierarchical Bayesian model with position-level priors (PyMC).

    mu_i = alpha[pos] + beta[pos] * lag1 + gamma * ppg_lag1
    y_i  ~ Normal(mu_i, sigma)

    Position-level alpha and beta share hyperpriors (partial pooling).
    Uses MAP estimation by default for speed; call sample() for full MCMC.
    """

    name = "hierarchical_bayes"

    def __init__(self, use_map: bool = True, draws: int = 500, tune: int = 300) -> None:
        if not _PYMC_AVAILABLE:
            raise ImportError("pymc is not installed. pip install pymc")
        self.use_map = use_map
        self.draws = draws
        self.tune = tune
        self._map: dict[str, Any] = {}
        self._positions: list[str] = []

    def _build_model(
        self,
        lag1: np.ndarray,
        ppg1: np.ndarray,
        pos_idx: np.ndarray,
        n_positions: int,
        y: np.ndarray | None = None,
    ):
        import pymc as pm
        coords = {"position": self._positions}
        with pm.Model(coords=coords) as model:
            # Hyperpriors
            alpha_mu = pm.Normal("alpha_mu", mu=50, sigma=40)
            alpha_sd = pm.HalfNormal("alpha_sd", sigma=20)
            beta_mu = pm.Normal("beta_mu", mu=0.75, sigma=0.10)
            beta_sd = pm.HalfNormal("beta_sd", sigma=0.15)

            # Position-level parameters
            alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sd, dims="position")
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sd, dims="position")
            gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

            sigma = pm.HalfNormal("sigma", sigma=40)

            mu = alpha[pos_idx] + beta[pos_idx] * lag1 + gamma * ppg1

            if y is not None:
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        return model

    def _prepare(self, X: pd.DataFrame) -> tuple:
        pos_codes = sorted(X["pos_code"].unique())
        self._positions = [str(c) for c in pos_codes]
        pos_map = {c: i for i, c in enumerate(pos_codes)}
        lag1 = X["points_ppr_lag1"].fillna(0).values.astype(float)
        ppg1 = X["ppg_lag1"].fillna(0).values.astype(float)
        pos_idx = X["pos_code"].map(pos_map).fillna(0).values.astype(int)
        return lag1, ppg1, pos_idx, len(pos_codes), pos_map

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HierarchicalBayesModel":
        import pymc as pm
        lag1, ppg1, pos_idx, n_pos, self._pos_map = self._prepare(X)
        y_arr = y.values.astype(float)

        model = self._build_model(lag1, ppg1, pos_idx, n_pos, y=y_arr)
        with model:
            if self.use_map:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._map = pm.find_MAP(progressbar=False)
            else:
                self._trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=2,
                    progressbar=False,
                    return_inferencedata=True,
                    target_accept=0.9,
                )
                # Store posterior means as MAP-equivalent for predict()
                import arviz as az
                post = self._trace.posterior
                self._map = {
                    "alpha": post["alpha"].mean(("chain", "draw")).values,
                    "beta": post["beta"].mean(("chain", "draw")).values,
                    "gamma": float(post["gamma"].mean()),
                }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag1, ppg1, pos_idx, n_pos, _ = self._prepare(X)

        alpha_vals = self._map.get("alpha", np.zeros(len(self._positions)))
        beta_vals = self._map.get("beta", np.ones(len(self._positions)) * 0.75)
        gamma_val = float(self._map.get("gamma", 0.0))

        # Remap pos_idx to stored positions
        pos_map = {c: i for i, c in enumerate(sorted(X["pos_code"].unique()))}
        # pos_idx already computed in _prepare; use the stored _pos_map
        pos_idx = X["pos_code"].map(self._pos_map).fillna(0).values.astype(int)
        # Clip to valid range
        pos_idx = np.clip(pos_idx, 0, len(alpha_vals) - 1)

        preds = alpha_vals[pos_idx] + beta_vals[pos_idx] * lag1 + gamma_val * ppg1
        return np.clip(preds, 0, None)

    def position_params(self) -> pd.DataFrame:
        """Return fitted alpha/beta per position for inspection."""
        alpha = self._map.get("alpha", [])
        beta = self._map.get("beta", [])
        return pd.DataFrame({
            "position": self._positions,
            "alpha": alpha,
            "beta": beta,
        })


def MODELS() -> list:
    """Return one fresh instance of every available model."""
    models = [
        RandomWalkModel(),
        PositionMeanModel(),
        ExponentialSmoothingModel(),
        RegressionToMeanModel(),
        RidgeModel(alpha=10.0),
    ]
    if _XGB_AVAILABLE:
        models.append(XGBoostModel())
    if _PYMC_AVAILABLE:
        models.append(HierarchicalBayesModel(use_map=True))
    return models
