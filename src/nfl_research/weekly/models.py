"""Weekly game-level forecast models.

All models expose fit(X, y) / predict(X) on DataFrames from
``weekly_feature_matrix()``.  ``name`` is used in reporting.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class RollingMeanModel:
    """Baseline: predict = average of the last N games (season average when N=all)."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.name = f"rolling_mean_{n}g"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RollingMeanModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        col = "ppr_ma3" if self.n == 3 else "ppr_season_avg"
        if col not in X.columns:
            col = "ppr_lag1"
        return X[col].fillna(0).values.clip(min=0)


class SeasonAvgModel:
    """Predict = current season average PPG (cumulative through prior week)."""

    name = "season_avg"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonAvgModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["ppr_season_avg"].fillna(X.get("ppr_lag1", pd.Series(0, index=X.index))).fillna(0).values.clip(min=0)


class OpponentAdjustedModel:
    """Season average PPG scaled by the opponent's position-level points-allowed rate.

    Adjustment factor = opp_ppr_allowed_avg / position_mean_allowed.
    Clipped to [0.6, 1.6] to avoid extreme swings early in the season.
    """

    name = "opp_adjusted"

    def __init__(self) -> None:
        self._pos_mean_allowed: dict[int, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OpponentAdjustedModel":
        if "opp_ppr_allowed_avg" in X.columns and "pos_code" in X.columns:
            for code in X["pos_code"].unique():
                mask = (X["pos_code"] == code) & X["opp_ppr_allowed_avg"].notna()
                if mask.any():
                    self._pos_mean_allowed[int(code)] = float(
                        X.loc[mask, "opp_ppr_allowed_avg"].mean()
                    )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = X["ppr_season_avg"].fillna(X.get("ppr_lag1", pd.Series(0, index=X.index))).fillna(0)

        if "opp_ppr_allowed_avg" not in X.columns or not self._pos_mean_allowed:
            return base.values.clip(min=0)

        factor = np.ones(len(X))
        for i, (_, row) in enumerate(X.iterrows()):
            opp_val = row.get("opp_ppr_allowed_avg", 0)
            pos_mean = self._pos_mean_allowed.get(int(row.get("pos_code", -1)), None)
            if pos_mean and pos_mean > 0 and not pd.isna(opp_val) and opp_val > 0:
                factor[i] = np.clip(opp_val / pos_mean, 0.6, 1.6)

        return (base.values * factor).clip(min=0)


class WeightedRollingModel:
    """Exponentially weighted blend of recent games.

    Weights: 0.50 * lag1 + 0.30 * ma3 + 0.20 * season_avg
    Bias term fit via OLS intercept on training data.
    """

    name = "weighted_rolling"

    def __init__(self) -> None:
        self._intercept: float = 0.0
        self._opp_coef: float = 0.0

    def _blend(self, X: pd.DataFrame) -> np.ndarray:
        lag1 = X["ppr_lag1"].fillna(0).values
        ma3  = X["ppr_ma3"].fillna(0).values
        ma3  = np.where(ma3 == 0, lag1, ma3)
        avg  = X["ppr_season_avg"].fillna(0).values
        avg  = np.where(avg == 0, ma3, avg)
        return 0.50 * lag1 + 0.30 * ma3 + 0.20 * avg

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeightedRollingModel":
        blend = self._blend(X)
        self._intercept = float(np.mean(y.values - blend))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._blend(X) + self._intercept).clip(min=0)


class RidgeWeeklyModel:
    """Ridge regression on the full weekly feature set."""

    name = "ridge_weekly"

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha = alpha
        self._pipeline: Pipeline | None = None
        self._cols: list[str] = []

    _BASE_COLS = [
        "ppr_lag1", "ppr_ma3", "ppr_ma5", "ppr_season_avg",
        "targets_ma3", "carries_ma3", "receptions_ma3",
        "target_share_ma3", "ppr_trend", "games_played",
        "week_norm", "opp_ppr_allowed_avg",
        "prior_season_ppg", "prior_season_games",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeWeeklyModel":
        self._cols = [c for c in self._BASE_COLS if c in X.columns]
        dummies = pd.get_dummies(X["pos_code"].astype(str), prefix="pos", drop_first=True)
        Xf = pd.concat([X[self._cols].fillna(0), dummies], axis=1)
        self._dummy_cols = list(Xf.columns)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(Xf, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dummies = pd.get_dummies(X["pos_code"].astype(str), prefix="pos", drop_first=True)
        Xf = pd.concat([X[[c for c in self._cols if c in X.columns]].fillna(0), dummies], axis=1)
        for c in self._dummy_cols:
            if c not in Xf.columns:
                Xf[c] = 0
        Xf = Xf[self._dummy_cols]
        return self._pipeline.predict(Xf).clip(min=0)  # type: ignore[union-attr]


class XGBoostWeeklyModel:
    """XGBoost on the full weekly feature set with opponent matchup."""

    name = "xgboost_weekly"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        reg_lambda: float = 3.0,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost not installed.")
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model: Any = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostWeeklyModel":
        skip = {"player_id", "player_name", "position", "season", "week",
                "season_type", "team", "opponent", "game_id"}
        self._cols = [c for c in X.columns if c not in skip]
        Xf = X[self._cols].fillna(0)
        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(Xf, y, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[[c for c in self._cols if c in X.columns]].fillna(0)
        return self._model.predict(Xf).clip(min=0)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_, index=self._cols
        ).sort_values(ascending=False)


def WEEKLY_MODELS() -> list:
    """One fresh instance of every available weekly model."""
    models: list = [
        RollingMeanModel(n=3),
        SeasonAvgModel(),
        WeightedRollingModel(),
        OpponentAdjustedModel(),
        RidgeWeeklyModel(alpha=5.0),
    ]
    if _XGB_AVAILABLE:
        models.append(XGBoostWeeklyModel())
    return models
