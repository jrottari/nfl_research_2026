"""Leak-aware walk-forward evaluation and nested feature-tier ablation."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from .feature_registry import cols_for_tier, cumulative_first_season
from .models import MODELS, MarketConsensusModel


@dataclass(frozen=True)
class LeakFinding:
    column: str
    spearman: float


def audit_target_correlations(
    panel: pd.DataFrame,
    *,
    target: str = "target",
    threshold: float = 0.85,
    max_tier: int = 4,
) -> list[LeakFinding]:
    """Return suspicious features for manual review; never auto-drop them."""
    findings: list[LeakFinding] = []
    for column in cols_for_tier(max_tier):
        if column not in panel or not pd.api.types.is_numeric_dtype(panel[column]):
            continue
        valid = panel[[column, target]].dropna()
        if len(valid) < 3 or valid[column].nunique() < 2:
            continue
        corr = float(spearmanr(valid[column], valid[target]).statistic)
        if np.isfinite(corr) and abs(corr) > threshold:
            findings.append(LeakFinding(column, corr))
    return sorted(findings, key=lambda item: abs(item.spearman), reverse=True)


def assert_tier_window(panel: pd.DataFrame, max_tier: int) -> None:
    expected = cumulative_first_season(max_tier)
    actual = int(pd.to_numeric(panel["season"]).min())
    if actual != expected:
        raise AssertionError(f"tier {max_tier} begins in {actual}, registry declares {expected}")


def _model_features(panel: pd.DataFrame, model) -> pd.DataFrame:
    columns = [c for c in cols_for_tier(model.feature_spec.max_tier) if c in panel]
    missing = [c for c in model.feature_spec.required if c not in columns]
    if missing:
        raise ValueError(f"{model.name} missing required features: {missing}")
    return panel[columns].copy()


def _season_walk_forward_splits(
    seasons: np.ndarray, min_train_seasons: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Nested walk-forward folds over training seasons, for in-fold tuning.

    Never a random split: fold k trains on the seasons before the k-th held-out
    season and validates on that season, mirroring the outer evaluation loop.
    """
    order = np.asarray(seasons)
    uniq = sorted(pd.unique(order).tolist())
    splits = []
    for i in range(min_train_seasons, len(uniq)):
        val_season = uniq[i]
        train_idx = np.flatnonzero(np.isin(order, uniq[:i]))
        val_idx = np.flatnonzero(order == val_season)
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))
    return splits


def tune_hyperparameters(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    *,
    min_train_seasons: int = 2,
) -> Any:
    """In-fold GridSearchCV over a nested walk-forward split of the training panel.

    Only models declaring a ``PARAM_GRID`` class attribute (Ridge alpha, XGBoost
    depth/learning-rate/reg_lambda) are tuned; every other model is returned
    fit but otherwise untouched. Tuning happens strictly inside the training
    fold passed in — the outer evaluation season is never part of the grid.
    """
    grid = getattr(model, "PARAM_GRID", None)
    if not grid:
        model.fit(X, y)
        return model
    splits = _season_walk_forward_splits(seasons.to_numpy(), min_train_seasons)
    if len(splits) < 2:
        model.fit(X, y)
        return model
    x_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    search = GridSearchCV(
        clone(model), grid, cv=splits, scoring="neg_mean_absolute_error", n_jobs=1, refit=True
    )
    search.fit(x_reset, y_reset)
    return search.best_estimator_


def walk_forward_evaluate(
    panel: pd.DataFrame,
    models: Iterable,
    *,
    eval_seasons: Iterable[int] | None = None,
    min_train_seasons: int = 3,
    tune: bool = False,
) -> pd.DataFrame:
    """Fit on seasons before T and predict T; never random-split player seasons.

    ``tune``: when True, models with a ``PARAM_GRID`` (Ridge, XGBoost) are
    hyperparameter-tuned inside each training fold via nested walk-forward
    GridSearchCV (Part 5.5), rather than using the hardcoded constructor
    defaults for every season. Off by default so existing callers/tests keep
    their current behavior.
    """
    seasons = sorted(int(s) for s in panel["season"].dropna().unique())
    eval_set = set(eval_seasons if eval_seasons is not None else seasons[1:])
    rows: list[dict] = []
    for season in seasons:
        if season not in eval_set:
            continue
        train = panel[panel["season"] < season]
        test = panel[panel["season"] == season]
        if train["season"].nunique() < min_train_seasons or test.empty:
            continue
        for prototype in models:
            model = clone(prototype)
            try:
                x_train = _model_features(train, model)
                x_test = _model_features(test, model)
                if tune:
                    model = tune_hyperparameters(model, x_train, train["target"], train["season"])
                else:
                    model.fit(x_train, train["target"])
                predictions = np.asarray(model.predict(x_test), dtype=float)
                sd = None
                if hasattr(model, "predict_sd"):
                    try:
                        sd = np.asarray(model.predict_sd(x_test), dtype=float)
                    except Exception:  # noqa: BLE001 - a missing/broken predict_sd shouldn't kill scoring
                        sd = None
            except (ImportError, ValueError) as exc:
                warnings.warn(f"Skipping {model.name} in {season}: {exc}", stacklevel=2)
                continue
            for index, (_, player) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "model": model.name,
                        "max_tier": model.feature_spec.max_tier,
                        "eval_season": season,
                        "player_id": player.get("player_id", ""),
                        "position": player.get("position", ""),
                        "actual": float(player["target"]),
                        "predicted": float(predictions[index]),
                        "distribution_sd": float(sd[index]) if sd is not None else float("nan"),
                    }
                )
    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions["error"] = predictions["predicted"] - predictions["actual"]
        predictions["abs_error"] = predictions["error"].abs()
        predictions["sq_error"] = predictions["error"] ** 2
    return predictions


def _top_k_precision(group: pd.DataFrame, k: int = 24) -> float:
    k = min(k, len(group))
    if not k:
        return float("nan")
    actual = set(group.nlargest(k, "actual").index)
    predicted = set(group.nlargest(k, "predicted").index)
    return len(actual & predicted) / k


def _vorp_weights(group: pd.DataFrame, replacement_rank: int = 24) -> pd.Series:
    actual = group["actual"]
    replacement = actual.nlargest(min(replacement_rank, len(actual))).iloc[-1]
    return (actual - replacement).clip(lower=0) + 1.0


def _gaussian_crps(actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma)."""
    from scipy.stats import norm

    z = (actual - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report raw error, within-position ranking, VORP error, and CRPS."""
    if predictions.empty:
        return pd.DataFrame()
    market = predictions[predictions["model"] == MarketConsensusModel.name]
    market_mae = market.groupby("eval_season")["abs_error"].mean().to_dict()
    rows: list[dict] = []
    tier_column = "ablation_tier" if "ablation_tier" in predictions else "max_tier"
    for (model, tier, season), group in predictions.groupby(["model", tier_column, "eval_season"]):
        pos_corrs = []
        precisions = []
        weighted_num = 0.0
        weighted_den = 0.0
        for _, pos in group.groupby("position"):
            if len(pos) >= 3 and pos["actual"].nunique() > 1 and pos["predicted"].nunique() > 1:
                pos_corrs.append(float(spearmanr(pos["actual"], pos["predicted"]).statistic))
            precisions.append(_top_k_precision(pos))
            weights = _vorp_weights(pos)
            weighted_num += float((weights * pos["abs_error"]).sum())
            weighted_den += float(weights.sum())
        sd = group["distribution_sd"] if "distribution_sd" in group else pd.Series(dtype=float)
        has_distribution = sd.notna().all() and len(sd) and (sd > 0).all()
        crps = (
            float(
                np.mean(
                    _gaussian_crps(
                        group["actual"].to_numpy(), group["predicted"].to_numpy(), sd.to_numpy()
                    )
                )
            )
            if has_distribution
            else float("nan")  # deterministic models do not expose a distribution
        )
        mae = float(mean_absolute_error(group["actual"], group["predicted"]))
        baseline = market_mae.get(season, float("nan"))
        rows.append(
            {
                "model": model,
                "max_tier": int(tier),
                "eval_season": int(season),
                "mae": mae,
                "rmse": float(mean_squared_error(group["actual"], group["predicted"]) ** 0.5),
                "spearman_within_position": float(np.nanmean(pos_corrs))
                if pos_corrs
                else float("nan"),
                "top24_precision": float(np.nanmean(precisions)),
                "vorp_weighted_mae": weighted_num / weighted_den if weighted_den else float("nan"),
                "crps": crps,
                "skill_vs_market": 1.0 - mae / baseline if baseline > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def shuffled_target_r2(
    panel: pd.DataFrame,
    model,
    *,
    eval_season: int,
    random_state: int = 42,
    repeats: int = 20,
) -> float:
    """Leak smoke test: mean shuffled-target holdout R² should be non-positive."""
    train = panel[panel["season"] < eval_season]
    test = panel[panel["season"] == eval_season]
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(repeats):
        fitted = clone(model)
        shuffled = pd.Series(rng.permutation(train["target"].to_numpy()), index=train.index)
        fitted.fit(_model_features(train, fitted), shuffled)
        predicted = fitted.predict(_model_features(test, fitted))
        scores.append(float(r2_score(test["target"], predicted)))
    return float(np.mean(scores))


def run_ablation(
    panel: pd.DataFrame,
    *,
    tiers: Iterable[int] = range(5),
    model_factory: Callable[[int], list] = MODELS,
    min_train_seasons: int = 3,
    tune: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested tier sweep with a tier-0 Ridge control on every restricted panel."""
    all_predictions: list[pd.DataFrame] = []
    for tier in tiers:
        first = cumulative_first_season(tier)
        restricted = panel[pd.to_numeric(panel["season"]) >= first].copy()
        predicted = walk_forward_evaluate(
            restricted, model_factory(tier), min_train_seasons=min_train_seasons, tune=tune
        )
        if tier > 0:
            from .models import RidgeModel

            control = walk_forward_evaluate(
                restricted,
                [RidgeModel(max_tier=0)],
                min_train_seasons=min_train_seasons,
                tune=tune,
            )
            if not control.empty:
                control["model"] = "ridge_tier0_control"
                predicted = pd.concat([predicted, control], ignore_index=True)
        if not predicted.empty:
            predicted["ablation_tier"] = tier
            all_predictions.append(predicted)
    raw = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return raw, score_predictions(raw)


def ablation_table(scores: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Rows=models, columns=tiers, cells=mean holdout metric across seasons."""
    return scores.pivot_table(index="model", columns="max_tier", values=metric, aggfunc="mean")
