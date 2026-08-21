"""Offline tests for the weekly forecasting subpackage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_nflverse import make_weekly

from nfl_research import schema
from nfl_research.weekly.cv import walk_forward_weekly_cv
from nfl_research.weekly.data import defense_vs_position
from nfl_research.weekly.evaluate import boom_bust_accuracy, weekly_summary_table
from nfl_research.weekly.features import (
    build_weekly_panel,
    weekly_feature_cols,
    weekly_feature_matrix,
)
from nfl_research.weekly.models import (
    _XGB_AVAILABLE,
    OpponentAdjustedModel,
    RidgeWeeklyModel,
    RollingMeanModel,
    SeasonAvgModel,
    WeightedRollingModel,
)

# ---- Synthetic data --------------------------------------------------------


def _make_weekly_std(season: int, n_players: int = 150, seed: int = 0) -> pd.DataFrame:
    raw = make_weekly(season=season, n_players=n_players, seed=seed + season)
    df = schema.standardize(raw, strict=False)
    df = schema.coerce_numeric(df)
    df["season"] = season
    df["ppg_ppr"] = df["fantasy_points_ppr"]  # already per game in weekly
    return df


@pytest.fixture(scope="module")
def multi_weekly():
    frames = [_make_weekly_std(yr) for yr in [2021, 2022, 2023]]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def defense_table(multi_weekly):
    return defense_vs_position(multi_weekly)


@pytest.fixture(scope="module")
def weekly_panel(multi_weekly, defense_table):
    return build_weekly_panel(multi_weekly, defense_df=defense_table)


# ---- defense_vs_position ---------------------------------------------------


def test_defense_table_has_expected_columns(defense_table):
    for col in ("def_team", "season", "week", "position", "opp_ppr_allowed_avg"):
        assert col in defense_table.columns


def test_defense_table_no_future_leakage(defense_table):
    # Week 1 should have NaN for opp_ppr_allowed_avg (no prior games)
    wk1 = defense_table[defense_table["week"] == 1]["opp_ppr_allowed_avg"]
    assert wk1.isna().all() or len(wk1) == 0, "Week 1 should not have defence data"


# ---- build_weekly_panel ----------------------------------------------------


def test_panel_has_rolling_features(weekly_panel):
    for col in ("ppr_lag1", "ppr_ma3", "ppr_season_avg", "games_played"):
        assert col in weekly_panel.columns


def test_panel_drops_rows_with_no_history(weekly_panel):
    assert (weekly_panel["games_played"] >= 1).all()


def test_panel_target_is_present(weekly_panel):
    assert "target" in weekly_panel.columns
    assert weekly_panel["target"].notna().any()


def test_feature_matrix_no_nans(weekly_panel):
    X = weekly_feature_matrix(weekly_panel)
    assert not X.isna().any().any()


def test_weekly_feature_cols_consistent(weekly_panel):
    cols = weekly_feature_cols()
    X = weekly_feature_matrix(weekly_panel)
    assert set(X.columns).issubset(set(cols) | set(weekly_panel.columns))


# ---- Individual models -----------------------------------------------------


@pytest.fixture(scope="module")
def train_test_weekly(weekly_panel):
    train = weekly_panel[weekly_panel["season"] < 2023].copy()
    test = weekly_panel[weekly_panel["season"] == 2023].copy()
    return train, test


def _fit_predict_weekly(model, train, test):
    X_tr = weekly_feature_matrix(train)
    y_tr = train["target"]
    X_te = weekly_feature_matrix(test)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def test_rolling_mean_nonnegative(train_test_weekly):
    train, test = train_test_weekly
    preds = _fit_predict_weekly(RollingMeanModel(n=3), train, test)
    assert np.all(preds >= 0)
    assert np.all(np.isfinite(preds))


def test_season_avg_nonnegative(train_test_weekly):
    train, test = train_test_weekly
    preds = _fit_predict_weekly(SeasonAvgModel(), train, test)
    assert np.all(preds >= 0)


def test_weighted_rolling_nonnegative(train_test_weekly):
    train, test = train_test_weekly
    preds = _fit_predict_weekly(WeightedRollingModel(), train, test)
    assert np.all(preds >= 0)


def test_opp_adjusted_runs(train_test_weekly):
    train, test = train_test_weekly
    preds = _fit_predict_weekly(OpponentAdjustedModel(), train, test)
    assert len(preds) == len(test)
    assert np.all(np.isfinite(preds))


def test_ridge_weekly_nonnegative(train_test_weekly):
    train, test = train_test_weekly
    preds = _fit_predict_weekly(RidgeWeeklyModel(), train, test)
    assert np.all(preds >= 0)


@pytest.mark.skipif(not _XGB_AVAILABLE, reason="xgboost not installed")
def test_xgboost_weekly_runs(train_test_weekly):
    from nfl_research.weekly.models import XGBoostWeeklyModel

    train, test = train_test_weekly
    preds = _fit_predict_weekly(XGBoostWeeklyModel(n_estimators=20), train, test)
    assert len(preds) == len(test)
    assert np.all(np.isfinite(preds))


# ---- Walk-forward CV -------------------------------------------------------


def test_weekly_cv_returns_dataframe(weekly_panel):
    models = [RollingMeanModel(n=3), SeasonAvgModel()]
    cv = walk_forward_weekly_cv(
        weekly_panel,
        models,
        eval_seasons=[2023],
        top_n_filter=50,
        min_prior_games=1,
        min_train_rows=10,
        verbose=False,
    )
    assert isinstance(cv, pd.DataFrame)
    assert "abs_error" in cv.columns


def test_weekly_cv_abs_error_nonnegative(weekly_panel):
    models = [RollingMeanModel(n=3)]
    cv = walk_forward_weekly_cv(
        weekly_panel,
        models,
        eval_seasons=[2023],
        top_n_filter=50,
        min_prior_games=1,
        min_train_rows=10,
        verbose=False,
    )
    if not cv.empty:
        assert (cv["abs_error"] >= 0).all()


def test_weekly_summary_table_sorted(weekly_panel):
    models = [RollingMeanModel(n=3), SeasonAvgModel(), WeightedRollingModel()]
    cv = walk_forward_weekly_cv(
        weekly_panel,
        models,
        eval_seasons=[2023],
        top_n_filter=50,
        min_prior_games=1,
        min_train_rows=10,
        verbose=False,
    )
    if not cv.empty:
        tbl = weekly_summary_table(cv)
        assert tbl["MAE"].is_monotonic_increasing


def test_boom_bust_accuracy_has_all_models(weekly_panel):
    models = [RollingMeanModel(n=3), SeasonAvgModel()]
    cv = walk_forward_weekly_cv(
        weekly_panel,
        models,
        eval_seasons=[2023],
        top_n_filter=50,
        min_prior_games=1,
        min_train_rows=10,
        verbose=False,
    )
    if not cv.empty:
        bb = boom_bust_accuracy(cv)
        assert set(bb["Model"]) == {m.name for m in models}
