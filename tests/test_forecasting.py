"""Offline tests for the forecasting subpackage.

All tests use synthetic data from fake_nflverse; no network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_nflverse import make_season_totals, make_weekly  # noqa: E402

from nfl_research import schema  # noqa: E402
from nfl_research.forecasting.cv import walk_forward_cv  # noqa: E402
from nfl_research.forecasting.data import top_n_in_season  # noqa: E402
from nfl_research.forecasting.evaluate import summary_table  # noqa: E402
from nfl_research.forecasting.features import build_panel, feature_matrix  # noqa: E402
from nfl_research.forecasting.models import (  # noqa: E402
    _XGB_AVAILABLE,
    ExponentialSmoothingModel,
    PositionMeanModel,
    RandomWalkModel,
    RegressionToMeanModel,
    RidgeModel,
)

# ---- Synthetic multi-season data -------------------------------------------


def _make_multi_season(seasons=(2020, 2021, 2022, 2023), n_players=200):
    """Build a long-format multi-season frame from the fake fixture."""
    frames = []
    for yr in seasons:
        weekly = make_weekly(season=yr, n_players=n_players, seed=yr)
        totals = make_season_totals(weekly)
        totals["season"] = yr
        totals = schema.standardize(totals, strict=False)
        totals = schema.coerce_numeric(totals)
        totals["ppg_ppr"] = totals["fantasy_points_ppr"] / totals["games"].replace(0, np.nan)
        totals["ppg_ppr"] = totals["ppg_ppr"].fillna(0)
        frames.append(totals)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def multi_season():
    return _make_multi_season()


@pytest.fixture(scope="module")
def panel(multi_season):
    return build_panel(multi_season)


# ---- build_panel tests -------------------------------------------------------


def test_panel_has_lag_columns(panel):
    for col in ("points_ppr_lag1", "ppg_lag1", "games_lag1", "exp_smooth"):
        assert col in panel.columns, f"missing: {col}"


def test_panel_drops_no_history_rows(panel):
    assert panel["points_ppr_lag1"].isna().sum() == 0


def test_panel_target_column_present(panel):
    assert "target" in panel.columns


def test_exp_smooth_within_reasonable_range(panel):
    es = panel["exp_smooth"].dropna()
    assert (es >= 0).all()
    assert es.max() < 700  # sanity: no absurd values


def test_career_season_nonnegative(panel):
    assert (panel["career_season"] >= 0).all()


def test_feature_matrix_no_nans(panel):
    X = feature_matrix(panel)
    assert not X.isna().any().any()


# ---- top_n_in_season --------------------------------------------------------


def test_top_n_in_season_returns_set(multi_season):
    raw = multi_season.rename(columns={"fantasy_points_ppr": "fantasy_points_ppr"})
    ids = top_n_in_season(raw, 2021, n=50)
    assert isinstance(ids, set)
    assert len(ids) <= 50


# ---- Individual models -------------------------------------------------------


@pytest.fixture(scope="module")
def train_test(panel):
    train = panel[panel["season"] < 2023]
    test = panel[panel["season"] == 2023]
    return train, test


def _fit_predict(model, train, test):
    X_tr = feature_matrix(train)
    y_tr = train["target"]
    X_te = feature_matrix(test)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def test_random_walk_equals_lag1(train_test):
    train, test = train_test
    preds = _fit_predict(RandomWalkModel(), train, test)
    expected = feature_matrix(test)["points_ppr_lag1"].values
    np.testing.assert_array_almost_equal(preds, expected)


def test_position_mean_produces_finite_values(train_test):
    train, test = train_test
    preds = _fit_predict(PositionMeanModel(), train, test)
    assert np.all(np.isfinite(preds))


def test_exp_smoothing_fits_alpha(train_test):
    train, test = train_test
    m = ExponentialSmoothingModel()
    _fit_predict(m, train, test)
    assert 0.05 <= m.fitted_alpha <= 0.95


def test_regression_to_mean_clamps_nonnegative(train_test):
    train, test = train_test
    preds = _fit_predict(RegressionToMeanModel(), train, test)
    assert np.all(preds >= 0)


def test_ridge_no_negative_predictions(train_test):
    train, test = train_test
    preds = _fit_predict(RidgeModel(), train, test)
    assert np.all(preds >= 0)


@pytest.mark.skipif(not _XGB_AVAILABLE, reason="xgboost not installed")
def test_xgboost_runs(train_test):
    from nfl_research.forecasting.models import XGBoostModel

    train, test = train_test
    preds = _fit_predict(XGBoostModel(n_estimators=30), train, test)
    assert len(preds) == len(test)
    assert np.all(np.isfinite(preds))


# ---- Walk-forward CV ---------------------------------------------------------


def test_walk_forward_cv_returns_dataframe(panel):
    models = [RandomWalkModel(), PositionMeanModel()]
    cv = walk_forward_cv(
        panel,
        models,
        eval_seasons=[2022, 2023],
        top_n_filter=50,
        min_train_seasons=1,
        verbose=False,
    )
    assert isinstance(cv, pd.DataFrame)
    assert set(["model", "actual", "predicted", "abs_error"]).issubset(cv.columns)


def test_cv_has_two_models(panel):
    models = [RandomWalkModel(), ExponentialSmoothingModel()]
    cv = walk_forward_cv(
        panel, models, eval_seasons=[2023], top_n_filter=50, min_train_seasons=1, verbose=False
    )
    assert cv["model"].nunique() == 2


def test_cv_abs_error_nonnegative(panel):
    models = [RandomWalkModel()]
    cv = walk_forward_cv(
        panel, models, eval_seasons=[2023], top_n_filter=50, min_train_seasons=1, verbose=False
    )
    assert (cv["abs_error"] >= 0).all()


# ---- Evaluate ---------------------------------------------------------------


def test_summary_table_has_all_models(panel):
    models = [RandomWalkModel(), PositionMeanModel(), ExponentialSmoothingModel()]
    cv = walk_forward_cv(
        panel,
        models,
        eval_seasons=[2022, 2023],
        top_n_filter=50,
        min_train_seasons=1,
        verbose=False,
    )
    tbl = summary_table(cv)
    assert set(tbl["Model"]) == {m.name for m in models}


def test_summary_table_sorted_by_mae(panel):
    models = [RandomWalkModel(), PositionMeanModel()]
    cv = walk_forward_cv(
        panel,
        models,
        eval_seasons=[2022, 2023],
        top_n_filter=50,
        min_train_seasons=1,
        verbose=False,
    )
    tbl = summary_table(cv)
    assert tbl["MAE"].is_monotonic_increasing
