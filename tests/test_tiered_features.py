"""Offline tests for tier builders, point-in-time guards, and ablation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nfl_research.forecasting.evaluation import (
    assert_tier_window,
    audit_target_correlations,
    score_predictions,
    shuffled_target_r2,
    walk_forward_evaluate,
)
from nfl_research.forecasting.feature_registry import cumulative_first_season
from nfl_research.forecasting.models import (
    NotFittedError,
    RandomWalkModel,
    RegressionToMeanModel,
    RidgeModel,
)
from nfl_research.forecasting.tier0 import add_missing_indicators
from nfl_research.forecasting.tier1 import build_ecr_features
from nfl_research.forecasting.tier2 import (
    build_depth_features,
    build_pbp_features,
    build_snap_features,
)
from nfl_research.forecasting.tier3 import build_nextgen_features, build_participation_features
from nfl_research.forecasting.tier4 import build_contract_features, build_injury_features

CUTOFF = datetime(2024, 8, 1)


def _panel(season: int = 2024) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["p1"],
            "player_name": ["One"],
            "position": ["WR"],
            "season": [season],
            "target": [200.0],
            "points_ppr_lag1": [180.0],
            "points_ppr_lag2": [np.nan],
            "points_ppr_lag3": [np.nan],
            "ppg_lag1": [12.0],
            "pos_code": [2],
        }
    )


def test_missing_lag_indicators_preserve_absence():
    out = add_missing_indicators(_panel())
    assert out.loc[0, ["lag1_missing", "lag2_missing", "lag3_missing"]].tolist() == [0, 1, 1]


def test_snap_builder_uses_prior_season_only():
    snaps = pd.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "season": [2023, 2023, 2024, 2024],
            "week": [1, 10, 1, 2],
            "offense_pct": [0.4, 0.8, 1.0, 1.0],
        }
    )
    out = build_snap_features(_panel(), snaps, as_of=CUTOFF)
    assert out.loc[0, "snap_pct_lag1"] == pytest.approx(0.6)
    assert out.loc[0, "wks_above_75pct_snaps"] == 1
    assert out.loc[0, "snap_pct_trend"] == pytest.approx(0.4)


def test_depth_builder_rejects_post_cutoff_chart():
    depth = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2024, 2024],
            "date": ["2024-07-15", "2024-10-01"],
            "depth": [2, 1],
        }
    )
    out = build_depth_features(_panel(), depth, as_of=CUTOFF)
    assert out.loc[0, "depth_chart_pos"] == 2


def test_pbp_builder_ignores_current_season():
    pbp = pd.DataFrame(
        {
            "season": [2023, 2023, 2024],
            "receiver_player_id": ["p1", "p2", "p1"],
            "rusher_player_id": [None, None, None],
            "posteam": ["A", "A", "A"],
            "air_yards": [100, 100, 9999],
            "yardline_100": [10, 40, 5],
        }
    )
    out = build_pbp_features(_panel(), pbp, as_of=CUTOFF)
    assert out.loc[0, "air_yards_share"] == pytest.approx(0.5)
    assert out.loc[0, "rz_target_share"] == pytest.approx(1.0)


def test_nextgen_and_participation_are_lagged():
    ngs = pd.DataFrame(
        {"player_id": ["p1", "p1"], "season": [2023, 2024], "avg_separation": [3.1, 99.0]}
    )
    out = build_nextgen_features(_panel(), [ngs], as_of=CUTOFF)
    assert out.loc[0, "separation_avg"] == pytest.approx(3.1)
    participation = pd.DataFrame(
        {"player_id": ["p1", "p1"], "season": [2023, 2024], "offense_pct": [0.7, 1.0]}
    )
    out = build_participation_features(out, participation, as_of=CUTOFF)
    assert out.loc[0, "participation_rate"] == pytest.approx(0.7)


def test_injury_builder_rejects_current_season():
    injuries = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2023, 2024],
            "date": ["2023-10-01", "2024-07-01"],
            "status": ["out", "out"],
        }
    )
    out = build_injury_features(_panel(), injuries, as_of=CUTOFF)
    assert out.loc[0, "games_missed_lag1"] == 1


def test_contract_requires_snapshot_date_and_honors_cutoff():
    contracts = pd.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "signed_date": ["2024-03-01", "2024-10-01"],
            "guaranteed": [10.0, 100.0],
            "end_year": [2026, 2028],
        }
    )
    out = build_contract_features(_panel(), contracts, as_of=CUTOFF)
    assert out.loc[0, "contract_guaranteed"] == 10.0
    assert out.loc[0, "contract_years_remaining"] == 2
    assert "contract_guaranteed" not in build_contract_features(
        _panel(), contracts.drop(columns="signed_date"), as_of=CUTOFF
    )


def test_ecr_join_excludes_idp_and_offense_only_pools():
    """FantasyPros reuses ecr_type='ro' across redraft-overall, redraft-idp, and
    redraft-offense. Ranking a mixed pool inflates a top-5 overall player's rank
    into the hundreds; page_type must isolate 'redraft-overall' alone."""
    rankings = pd.DataFrame(
        {
            "scrape_date": ["2024-08-15"] * 4,
            "player": ["One", "Two", "Three", "Four"],
            "pos": ["WR", "WR", "LB", "RB"],
            "ecr": [1.0, 2.0, 3.0, 4.0],
            "page_type": ["redraft-overall", "redraft-overall", "redraft-idp", "redraft-offense"],
            "ecr_type": ["ro", "ro", "ro", "ro"],
        }
    )
    out = build_ecr_features(_panel(season=2024), as_of=CUTOFF, rankings=rankings)
    # "One" is truly the #1 overall redraft-overall player and must rank 1, not be
    # diluted by the unrelated IDP/offense-only rows sharing the same ecr_type.
    row = out[out["player_name"] == "One"]
    assert row["ecr_rank"].iloc[0] == 1.0


def test_regression_to_mean_uses_ols_anchor():
    x = pd.DataFrame({"points_ppr_lag1": [100.0, 200.0, 300.0], "pos_code": [0, 0, 0]})
    y = pd.Series([80.0, 130.0, 180.0])
    model = RegressionToMeanModel().fit(x, y)
    assert model.predict(pd.DataFrame({"points_ppr_lag1": [200.0], "pos_code": [0]}))[
        0
    ] == pytest.approx(130.0)


def test_not_fitted_error_is_clean():
    with pytest.raises(NotFittedError):
        RidgeModel(max_tier=0).predict(pd.DataFrame())


def _evaluation_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for season in range(2018, 2025):
        for i in range(48):
            lag = rng.uniform(20, 300)
            rows.append(
                {
                    "player_id": f"p{i}",
                    "position": ["QB", "RB", "WR", "TE"][i % 4],
                    "season": season,
                    "points_ppr_lag1": lag,
                    "points_ppr_lag2": lag * 0.9,
                    "points_ppr_lag3": lag * 0.8,
                    "ppg_lag1": lag / 15,
                    "ppg_lag2": lag / 16,
                    "games_lag1": 15,
                    "games_lag2": 16,
                    "trend_1": lag * 0.1,
                    "trend_2": lag * 0.1,
                    "exp_smooth": lag * 0.9,
                    "career_season": 3,
                    "pos_code": i % 4,
                    "target": max(0, lag * 0.7 + rng.normal(0, 35)),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_and_metric_families():
    raw = walk_forward_evaluate(_evaluation_panel(), [RandomWalkModel()], min_train_seasons=2)
    scores = score_predictions(raw)
    required = {
        "mae",
        "rmse",
        "spearman_within_position",
        "top24_precision",
        "vorp_weighted_mae",
        "crps",
    }
    assert required <= set(scores.columns)
    assert scores["eval_season"].nunique() >= 2


def test_leak_audit_flags_target_clone():
    panel = _evaluation_panel()
    panel["points_ppr_lag1"] = panel["target"]
    findings = audit_target_correlations(panel, threshold=0.85, max_tier=0)
    assert findings[0].column == "points_ppr_lag1"


def test_shuffled_target_has_no_positive_holdout_signal():
    r2 = shuffled_target_r2(_evaluation_panel(), RidgeModel(max_tier=0), eval_season=2024)
    assert r2 < 0.1


def test_tier_window_matches_registry():
    first = cumulative_first_season(2)
    frame = pd.DataFrame({"season": [first, first + 1]})
    assert_tier_window(frame, 2)
    with pytest.raises(AssertionError):
        assert_tier_window(pd.DataFrame({"season": [first + 1]}), 2)


def test_in_fold_tuning_selects_ridge_alpha_without_leaking_eval_season():
    """Part 5.5: Ridge alpha should be tunable inside the training fold, and the
    grid search must never see the held-out evaluation season's rows."""
    panel = _evaluation_panel()
    raw = walk_forward_evaluate(
        panel, [RidgeModel(max_tier=0)], eval_seasons=[2024], min_train_seasons=2, tune=True
    )
    assert not raw.empty
    assert raw["eval_season"].unique().tolist() == [2024]
    assert np.isfinite(raw["predicted"]).all()


def test_crps_is_finite_when_a_model_exposes_predict_sd():
    """HierarchicalBayesModel.predict_sd() should feed a real Gaussian CRPS,
    while deterministic models (no predict_sd) keep CRPS as NaN."""
    predictions = pd.DataFrame(
        {
            "model": ["m", "m", "rw", "rw"],
            "max_tier": [1, 1, 0, 0],
            "eval_season": [2024, 2024, 2024, 2024],
            "player_id": ["p1", "p2", "p1", "p2"],
            "position": ["WR", "WR", "WR", "WR"],
            "actual": [100.0, 150.0, 100.0, 150.0],
            "predicted": [110.0, 140.0, 90.0, 160.0],
            "distribution_sd": [20.0, 20.0, np.nan, np.nan],
        }
    )
    predictions["error"] = predictions["predicted"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["sq_error"] = predictions["error"] ** 2
    scores = score_predictions(predictions).set_index("model")
    assert np.isfinite(scores.loc["m", "crps"])
    assert scores.loc["m", "crps"] > 0
    assert np.isnan(scores.loc["rw", "crps"])


def test_tuning_is_a_noop_for_models_without_a_param_grid():
    panel = _evaluation_panel()
    tuned = walk_forward_evaluate(
        panel, [RandomWalkModel()], eval_seasons=[2024], min_train_seasons=2, tune=True
    )
    untuned = walk_forward_evaluate(
        panel, [RandomWalkModel()], eval_seasons=[2024], min_train_seasons=2, tune=False
    )
    np.testing.assert_array_almost_equal(
        tuned.sort_values("player_id")["predicted"].to_numpy(),
        untuned.sort_values("player_id")["predicted"].to_numpy(),
    )
