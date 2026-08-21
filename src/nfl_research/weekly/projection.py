"""Shared weekly-model training + player projection.

Both ``scripts/weekly_lineup.py`` (single-league board) and the Sleeper
multi-league optimizer (``nfl_research.sleeper``) need the same thing: train
Ridge once on all available history, then project an arbitrary set of
player ids for an upcoming (season, week). Factored out here so a run that
touches several leagues trains the model exactly once.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from ..forecasting.data import load_multi_season
from .data import defense_vs_position, load_multi_season_weekly
from .features import build_weekly_panel, make_upcoming_row, weekly_feature_matrix
from .models import RidgeWeeklyModel
from .variance import (
    add_variance_features,
    attach_risk_bands,
    fit_explosiveness_scaler,
    fit_tercile_thresholds,
    game_log_variance_snapshot,
    score_explosiveness,
)

_POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


@dataclass
class WeeklyModelBundle:
    model: RidgeWeeklyModel
    scaler: dict = field(repr=False)
    tercile_thresholds: dict[str, tuple[float, float]]
    weekly: pd.DataFrame = field(repr=False)
    season_totals: pd.DataFrame = field(repr=False)
    defense_df: pd.DataFrame = field(repr=False)
    panel: pd.DataFrame = field(repr=False)


def fit_weekly_model(seasons: list[int], alpha: float = 5.0) -> WeeklyModelBundle:
    """Load all available weekly/season data for ``seasons``, build the panel
    (rolling + variance features), and fit the production Ridge model on every
    row. Seasons already cached locally are read from disk; the current season
    is fetched live. See reports/weekly_forecast_report.md for why Ridge.
    """
    weekly = load_multi_season_weekly(seasons)
    season_totals = load_multi_season(seasons)
    defense_df = defense_vs_position(weekly)

    panel = build_weekly_panel(weekly, defense_df=defense_df, prior_season_df=season_totals)
    panel = add_variance_features(panel)

    model = RidgeWeeklyModel(alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(weekly_feature_matrix(panel), panel["target"])

    scaler = fit_explosiveness_scaler(panel)
    tercile_thresholds = fit_tercile_thresholds(panel)

    return WeeklyModelBundle(
        model=model,
        scaler=scaler,
        tercile_thresholds=tercile_thresholds,
        weekly=weekly,
        season_totals=season_totals,
        defense_df=defense_df,
        panel=panel,
    )


def _prior_season_fallback_row(
    player_id: str,
    position: str,
    player_name: str,
    ppg: float,
    prior_games: float,
    season: int,
    week: int,
    prior_weekly: pd.DataFrame,
) -> dict:
    hist = prior_weekly[prior_weekly["player_id"] == player_id].sort_values("week")
    values = hist["fantasy_points_ppr"].tolist() if not hist.empty else [ppg]
    snap = game_log_variance_snapshot(values, position)
    return {
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "season": season,
        "week": week,
        "ppr_lag1": ppg,
        "ppr_ma3": ppg,
        "ppr_ma5": ppg,
        "ppr_season_avg": ppg,
        "targets_ma3": 0.0,
        "carries_ma3": 0.0,
        "receptions_ma3": 0.0,
        "target_share_ma3": float("nan"),
        "ppr_trend": 0.0,
        "games_played": 0,
        "week_norm": (week - 1) / 16.0,
        "pos_code": _POS_CODE.get(position, -1),
        "opp_ppr_allowed_avg": 0.0,
        "prior_season_ppg": ppg,
        "prior_season_games": float(prior_games),
        **snap,
        "data_source": "prior_season_only",
    }


def project_players(
    candidate_ids: set[str],
    bundle: WeeklyModelBundle,
    season: int,
    week: int,
    min_prior_games: int = 4,
    prior_top_n: int | None = None,
) -> pd.DataFrame:
    """Project every id in ``candidate_ids`` for (season, week).

    Uses current-season history when available (``data_source=current_season``);
    otherwise falls back to last season's per-game average
    (``data_source=prior_season_only``) for any candidate who played at least
    ``min_prior_games`` games last season. ``prior_top_n`` optionally further
    restricts the fallback pool to the top-N prior-season PPR scorers (used by
    the generic all-players board to keep it from filling with irrelevant
    names); leave it ``None`` when ``candidate_ids`` is already a specific,
    small set (e.g. a Sleeper roster), so every rostered player is covered
    regardless of rank.

    Returns a board with ``proj_points``, ``floor``, ``ceiling``,
    ``explosiveness_score``, ``risk_tier``, and ``data_source`` columns.
    Positions outside QB/RB/WR/TE (e.g. K, DEF) are not modeled and never
    appear in the output — see the Sleeper report for how callers handle that.
    """
    weekly, season_totals, defense_df = bundle.weekly, bundle.season_totals, bundle.defense_df
    current = weekly[(weekly["season"] == season) & (weekly["week"] < week)]

    rows: list[pd.DataFrame] = []
    covered: set[str] = set()

    for pid in candidate_ids:
        hist = current[current["player_id"] == pid]
        if hist.empty:
            continue
        row = make_upcoming_row(
            pid, weekly, upcoming_week=week, upcoming_season=season, defense_df=defense_df
        )
        if row is None:
            continue
        position = row["position"].iloc[0]
        hist_values = hist.sort_values("week")["fantasy_points_ppr"].tolist()
        snap = game_log_variance_snapshot(hist_values, position)
        for k, v in snap.items():
            row[k] = v
        row["data_source"] = "current_season"
        rows.append(row)
        covered.add(pid)

    remaining = candidate_ids - covered
    if remaining:
        prior = season_totals[
            (season_totals["season"] == season - 1)
            & season_totals["player_id"].isin(remaining)
            & (season_totals["games"] >= min_prior_games)
        ].copy()
        if prior_top_n:
            prior = prior.sort_values("fantasy_points_ppr", ascending=False).head(prior_top_n)
        prior_weekly = weekly[weekly["season"] == season - 1]
        for _, prec in prior.iterrows():
            rows.append(
                pd.DataFrame(
                    [
                        _prior_season_fallback_row(
                            prec["player_id"],
                            prec["position"],
                            prec.get("player_name", prec["player_id"]),
                            float(prec["ppg_ppr"]),
                            prec["games"],
                            season,
                            week,
                            prior_weekly,
                        )
                    ]
                )
            )
            covered.add(prec["player_id"])

    if not rows:
        return pd.DataFrame()

    fc = pd.concat(rows, ignore_index=True)
    X_fc = weekly_feature_matrix(fc)
    fc["proj_points"] = bundle.model.predict(X_fc).round(1)
    fc["explosiveness_score"] = [
        score_explosiveness(r["ppr_cv5"], r["boom_rate5"], r["position"], bundle.scaler)
        for _, r in fc.iterrows()
    ]
    fc = attach_risk_bands(fc, bundle.tercile_thresholds)
    return fc
