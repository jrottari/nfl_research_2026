"""Weekly lineup helper — the one command to run during the season for start/sit calls.

Trains the production weekly model (Ridge; statistically tied with XGBoost —
see reports/weekly_forecast_report.md) on every available season, projects
the upcoming week for every player with usable history, and attaches an
explosiveness score plus calibrated floor/ceiling bands so a start/sit call
isn't just "which median projection is higher."

Usage
-----
    # Auto-detect season + upcoming week, export CSV, print position boards
    python scripts/weekly_lineup.py

    # Force season/week (useful once the season is under way)
    python scripts/weekly_lineup.py --season 2026 --week 5

    # Head-to-head start/sit call
    python scripts/weekly_lineup.py --compare "Bijan Robinson" "James Cook"

Week-1 caveat
-------------
Within-season models need at least one game of *this* season's role/usage to
say anything. For players with zero games so far this season, this tool
falls back to last season's per-game average as a rough prior (tagged
``prior_season_only`` in the output) — treat those rows as low-confidence.
For a real preseason/week-1 board, use ``scripts/run_2026_tier1_forecast.py``
instead, which is built for exactly that (market consensus + draft capital,
no within-season history required).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR, Settings
from nfl_research.forecasting.data import load_multi_season
from nfl_research.weekly.data import defense_vs_position, load_multi_season_weekly
from nfl_research.weekly.features import (
    build_weekly_panel,
    make_upcoming_row,
    weekly_feature_matrix,
)
from nfl_research.weekly.models import RidgeWeeklyModel
from nfl_research.weekly.variance import (
    add_variance_features,
    fit_explosiveness_scaler,
    game_log_variance_snapshot,
    score_explosiveness,
)

DEFAULT_BAND_OFFSETS = {
    # Fallback if data/exports/weekly_variance_bands.csv hasn't been (re)generated
    # by scripts/analyze_weekly_variance.py. Fit on real 2023-2025 walk-forward
    # residuals; see reports/weekly_forecast_report.md for the calibration and
    # out-of-sample coverage check (0.607 vs a 0.60 target).
    "Low": (-5.21, 6.32),
    "Medium": (-5.64, 5.20),
    "High": (-6.26, 6.47),
}


def infer_current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def load_band_offsets() -> dict[str, tuple[float, float]]:
    path = EXPORT_DIR / "weekly_variance_bands.csv"
    if not path.exists():
        return DEFAULT_BAND_OFFSETS
    df = pd.read_csv(path)
    return {row["tercile"]: (row["floor_offset"], row["ceiling_offset"]) for _, row in df.iterrows()}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Weekly PPR lineup helper")
    p.add_argument("--first-season", type=int, default=2021, help="earliest training season (default 2021)")
    p.add_argument("--last-season", type=int, default=None, help="most recent completed season (default: auto)")
    p.add_argument("--season", type=int, default=None, help="season to forecast (default: current)")
    p.add_argument("--week", type=int, default=None, help="upcoming week to forecast (default: next unplayed)")
    p.add_argument("--top-n-fallback", type=int, default=150,
                   help="prior-season top-N cutoff for the cold-start fallback (default 150)")
    p.add_argument("--compare", nargs=2, metavar=("PLAYER_A", "PLAYER_B"), default=None,
                   help="print a head-to-head start/sit comparison for two players")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-export", action="store_true")
    return p.parse_args(argv)


def _prior_season_fallback_rows(
    candidate_ids: set[str],
    already_covered: set[str],
    season_totals: pd.DataFrame,
    prior_season: int,
    upcoming_week: int,
    upcoming_season: int,
    prior_weekly: pd.DataFrame,
    top_n: int,
) -> list[pd.DataFrame]:
    """Cold-start rows for players with zero games so far this season: use last
    season's per-game average as the entire signal. Restricted to players who
    were relevant (top-N PPR) last season, so the board isn't full of scrubs.
    """
    prior = season_totals[season_totals["season"] == prior_season].copy()
    if prior.empty:
        return []
    prior = prior[prior["games"] >= 4].sort_values("fantasy_points_ppr", ascending=False).head(top_n)

    rows = []
    for _, prec in prior.iterrows():
        pid = prec["player_id"]
        if pid in already_covered or pid not in candidate_ids:
            continue
        ppg = float(prec["ppg_ppr"])
        position = prec["position"]
        hist = prior_weekly[prior_weekly["player_id"] == pid].sort_values("week")
        snap = game_log_variance_snapshot(
            hist["fantasy_points_ppr"].tolist() if not hist.empty else [ppg], position
        )
        row = {
            "player_id": pid,
            "player_name": prec.get("player_name", pid),
            "position": position,
            "season": upcoming_season,
            "week": upcoming_week,
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
            "week_norm": (upcoming_week - 1) / 16.0,
            "pos_code": {"QB": 0, "RB": 1, "WR": 2, "TE": 3}.get(position, -1),
            "opp_ppr_allowed_avg": 0.0,
            "prior_season_ppg": ppg,
            "prior_season_games": float(prec["games"]),
            "ppr_std5": snap["ppr_std5"],
            "ppr_cv5": snap["ppr_cv5"],
            "boom_rate5": snap["boom_rate5"],
            "bust_rate5": snap["bust_rate5"],
            "data_source": "prior_season_only",
        }
        rows.append(pd.DataFrame([row]))
    return rows


def build_forecast_board(
    season: int, week: int, first_season: int, last_season: int, top_n_fallback: int
) -> pd.DataFrame:
    all_seasons = list(range(first_season, last_season + 1))
    if season not in all_seasons:
        all_seasons.append(season)
        all_seasons.sort()

    print(f"Loading weekly data {all_seasons[0]}-{all_seasons[-1]} (season {season} live)...")
    weekly = load_multi_season_weekly(all_seasons)
    season_totals = load_multi_season(all_seasons)
    defense_df = defense_vs_position(weekly)

    print("Building training panel...")
    panel = build_weekly_panel(weekly, defense_df=defense_df, prior_season_df=season_totals)
    panel = add_variance_features(panel)
    print(f"  {len(panel):,} training rows")

    model = RidgeWeeklyModel(alpha=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(weekly_feature_matrix(panel), panel["target"])

    scaler = fit_explosiveness_scaler(panel)

    current = weekly[(weekly["season"] == season) & (weekly["week"] < week)]
    active_ids = set(current["player_id"].unique())

    print(f"Building week {week} forecast rows for {len(active_ids)} active players...")
    fc_rows = []
    for pid in active_ids:
        row = make_upcoming_row(pid, weekly, upcoming_week=week, upcoming_season=season, defense_df=defense_df)
        if row is None:
            continue
        hist_vals = current[current["player_id"] == pid].sort_values("week")["fantasy_points_ppr"].tolist()
        position = row["position"].iloc[0]
        snap = game_log_variance_snapshot(hist_vals, position)
        for k, v in snap.items():
            row[k] = v
        row["data_source"] = "current_season"
        fc_rows.append(row)

    fallback_rows = _prior_season_fallback_rows(
        candidate_ids=set(season_totals[season_totals["season"] == season - 1]["player_id"]),
        already_covered=active_ids,
        season_totals=season_totals,
        prior_season=season - 1,
        upcoming_week=week,
        upcoming_season=season,
        prior_weekly=weekly[weekly["season"] == season - 1],
        top_n=top_n_fallback,
    )
    fc_rows.extend(fallback_rows)

    if not fc_rows:
        return pd.DataFrame()

    fc = pd.concat(fc_rows, ignore_index=True)
    X_fc = weekly_feature_matrix(fc)
    fc["proj_points"] = model.predict(X_fc).round(1)

    fc["explosiveness_score"] = [
        score_explosiveness(row["ppr_cv5"], row["boom_rate5"], row["position"], scaler)
        for _, row in fc.iterrows()
    ]
    bands = load_band_offsets()
    tercile_edges = fc.groupby("position")["explosiveness_score"].transform(
        lambda s: pd.qcut(s, 3, labels=["Low", "Medium", "High"], duplicates="drop")
        if s.nunique() >= 3
        else pd.Series(["Medium"] * len(s), index=s.index)
    )
    fc["risk_tier"] = tercile_edges.astype(str)
    default_offset = (-5.5, 6.5)
    offsets = fc["risk_tier"].map(lambda t: bands.get(t, default_offset))
    fc["floor"] = (fc["proj_points"] + offsets.map(lambda t: t[0])).clip(lower=0).round(1)
    fc["ceiling"] = (fc["proj_points"] + offsets.map(lambda t: t[1])).round(1)

    cols = [
        "player_name", "position", "season", "week", "proj_points", "floor", "ceiling",
        "explosiveness_score", "risk_tier", "boom_rate5", "data_source",
    ]
    board = fc[cols].sort_values(["position", "proj_points"], ascending=[True, False]).reset_index(drop=True)
    return board


def print_compare(board: pd.DataFrame, name_a: str, name_b: str) -> None:
    def _find(name: str) -> pd.Series | None:
        matches = board[board["player_name"].str.contains(name, case=False, na=False)]
        if matches.empty:
            print(f"  No match found for '{name}'")
            return None
        return matches.iloc[0]

    a, b = _find(name_a), _find(name_b)
    if a is None or b is None:
        return

    print("\n--- Start/Sit Comparison ---")
    print(pd.DataFrame([a, b])[["player_name", "position", "proj_points", "floor", "ceiling",
                                 "explosiveness_score", "risk_tier"]].to_string(index=False))

    higher_floor = a if a["floor"] >= b["floor"] else b
    higher_ceiling = a if a["ceiling"] >= b["ceiling"] else b
    print(f"\nHigher floor (protect a lead):  {higher_floor['player_name']} ({higher_floor['floor']} pts)")
    print(f"Higher ceiling (need upside):   {higher_ceiling['player_name']} ({higher_ceiling['ceiling']} pts)")
    if higher_floor["player_name"] == higher_ceiling["player_name"]:
        print(f"-> {higher_floor['player_name']} dominates on both ends - the easier call.")
    else:
        print("-> Genuine risk tradeoff: pick based on your matchup situation this week.")


def main(argv=None) -> int:
    args = parse_args(argv)
    season = args.season or infer_current_season()
    last_season = args.last_season or (season - 1 if season not in (2021,) else season)
    last_season = max(last_season, args.first_season)

    if args.week is None:
        weekly_live = load_multi_season_weekly([season])
        if weekly_live.empty or "season" not in weekly_live.columns:
            week = 1
        else:
            played = weekly_live[weekly_live["season"] == season]["week"]
            week = int(played.max()) + 1 if not played.empty else 1
    else:
        week = args.week

    board = build_forecast_board(
        season=season,
        week=week,
        first_season=args.first_season,
        last_season=last_season,
        top_n_fallback=args.top_n_fallback,
    )

    if board.empty:
        print("No forecast rows produced - check that the requested season/week has data available.")
        return 1

    if args.compare:
        print_compare(board, args.compare[0], args.compare[1])
        return 0

    for pos in ("QB", "RB", "WR", "TE"):
        sub = board[board["position"] == pos].head(20)
        if sub.empty:
            continue
        print(f"\n--- {pos} - Week {week} ({season}) ---")
        print(sub.drop(columns=["position", "season", "week"]).to_string(index=False))

    if not args.no_export:
        out_path = args.out or EXPORT_DIR / f"{season}_wk{week:02d}_lineup.csv"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        board.to_csv(out_path, index=False)
        print(f"\nLineup board -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
