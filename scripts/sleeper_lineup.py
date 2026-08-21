"""Pull every Sleeper league for a user and optimize each one's starting lineup.

One command covers all your leagues in a season: pulls each roster from
Sleeper, projects every rostered QB/RB/WR/TE with the same production model
as ``scripts/weekly_lineup.py`` (trained once, reused across leagues), solves
the optimal starting lineup for each league's actual slot requirements
(FLEX-aware), and diffs it against what's currently started so you get a
concrete swap list.

Usage
-----
    # All of your NFL leagues, next unplayed week, optimize for median points
    python scripts/sleeper_lineup.py --username YOUR_SLEEPER_USERNAME

    # A specific week/season, optimize for ceiling (you're chasing points)
    python scripts/sleeper_lineup.py --username YOUR_SLEEPER_USERNAME --week 6 --objective ceiling

    # Just one league
    python scripts/sleeper_lineup.py --username YOUR_SLEEPER_USERNAME --league "Dynasty Warfare"

Scope / limitations
--------------------
- Only QB/RB/WR/TE are modeled (same as the rest of this system) — K/DEF
  slots are passed through unchanged from your actual current Sleeper
  starter, since there's no projection to optimize them against.
- Scoring format is adjusted for reception value only (full/half/standard
  PPR) via ``scoring_settings['rec']`` — other custom scoring (TD point
  values, yardage bonuses, IDP) is not modeled; treat point totals as
  directionally right, not exact, for leagues with unusual scoring.
- IR/taxi-squad players are excluded from the optimizer (not startable) but
  still shown in the roster if you want to sanity check the pull.
- Week-1 / zero-game players fall back to last season's per-game average,
  same cold-start behavior as ``weekly_lineup.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR
from nfl_research.sleeper.optimize import (
    MODELED_POSITIONS,
    SLOT_ELIGIBILITY,
    adjust_for_scoring,
    modeled_slots,
    optimize_lineup,
)
from nfl_research.sleeper.rosters import LeagueRoster, fetch_all_league_rosters
from nfl_research.weekly.data import load_multi_season_weekly
from nfl_research.weekly.projection import fit_weekly_model, project_players

# League/team/player names come from Sleeper as arbitrary user-entered Unicode
# (emoji, smart quotes, etc.) — Windows consoles default to cp1252, which can't
# encode most of it. Replace rather than crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def infer_current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sleeper multi-league lineup optimizer")
    p.add_argument("--username", required=True, help="Sleeper username or numeric user_id")
    p.add_argument("--season", type=int, default=None, help="season (default: current)")
    p.add_argument("--week", type=int, default=None, help="upcoming week (default: next unplayed)")
    p.add_argument("--first-season", type=int, default=2021,
                   help="earliest training season (default 2021)")
    p.add_argument("--last-season", type=int, default=None,
                   help="most recent completed season (default: auto)")
    p.add_argument("--objective", choices=["mean", "floor", "ceiling"], default="mean",
                   help="optimize for median, floor (protect a lead), or ceiling (need upside)")
    p.add_argument("--league", default=None,
                   help="only process leagues whose name contains this substring")
    p.add_argument("--no-export", action="store_true")
    return p.parse_args(argv)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "league"


def _fill_unmodeled_slot(slot: str, roster_df: pd.DataFrame) -> dict | None:
    """K/DEF/IDP slots aren't projected — carry over the actual current
    Sleeper starter for that slot, or any eligible rostered player if none is
    currently started, or leave it empty."""
    eligible = SLOT_ELIGIBILITY.get(slot, {slot})
    startable = roster_df["position"].isin(eligible) & ~roster_df["is_ir"] & ~roster_df["is_taxi"]
    pool = roster_df[startable]
    if pool.empty:
        return None
    started = pool[pool["is_starter"]]
    pick = started.iloc[0] if not started.empty else pool.iloc[0]
    row = pick.to_dict()
    row["slot"] = slot
    return row


def process_league(
    league: LeagueRoster, bundle, season: int, week: int, objective: str
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    print(f"\n{'=' * 70}\n{league.league_name}  (team: {league.team_name})\n{'=' * 70}")

    roster_df = league.players
    if roster_df.empty:
        print("  Roster is empty (league likely hasn't drafted yet) — skipping.")
        return None

    candidates = set(
        roster_df.loc[
            roster_df["position"].isin(MODELED_POSITIONS)
            & roster_df["gsis_id"].notna()
            & ~roster_df["is_ir"]
            & ~roster_df["is_taxi"],
            "gsis_id",
        ]
    )
    if not candidates:
        print("  No modeled (QB/RB/WR/TE) players resolved on this roster — skipping.")
        return

    projected = project_players(candidates, bundle, season=season, week=week)
    if projected.empty:
        print("  No projections could be built for this roster.")
        return

    projected = adjust_for_scoring(projected, league.scoring_settings)
    proj_cols = ["player_id", "proj_points", "floor", "ceiling", "explosiveness_score", "risk_tier"]
    board = roster_df.merge(
        projected[proj_cols], left_on="gsis_id", right_on="player_id", how="left",
    )

    objective_col = {"mean": "proj_points", "floor": "floor", "ceiling": "ceiling"}[objective]
    startable = board[
        board["position"].isin(MODELED_POSITIONS)
        & board["proj_points"].notna()
        & ~board["is_ir"]
        & ~board["is_taxi"]
    ].reset_index(drop=True)

    m_slots = modeled_slots(league.roster_positions)
    starters_df, bench_df, empty_slots = optimize_lineup(
        startable, m_slots, objective_col=objective_col
    )

    unmodeled = [s for s in league.roster_positions if s not in m_slots]
    unmodeled_rows = [r for s in unmodeled if (r := _fill_unmodeled_slot(s, roster_df)) is not None]

    lineup = pd.concat([starters_df, pd.DataFrame(unmodeled_rows)], ignore_index=True, sort=False)

    wanted_cols = ["slot", "player_name", "position", "proj_points", "floor", "ceiling",
                   "explosiveness_score", "risk_tier"]
    display_cols = [c for c in wanted_cols if c in lineup.columns]
    print(f"\nRecommended starters (optimizing for {objective}):")
    print(lineup[display_cols].to_string(index=False))
    if empty_slots:
        print(f"\nCould not fill: {empty_slots} (not enough eligible rostered players)")

    current_starters = set(roster_df.loc[roster_df["is_starter"], "player_name"])
    recommended = set(lineup["player_name"]) if not lineup.empty else set()
    bench_but_should_start = recommended - current_starters
    started_but_should_bench = current_starters - recommended
    if bench_but_should_start or started_but_should_bench:
        print("\nSuggested swaps:")
        for name in bench_but_should_start:
            print(f"  START: {name}")
        for name in started_but_should_bench:
            print(f"  BENCH: {name}")
    else:
        print("\nYour current lineup already matches the recommendation.")

    return lineup, bench_df


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

    print(f"Pulling Sleeper leagues for '{args.username}', season {season}...")
    leagues = fetch_all_league_rosters(args.username, season)
    if args.league:
        leagues = [lg for lg in leagues if args.league.lower() in lg.league_name.lower()]
    if not leagues:
        print("No matching Sleeper leagues found.")
        return 1
    print(f"Found {len(leagues)} league(s): {[lg.league_name for lg in leagues]}")

    all_seasons = list(range(args.first_season, last_season + 1))
    if season not in all_seasons:
        all_seasons.append(season)
        all_seasons.sort()
    print(f"\nTraining weekly model on {all_seasons[0]}-{all_seasons[-1]}...")
    bundle = fit_weekly_model(all_seasons)
    print(f"  {len(bundle.panel):,} training rows")

    for league in leagues:
        result = process_league(league, bundle, season, week, args.objective)
        if result is None or args.no_export:
            continue
        lineup, bench_df = result
        out_dir = EXPORT_DIR / "sleeper"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{season}_wk{week:02d}_{_slugify(league.league_name)}.csv"
        pd.concat([lineup.assign(status="starter"), bench_df.assign(status="bench", slot="")],
                  ignore_index=True, sort=False).to_csv(out_path, index=False)
        print(f"\n  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
