"""Push this week's three lineup recommendations into the Google Sheets
dashboard — see ``scripts/three_lineups.py`` for what the three are
(ecr_consensus, ecr_ceiling, xgboost_points) and why.

Requires one-time OAuth setup — see ``nfl_research.sheets.auth`` /
project memory for the Google Cloud Console steps to get a
``client_secret.json``. The first run opens a browser consent screen.

Writes, in one persistent spreadsheet:
  - One tab per league: all three lineups stacked (tagged by a ``lineup``
    column) — one row per recommended starter, with whichever columns that
    lineup type doesn't use (e.g. ``rank`` for the XGBoost lineup) left
    blank.
  - One "<League> - Roster" tab per league: every rostered player (starters
    and bench alike, IR/taxi included) with ECR (rank/ecr/best/worst/sd) and
    the XGBoost projection (proj_points/floor/ceiling/explosiveness_score)
    side by side — not just whoever the 3 lineups picked.
  - "All Leagues Summary": one row per league x lineup type, with that
    lineup's headline metric (avg starting rank for the two ECR lineups,
    avg/total expected points for the XGBoost lineup).
  - "History": append-only log, one row per league per week, with all
    three headline metrics and swap counts — the source data for a Sheets
    trend chart across the season.
  - "Player Overview": one row per unique rostered player, with
    ecr_consensus_rank/ecr_best_rank/ecr_worst_rank/xgboost_proj_points/
    xgboost_ceiling_points/last_game_points/last_game, one checkbox column
    per league (currently started — blank if not on that roster), and two
    live-formula columns (num_leagues_rostered/num_leagues_starting) that
    recompute in Sheets as you toggle checkboxes by hand.
  - "Correlation Score": raw/max/normalized score for how consistently you
    start players you own in multiple leagues — live formulas against
    Player Overview's checkbox-driven columns (see
    ``nfl_research.sheets.player_overview`` for the exact definition).
  - "ECR Range by League": each league's average best/consensus/worst-case
    starting ECR among its *currently started* players (live AVERAGEIF
    formulas against the Player Overview checkboxes — also updates as you
    toggle them). Lower ECR = better.
  - Charts (native Sheets charts, rebuilt fresh every run):
    "Trend - Consensus Rank" / "Trend - XGBoost Points" (one line per
    league, pivoted from the full History log), "Chart Source - Rank" /
    "Chart Source - Points" (this week's cross-league comparison bar
    charts), "Player Overview - Top Starts" (bar chart of which players are
    started/rostered across the most leagues), and "ECR Range by League"'s
    own grouped bar chart (best/consensus/worst per league).

Usage
-----
    python scripts/update_dashboard.py
    python scripts/update_dashboard.py --league "Daddy's Dynasty"
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

from nfl_research.config import SLEEPER_USERNAME
from nfl_research.ecr.fetch import load_weekly_ecr
from nfl_research.ecr.report import build_league_board
from nfl_research.roster_board import build_full_roster_board
from nfl_research.sheets.auth import get_client
from nfl_research.sheets.dashboard import (
    SUMMARY_TAB,
    append_history,
    get_or_create_spreadsheet,
    push_dataframe,
)
from nfl_research.sheets.player_overview import (
    build_player_overview_frame,
    push_correlation_score,
    push_ecr_range_summary,
    push_leaderboard_chart,
    push_player_overview,
)
from nfl_research.sheets.trends import push_summary_charts, push_trend_chart, read_history
from nfl_research.sleeper import api
from nfl_research.sleeper.report import build_model_league_board
from nfl_research.sleeper.rosters import fetch_all_league_rosters
from nfl_research.weekly.projection import fit_weekly_model

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

TAB_COLS = ["lineup", "slot", "player_name", "position",
            "rank", "ecr", "best", "worst", "proj_points", "floor", "ceiling"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Push the three lineup variants to the Sheets dashboard")
    p.add_argument("--username", default=SLEEPER_USERNAME)
    p.add_argument("--league", default=None,
                   help="only process leagues whose name contains this substring")
    p.add_argument("--first-season", type=int, default=2021)
    p.add_argument("--force-refresh", action="store_true",
                   help="re-scrape FantasyPros instead of using the cached ECR snapshot")
    return p.parse_args(argv)


def _swap_count(recommended_df: pd.DataFrame, current_starters: set[str]) -> int:
    if recommended_df.empty:
        return 0
    recommended = set(recommended_df["player_name"])
    return len(recommended - current_starters) + len(current_starters - recommended)


def _roster_tab_name(league_name: str) -> str:
    suffix = " - Roster"
    return league_name[: 100 - len(suffix)] + suffix


def main(argv=None) -> int:
    args = parse_args(argv)
    state = api.get_nfl_state()
    season, week = int(state["season"]), int(state["week"])
    run_date = dt.date.today().isoformat()

    print(f"Pulling Sleeper leagues for '{args.username}', season {season}...")
    leagues = fetch_all_league_rosters(args.username, season)
    if args.league:
        leagues = [lg for lg in leagues if args.league.lower() in lg.league_name.lower()]
    leagues = [lg for lg in leagues if not lg.players.empty]
    if not leagues:
        print("No matching (drafted) Sleeper leagues found.")
        return 1

    print("Loading FantasyPros weekly ECR...")
    ecr = load_weekly_ecr(force_refresh=args.force_refresh)

    print(f"Training XGBoost weekly model on {args.first_season}-{season}...")
    bundle = fit_weekly_model(list(range(args.first_season, season + 1)), model_type="xgboost")
    print(f"  {len(bundle.panel):,} training rows")

    print("Authenticating to Google Sheets (first run opens a browser)...")
    client = get_client()
    spreadsheet = get_or_create_spreadsheet(client)
    print(f"Dashboard: {spreadsheet.url}")

    summary_rows, history_rows, roster_boards = [], [], []

    for league in leagues:
        print(f"\n{league.league_name}...")
        current_starters = set(league.players.loc[league.players["is_starter"], "player_name"])

        consensus_df, _b1, empty1, _u1 = build_league_board(league, ecr, "consensus")
        ceiling_df, _b2, empty2, _u2 = build_league_board(league, ecr, "ceiling")
        points_df, _bench_df, empty3 = build_model_league_board(
            league, bundle, season, week, objective_col="proj_points"
        )

        lineups = {
            "ecr_consensus": (consensus_df, empty1),
            "ecr_ceiling": (ceiling_df, empty2),
            "xgboost_points": (points_df, empty3),
        }

        tab_blocks = []
        for tag, (df, empty_slots) in lineups.items():
            if df.empty:
                continue
            block = df[[c for c in TAB_COLS if c in df.columns]].copy()
            block.insert(0, "lineup", tag)
            tab_blocks.append(block)

            metric_col, metric = (
                ("rank", df["rank"].mean()) if "rank" in df.columns
                else ("proj_points", df["proj_points"].mean())
            )
            summary_rows.append({
                "league": league.league_name, "lineup": tag,
                "avg_metric": round(metric, 2),
                "metric_name": metric_col,
                "total_proj_points": round(df["proj_points"].sum(), 1)
                    if "proj_points" in df.columns else None,
                "unfilled_slots": ", ".join(empty_slots) if empty_slots else "",
            })

        tab_df = pd.concat(tab_blocks, ignore_index=True, sort=False) if tab_blocks else pd.DataFrame()
        if not tab_df.empty:
            tab_df.insert(0, "week", week)
        push_dataframe(spreadsheet, league.league_name[:99], tab_df)

        roster_df = build_full_roster_board(league, ecr, bundle, season, week)
        roster_boards.append(roster_df)
        tab_roster_df = roster_df.copy()
        if not tab_roster_df.empty:
            tab_roster_df.insert(0, "week", week)
        push_dataframe(spreadsheet, _roster_tab_name(league.league_name), tab_roster_df)

        history_rows.append({
            "date": run_date, "season": season, "week": week,
            "league": league.league_name,
            "avg_rank_consensus": round(consensus_df["rank"].mean(), 2)
                if not consensus_df.empty else None,
            "avg_best_rank_ceiling": round(ceiling_df["best"].mean(), 2)
                if not ceiling_df.empty else None,
            "avg_exp_points_xgboost": round(points_df["proj_points"].mean(), 2)
                if not points_df.empty else None,
            "total_exp_points_xgboost": round(points_df["proj_points"].sum(), 1)
                if not points_df.empty else None,
            "swap_count_consensus": _swap_count(consensus_df, current_starters),
            "swap_count_ceiling": _swap_count(ceiling_df, current_starters),
            "swap_count_xgboost": _swap_count(points_df, current_starters),
        })
        print(f"  -> tab '{league.league_name}' updated "
              f"({len(tab_blocks)}/3 lineups produced), "
              f"roster tab: {len(roster_df)} players")

    summary_df = pd.DataFrame(summary_rows)
    push_dataframe(spreadsheet, SUMMARY_TAB, summary_df)
    append_history(spreadsheet, pd.DataFrame(history_rows))

    overview_frame, league_cols = build_player_overview_frame(leagues, roster_boards)
    n_overview = push_player_overview(spreadsheet, overview_frame, league_cols)
    print(f"\n'Player Overview' tab updated: {n_overview} unique players across {len(leagues)} leagues")

    n_top = push_leaderboard_chart(spreadsheet, overview_frame, league_cols)
    print(f"'Player Overview - Top Starts' chart updated: top {n_top} players")

    n_ecr_range = push_ecr_range_summary(spreadsheet, overview_frame, league_cols)
    print(f"'ECR Range by League' tab + chart updated ({n_ecr_range} leagues)")

    push_correlation_score(spreadsheet, overview_frame, league_cols)
    print("'Correlation Score' tab updated")

    print("\nBuilding trend charts from the full History log...")
    full_history = read_history(spreadsheet)
    n_weeks = push_trend_chart(
        spreadsheet, full_history, "avg_rank_consensus", "Trend - Consensus Rank",
        "Avg Starting Consensus Rank by Week (lower = better)", reverse_y=True,
    )
    push_trend_chart(
        spreadsheet, full_history, "avg_exp_points_xgboost", "Trend - XGBoost Points",
        "Avg XGBoost Projected Points by Week",
    )
    print(f"  {n_weeks} week(s) of history charted")

    push_summary_charts(spreadsheet, summary_df)
    print("This week's Summary bar charts updated")

    print(f"\nDone. Open the dashboard: {spreadsheet.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
