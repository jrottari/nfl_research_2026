"""Weekly starting lineup, optimized against live FantasyPros ECR (rather
than this project's own Ridge model in ``scripts/weekly_lineup.py`` /
``scripts/sleeper_lineup.py``).

Covers every slot type across your Sleeper leagues, including K/DST/IDP_FLEX
that the model-based optimizer can't project. See
``nfl_research.ecr.optimize`` for the cross-position normalization this
relies on for FLEX-type slots — that's the piece most worth workshopping.

Usage
-----
    # All leagues, consensus ECR (minimize average starting rank)
    python scripts/ecr_lineup.py

    # One league, protecting a lead (worst-case-expert rank per starter)
    python scripts/ecr_lineup.py --league "Daddy's Dynasty" --mode floor

    # Chasing points (best-case-expert rank per starter)
    python scripts/ecr_lineup.py --mode ceiling
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR, SLEEPER_USERNAME
from nfl_research.ecr.fetch import load_weekly_ecr
from nfl_research.ecr.report import build_league_board
from nfl_research.sleeper import api
from nfl_research.sleeper.rosters import LeagueRoster, fetch_all_league_rosters

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FantasyPros weekly-ECR lineup optimizer")
    p.add_argument("--username", default=SLEEPER_USERNAME,
                   help=f"Sleeper username or numeric user_id (default: {SLEEPER_USERNAME})")
    p.add_argument("--season", type=int, default=None, help="season (default: current, per Sleeper)")
    p.add_argument("--mode", choices=["consensus", "ceiling", "floor"], default="consensus",
                   help="consensus = best average rank; ceiling = chase upside; "
                        "floor = protect against a bad week (default: consensus)")
    p.add_argument("--league", default=None,
                   help="only process leagues whose name contains this substring")
    p.add_argument("--force-refresh", action="store_true",
                   help="re-scrape FantasyPros instead of using the cached snapshot")
    p.add_argument("--no-export", action="store_true")
    return p.parse_args(argv)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "league"


def process_league(league: LeagueRoster, ecr: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    print(f"\n{'=' * 70}\n{league.league_name}  (team: {league.team_name})\n{'=' * 70}")

    if league.players.empty:
        print("  Roster is empty (league likely hasn't drafted yet) — skipping.")
        return None

    starters_df, bench_df, empty_slots, unmatched = build_league_board(league, ecr, mode)
    if not unmatched.empty:
        names = ", ".join(unmatched["player_name"].tolist())
        print(f"  No ECR match (excluded from recommendation): {names}")

    wanted_cols = ["slot", "player_name", "position", "rank", "ecr", "best", "worst", "sd"]
    display_cols = [c for c in wanted_cols if c in starters_df.columns]
    print(f"\nRecommended starters (optimizing for {mode}):")
    if not starters_df.empty:
        print(starters_df[display_cols].sort_values("slot").to_string(index=False))
    if empty_slots:
        print(f"\nCould not fill: {empty_slots} (no ECR-matched eligible player left)")

    if not starters_df.empty:
        avg_rank = starters_df["rank"].mean()
        print(f"\nAverage positional ECR rank of recommended starters: {avg_rank:.2f}")

    current_starters = set(league.players.loc[league.players["is_starter"], "player_name"])
    recommended = set(starters_df["player_name"]) if not starters_df.empty else set()
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

    return starters_df, bench_df


def main(argv=None) -> int:
    args = parse_args(argv)
    state = api.get_nfl_state()
    season = args.season or int(state["season"])
    week = int(state["week"])

    print(f"Pulling Sleeper leagues for '{args.username}', season {season}...")
    leagues = fetch_all_league_rosters(args.username, season)
    if args.league:
        leagues = [lg for lg in leagues if args.league.lower() in lg.league_name.lower()]
    if not leagues:
        print("No matching Sleeper leagues found.")
        return 1
    print(f"Found {len(leagues)} league(s): {[lg.league_name for lg in leagues]}")

    print(f"\nLoading FantasyPros weekly ECR (week {week})...")
    ecr = load_weekly_ecr(force_refresh=args.force_refresh)
    print(f"  {len(ecr):,} ranked players across {ecr['position'].nunique()} positions")

    for league in leagues:
        result = process_league(league, ecr, args.mode)
        if result is None or args.no_export:
            continue
        starters_df, bench_df = result
        out_dir = EXPORT_DIR / "ecr_lineups" / f"week_{week}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{season}_wk{week:02d}_{_slugify(league.league_name)}_{args.mode}.csv"
        pd.concat(
            [starters_df.assign(status="starter"), bench_df.assign(status="bench", slot="")],
            ignore_index=True, sort=False,
        ).to_csv(out_path, index=False)
        print(f"\n  -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
