"""Three lineup recommendations per Sleeper league, side by side:

  1. ecr_consensus — maximize average FantasyPros consensus rank (best
     average ECR across starters).
  2. ecr_ceiling   — maximize average best-case ECR (each starter's single
     best individual-expert rank) — the top-end-variance / upside lineup.
  3. xgboost_points — maximize total expected PPR points from this repo's
     own XGBoost weekly forecast model (see reports/weekly_forecast_report.md;
     Ridge is the production default elsewhere, XGBoost is used here
     specifically because that's the variant asked for).

Console preview only (mirrors scripts/ecr_lineup.py / scripts/sleeper_lineup.py);
see scripts/update_dashboard.py to push all three into the Sheets dashboard.

Usage
-----
    python scripts/three_lineups.py
    python scripts/three_lineups.py --league "Daddy's Dynasty"
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
from nfl_research.sleeper.report import build_model_league_board
from nfl_research.sleeper.rosters import fetch_all_league_rosters
from nfl_research.weekly.projection import fit_weekly_model

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Three-lineup comparison per Sleeper league")
    p.add_argument("--username", default=SLEEPER_USERNAME)
    p.add_argument("--league", default=None,
                   help="only process leagues whose name contains this substring")
    p.add_argument("--first-season", type=int, default=2021)
    p.add_argument("--force-refresh", action="store_true",
                   help="re-scrape FantasyPros instead of using the cached ECR snapshot")
    p.add_argument("--no-export", action="store_true")
    return p.parse_args(argv)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "league"


def _print_lineup(label: str, df: pd.DataFrame, metric_col: str, metric_label: str) -> None:
    print(f"\n--- {label} ---")
    if df.empty:
        print("  (no lineup produced)")
        return
    cols = [c for c in
            ["slot", "player_name", "position", "rank", "ecr", "best", "worst",
             "proj_points", "floor", "ceiling", "explosiveness_score"]
            if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"  {metric_label}: {df[metric_col].mean():.2f}"
          + (f"  (total: {df[metric_col].sum():.1f})" if metric_col == "proj_points" else ""))


def main(argv=None) -> int:
    args = parse_args(argv)
    state = api.get_nfl_state()
    season, week = int(state["season"]), int(state["week"])

    print(f"Pulling Sleeper leagues for '{args.username}', season {season}...")
    leagues = fetch_all_league_rosters(args.username, season)
    if args.league:
        leagues = [lg for lg in leagues if args.league.lower() in lg.league_name.lower()]
    leagues = [lg for lg in leagues if not lg.players.empty]
    if not leagues:
        print("No matching (drafted) Sleeper leagues found.")
        return 1
    print(f"Found {len(leagues)} league(s): {[lg.league_name for lg in leagues]}")

    print("\nLoading FantasyPros weekly ECR...")
    ecr = load_weekly_ecr(force_refresh=args.force_refresh)

    print(f"Training XGBoost weekly model on {args.first_season}-{season}...")
    bundle = fit_weekly_model(list(range(args.first_season, season + 1)), model_type="xgboost")
    print(f"  {len(bundle.panel):,} training rows")

    for league in leagues:
        print(f"\n{'=' * 70}\n{league.league_name}  (team: {league.team_name})\n{'=' * 70}")

        consensus_df, _b1, _e1, _u1 = build_league_board(league, ecr, "consensus")
        ceiling_df, _b2, _e2, _u2 = build_league_board(league, ecr, "ceiling")
        points_df, bench_df, empty_slots = build_model_league_board(
            league, bundle, season, week, objective_col="proj_points"
        )

        _print_lineup("1. ECR consensus (best average rank)", consensus_df, "rank", "Avg rank")
        _print_lineup("2. ECR ceiling (top-end variance)", ceiling_df, "best", "Avg best-case rank")
        _print_lineup("3. XGBoost expected points", points_df, "proj_points", "Avg exp. points/starter")
        if empty_slots:
            print(f"\n  XGBoost lineup could not fill: {empty_slots}")

        if args.no_export:
            continue
        out_dir = EXPORT_DIR / "three_lineups" / f"week_{week}"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(league.league_name)
        for tag, df in (("ecr_consensus", consensus_df), ("ecr_ceiling", ceiling_df),
                        ("xgboost_points", points_df)):
            if df.empty:
                continue
            out_path = out_dir / f"{season}_wk{week:02d}_{slug}_{tag}.csv"
            df.to_csv(out_path, index=False)
        print(f"\n  -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
