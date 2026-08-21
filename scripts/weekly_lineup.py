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

For a specific Sleeper league's roster instead of the generic all-players
board, see ``scripts/sleeper_lineup.py``, which reuses this same trained
model via ``nfl_research.weekly.projection``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR
from nfl_research.weekly.data import load_multi_season_weekly
from nfl_research.weekly.projection import fit_weekly_model, project_players


def infer_current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Weekly PPR lineup helper")
    p.add_argument("--first-season", type=int, default=2021,
                   help="earliest training season (default 2021)")
    p.add_argument("--last-season", type=int, default=None,
                   help="most recent completed season (default: auto)")
    p.add_argument("--season", type=int, default=None, help="season to forecast (default: current)")
    p.add_argument("--week", type=int, default=None,
                   help="upcoming week to forecast (default: next unplayed)")
    p.add_argument("--top-n-fallback", type=int, default=150,
                   help="prior-season top-N cutoff for the cold-start fallback (default 150)")
    p.add_argument("--compare", nargs=2, metavar=("PLAYER_A", "PLAYER_B"), default=None,
                   help="print a head-to-head start/sit comparison for two players")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-export", action="store_true")
    return p.parse_args(argv)


def build_forecast_board(
    season: int, week: int, first_season: int, last_season: int, top_n_fallback: int
):
    all_seasons = list(range(first_season, last_season + 1))
    if season not in all_seasons:
        all_seasons.append(season)
        all_seasons.sort()

    print(f"Loading weekly data {all_seasons[0]}-{all_seasons[-1]} (season {season} live)...")
    bundle = fit_weekly_model(all_seasons)
    print(f"  {len(bundle.panel):,} training rows")

    current = bundle.weekly[(bundle.weekly["season"] == season) & (bundle.weekly["week"] < week)]
    active_ids = set(current["player_id"].unique())
    prior_totals = bundle.season_totals[bundle.season_totals["season"] == season - 1]
    all_ids = active_ids | set(prior_totals["player_id"])

    print(f"Building week {week} forecast rows for {len(active_ids)} active players...")
    fc = project_players(all_ids, bundle, season=season, week=week, prior_top_n=top_n_fallback)
    if fc.empty:
        return fc

    cols = [
        "player_name", "position", "season", "week", "proj_points", "floor", "ceiling",
        "explosiveness_score", "risk_tier", "boom_rate5", "data_source",
    ]
    ordered = fc[cols].sort_values(["position", "proj_points"], ascending=[True, False])
    return ordered.reset_index(drop=True)


def print_compare(board, name_a: str, name_b: str) -> None:
    def _find(name: str):
        matches = board[board["player_name"].str.contains(name, case=False, na=False)]
        if matches.empty:
            print(f"  No match found for '{name}'")
            return None
        return matches.iloc[0]

    a, b = _find(name_a), _find(name_b)
    if a is None or b is None:
        return

    import pandas as pd

    print("\n--- Start/Sit Comparison ---")
    print(pd.DataFrame([a, b])[["player_name", "position", "proj_points", "floor", "ceiling",
                                 "explosiveness_score", "risk_tier"]].to_string(index=False))

    higher_floor = a if a["floor"] >= b["floor"] else b
    higher_ceiling = a if a["ceiling"] >= b["ceiling"] else b
    print(f"\nHigher floor (protect a lead):  "
          f"{higher_floor['player_name']} ({higher_floor['floor']} pts)")
    print(f"Higher ceiling (need upside):   "
          f"{higher_ceiling['player_name']} ({higher_ceiling['ceiling']} pts)")
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
        print("No forecast rows produced - check that the season/week has data available.")
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
