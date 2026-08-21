"""Within-season weekly PPR forecast pipeline.

Loads multi-season weekly data, builds rolling features, runs walk-forward CV
by week, and produces a next-week forecast for every rostered player.

Usage
-----
    # Full CV + next-week forecast (current season)
    python scripts/run_weekly_forecast.py

    # Specify seasons and upcoming week
    python scripts/run_weekly_forecast.py --eval-seasons 2023 2024 --forecast-week 8

    # Skip XGBoost for speed
    python scripts/run_weekly_forecast.py --no-xgb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR
from nfl_research.forecasting.data import load_multi_season
from nfl_research.weekly.cv import walk_forward_weekly_cv
from nfl_research.weekly.data import defense_vs_position, load_multi_season_weekly
from nfl_research.weekly.evaluate import print_weekly_report, weekly_summary_table
from nfl_research.weekly.features import (
    build_weekly_panel,
    make_upcoming_row,
    weekly_feature_matrix,
)
from nfl_research.weekly.models import WEEKLY_MODELS, _XGB_AVAILABLE


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Weekly PPR fantasy football forecast")
    p.add_argument("--first-season", type=int, default=2015,
                   help="earliest season to load for training (default 2015)")
    p.add_argument("--last-season", type=int, default=2025,
                   help="most recent completed season (default 2025)")
    p.add_argument("--eval-seasons", type=int, nargs="+", default=None,
                   help="seasons to include in CV (default: last 3)")
    p.add_argument("--forecast-week", type=int, default=None,
                   help="upcoming week number to forecast (default: max week + 1)")
    p.add_argument("--forecast-season", type=int, default=None,
                   help="season to forecast (default: last-season)")
    p.add_argument("--top-n", type=int, default=150,
                   help="score only top-N prior-season PPR players (default 150)")
    p.add_argument("--no-xgb", action="store_true", help="skip XGBoost (faster)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    all_seasons = list(range(args.first_season, args.last_season + 1))
    eval_seasons = args.eval_seasons or all_seasons[-3:]
    fc_season    = args.forecast_season or args.last_season

    print(f"\nLoading weekly data {all_seasons[0]}–{all_seasons[-1]}...")
    weekly = load_multi_season_weekly(all_seasons)
    print(f"  {len(weekly):,} player-game rows")

    print("Loading season totals for prior-season features...")
    season_df = load_multi_season(all_seasons)

    print("Computing defense-vs-position matchup table...")
    defense_df = defense_vs_position(weekly)

    print("Building weekly panel with rolling features...")
    panel = build_weekly_panel(weekly, defense_df=defense_df, prior_season_df=season_df)
    print(f"  {len(panel):,} panel rows (≥1 prior game)")

    models = WEEKLY_MODELS()
    if args.no_xgb:
        models = [m for m in models if "xgboost" not in m.name]
    print(f"\nModels: {[m.name for m in models]}")

    # ---- Walk-forward weekly CV ---------------------------------------------
    print(f"\nWalk-forward weekly CV  (seasons {eval_seasons})...")
    cv = walk_forward_weekly_cv(
        panel,
        models,
        eval_seasons=eval_seasons,
        top_n_filter=args.top_n,
        min_prior_games=2,
        verbose=True,
    )

    if cv.empty:
        print("No CV results produced — check data coverage for eval seasons.")
        return 1

    print_weekly_report(cv)

    tbl = weekly_summary_table(cv)
    best_name = tbl.iloc[0]["Model"]
    print(f"\nBest weekly model: {best_name}")

    # ---- Retrain on all data and forecast upcoming week ---------------------
    max_week_in_data = int(panel[panel["season"] == fc_season]["week"].max()) if fc_season in panel["season"].values else 0
    forecast_week    = args.forecast_week or (max_week_in_data + 1 if max_week_in_data > 0 else 1)

    if forecast_week > 18:
        print(f"\nForecast week {forecast_week} > 18 — season appears complete. No upcoming forecast.")
    else:
        print(f"\nRetraining on all data and forecasting season {fc_season} week {forecast_week}...")

        X_all = weekly_feature_matrix(panel)
        y_all = panel["target"]

        retrained: dict = {}
        for model in models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_all, y_all)
                retrained[model.name] = model
            except Exception as exc:
                print(f"  [{model.name}] retrain error: {exc}")

        # Build forecast rows for active players
        active_players = (
            panel[
                (panel["season"] == fc_season) &
                (panel["week"] < forecast_week)
            ]["player_id"].unique()
        )

        fc_rows = []
        for pid in active_players:
            row = make_upcoming_row(
                pid,
                weekly,
                upcoming_week=forecast_week,
                upcoming_season=fc_season,
                defense_df=defense_df,
            )
            if row is not None:
                fc_rows.append(row)

        if not fc_rows:
            print("No forecast rows built.")
        else:
            fc_meta = pd.concat(fc_rows, ignore_index=True)
            X_fc    = weekly_feature_matrix(fc_meta)

            preds: dict[str, np.ndarray] = {}
            for name, model in retrained.items():
                try:
                    preds[name] = model.predict(X_fc)
                except Exception as exc:
                    print(f"  [{name}] predict error: {exc}")

            fc_out = fc_meta[["player_id", "player_name", "position"]].copy()
            fc_out["season_avg_ppg"] = fc_meta["ppr_season_avg"].round(1)
            fc_out["last_game_ppr"]  = fc_meta["ppr_lag1"].round(1)
            fc_out["opp_strength"]   = fc_meta["opp_ppr_allowed_avg"].round(1)
            for name, arr in preds.items():
                fc_out[f"proj_{name}"] = arr.round(1)

            best_col = f"proj_{best_name}"
            if best_col in fc_out.columns:
                fc_out = fc_out.sort_values(best_col, ascending=False)

            print(f"\n--- Week {forecast_week} PPR Projections (top 25) ---")
            print(fc_out.head(25).to_string(index=False))

            if not args.no_export:
                out_path = args.out or EXPORT_DIR / f"{fc_season}_wk{forecast_week:02d}_projections.csv"
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fc_out.to_csv(out_path, index=False)

                cv_path = out_path.parent / f"weekly_cv_results.csv"
                cv.to_csv(cv_path, index=False)
                print(f"\nProjections -> {out_path}")
                print(f"CV results  -> {cv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
