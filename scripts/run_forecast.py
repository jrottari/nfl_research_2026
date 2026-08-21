"""End-to-end 2026 PPR fantasy football forecast.

Steps
-----
1. Load nflverse season totals for 2010-2025.
2. Build a panel with lag features.
3. Walk-forward CV on 2016-2024 (train on prior seasons, predict each year).
4. Print a model comparison table.
5. Retrain every model on ALL data through 2025.
6. Forecast 2026 PPR totals for players who finished top-200 in 2025.
7. Export forecast CSV to data/exports/.

Usage
-----
    python scripts/run_forecast.py
    python scripts/run_forecast.py --seasons 2012 2025 --eval-start 2017
    python scripts/run_forecast.py --no-bayes   # skip PyMC if slow
    python scripts/run_forecast.py --top-n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- repo path setup -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR
from nfl_research.forecasting.cv import walk_forward_cv
from nfl_research.forecasting.data import load_multi_season, top_n_in_season
from nfl_research.forecasting.evaluate import make_forecast_df, print_report, summary_table
from nfl_research.forecasting.features import build_panel, feature_matrix, make_forecast_row
from nfl_research.forecasting.models import (
    ExponentialSmoothingModel,
    HierarchicalBayesModel,
    PositionMeanModel,
    RegressionToMeanModel,
    RidgeModel,
    XGBoostModel,
    RandomWalkModel,
    _PYMC_AVAILABLE,
    _XGB_AVAILABLE,
)


def build_models(include_bayes: bool = True) -> list:
    models = [
        RandomWalkModel(),
        PositionMeanModel(),
        ExponentialSmoothingModel(),
        RegressionToMeanModel(),
        RidgeModel(alpha=10.0),
    ]
    if _XGB_AVAILABLE:
        models.append(XGBoostModel())
    if _PYMC_AVAILABLE and include_bayes:
        models.append(HierarchicalBayesModel(use_map=True))
    return models


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="2026 PPR fantasy football forecast")
    p.add_argument("--first-season", type=int, default=2010,
                   help="earliest nflverse season to load (default 2010)")
    p.add_argument("--last-season", type=int, default=2025,
                   help="most recent nflverse season (default 2025)")
    p.add_argument("--eval-start", type=int, default=2016,
                   help="first season to use as a CV eval year (default 2016)")
    p.add_argument("--top-n", type=int, default=200,
                   help="forecast top-N PPR finishers from 2025 (default 200)")
    p.add_argument("--no-bayes", action="store_true",
                   help="skip the PyMC hierarchical model (faster)")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV path (default: data/exports/2026_ppr_forecast.csv)")
    p.add_argument("--no-export", action="store_true", help="skip writing the CSV")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    forecast_season = args.last_season + 1
    all_seasons = list(range(args.first_season, args.last_season + 1))
    eval_seasons = list(range(args.eval_start, args.last_season))  # don't eval on 2025 (no 2026 actuals)

    print(f"\nLoading nflverse season totals {all_seasons[0]}-{all_seasons[-1]}...")
    raw = load_multi_season(all_seasons)
    print(f"  {len(raw):,} player-seasons across {raw['season'].nunique()} seasons")

    print("\nBuilding feature panel...")
    panel = build_panel(raw)
    print(f"  {len(panel):,} panel rows (require >= 1 year of history)")

    models = build_models(include_bayes=not args.no_bayes)
    model_names = [m.name for m in models]
    print(f"\nModels: {model_names}")

    # ---- Walk-forward CV ----------------------------------------------------
    print(f"\nWalk-forward CV  (eval seasons {eval_seasons[0]}-{eval_seasons[-1]})...")
    print(f"  Training restricted to top-{args.top_n} players per season")
    cv_results = walk_forward_cv(
        panel,
        models,
        eval_seasons=eval_seasons,
        top_n_filter=args.top_n,
        top_n_train_filter=args.top_n,
        min_train_seasons=5,
        verbose=True,
    )

    print_report(cv_results)

    tbl = summary_table(cv_results)
    best_model_name = tbl.iloc[0]["Model"]
    print(f"\nBest model for 2026 forecast: {best_model_name}")

    if args.no_export and cv_results.empty:
        print("No CV results and --no-export set. Done.")
        return 0

    # ---- Retrain on all data through last_season ----------------------------
    from nfl_research.forecasting.cv import _filter_train_to_top_n
    print(f"\nRetraining all models on {all_seasons[0]}-{args.last_season} (top-{args.top_n} filter)...")
    train_panel = _filter_train_to_top_n(panel, args.top_n)
    X_all = feature_matrix(train_panel)
    y_all = train_panel["target"]

    retrained: dict[str, object] = {}
    for model in models:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_all, y_all)
            retrained[model.name] = model
            print(f"  [{model.name}] fit on {len(train_panel):,} rows")
        except Exception as exc:
            print(f"  [{model.name}] ERROR during retrain: {exc}")

    if isinstance(retrained.get("xgboost"), XGBoostModel):
        print("\nXGBoost feature importances (top 8):")
        fi = retrained["xgboost"].feature_importance().head(8)
        print(fi.to_string())

    if isinstance(retrained.get("hierarchical_bayes"), HierarchicalBayesModel):
        print("\nHierarchical Bayes position params:")
        print(retrained["hierarchical_bayes"].position_params().to_string(index=False))

    # ---- Build 2026 forecast ------------------------------------------------
    print(f"\nBuilding {forecast_season} forecasts for top-{args.top_n} from {args.last_season}...")

    top_ids = top_n_in_season(
        raw.rename(columns={"fantasy_points_ppr": "fantasy_points_ppr"}),
        args.last_season,
        n=args.top_n,
    )

    forecast_rows = []
    for pid in top_ids:
        frow = make_forecast_row(pid, panel, forecast_season)
        if frow is not None:
            forecast_rows.append(frow)

    if not forecast_rows:
        print("ERROR: could not construct any forecast rows. Check data loading.")
        return 1

    forecast_meta = pd.concat(forecast_rows, ignore_index=True)
    X_fc = feature_matrix(forecast_meta)

    preds: dict[str, np.ndarray] = {}
    for name, model in retrained.items():
        try:
            preds[name] = model.predict(X_fc)
        except Exception as exc:
            print(f"  [{name}] predict error: {exc}")

    fc_df = make_forecast_df(preds, forecast_meta, best_model=best_model_name)

    # Add position rank for the forecast
    fc_df["pos_rank_2025"] = (
        fc_df.groupby("position")["rw_forecast"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    print(f"\n--- {forecast_season} PPR FORECASTS (top 20) ---")
    display_cols = ["player_name", "position", "rw_forecast"]
    if "forecast" in fc_df.columns:
        display_cols += ["forecast", "vs_rw"]
    print(fc_df[display_cols].head(20).to_string(index=False))

    if not args.no_export:
        out_path = args.out or EXPORT_DIR / f"{forecast_season}_ppr_forecast.csv"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fc_df.to_csv(out_path, index=False)
        print(f"\nForecast saved -> {out_path}")

        cv_path = out_path.parent / f"{forecast_season}_ppr_forecast_cv.csv"
        cv_results.to_csv(cv_path, index=False)
        print(f"CV results  saved -> {cv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
