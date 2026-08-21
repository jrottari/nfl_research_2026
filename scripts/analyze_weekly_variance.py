"""Concludes the within-season explosiveness/variance analysis.

Answers three questions with real historical data (2021-2025, cached offline):

1. Is week-to-week volatility (coefficient of variation, boom rate) a
   persistent player trait, or just noise?  (split-half correlation)
2. Does it explain part of what the point-forecast models miss?
   (correlation between pre-game explosiveness and that week's abs error)
3. Can floor/ceiling bands calibrated from historical residuals, split by
   explosiveness tercile, generalize out-of-sample?  (fit on 2023-2024,
   check coverage on the held-out 2025 season)

Usage
-----
    python scripts/analyze_weekly_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.config import EXPORT_DIR
from nfl_research.forecasting.data import load_multi_season
from nfl_research.weekly.cv import walk_forward_weekly_cv
from nfl_research.weekly.data import defense_vs_position, load_multi_season_weekly
from nfl_research.weekly.features import build_weekly_panel
from nfl_research.weekly.models import RidgeWeeklyModel, SeasonAvgModel, XGBoostWeeklyModel
from nfl_research.weekly.variance import (
    add_variance_features,
    calibrate_bands,
    coverage_check,
    validate_variance_persistence,
    validate_variance_predicts_error,
)


def main() -> int:
    seasons = list(range(2021, 2026))

    print(f"Loading weekly data {seasons[0]}-{seasons[-1]}...")
    weekly = load_multi_season_weekly(seasons)
    season_df = load_multi_season(seasons)
    defense_df = defense_vs_position(weekly)

    print("Building panel + variance features...")
    panel = build_weekly_panel(weekly, defense_df=defense_df, prior_season_df=season_df)
    panel = add_variance_features(panel)
    print(f"  {len(panel):,} panel rows")

    print("\n=== 1. Persistence: does early-season CV predict late-season CV? ===")
    persistence = validate_variance_persistence(panel, min_games=8)
    print(persistence.to_string(index=False))

    print("\n=== 2. Walk-forward CV (ridge, xgboost, season_avg) ===")
    models = [RidgeWeeklyModel(alpha=5.0), XGBoostWeeklyModel(), SeasonAvgModel()]
    cv = walk_forward_weekly_cv(
        panel, models, eval_seasons=[2023, 2024, 2025], top_n_filter=150, min_prior_games=2, verbose=True
    )

    print("\n=== 3. Does pre-game explosiveness correlate with forecast error? ===")
    err_corr = validate_variance_predicts_error(cv, panel)
    print(err_corr.to_string(index=False))

    print("\n=== 4. Floor/ceiling calibration: fit 2023-2024, test coverage on 2025 ===")
    cv_ridge = cv[cv["model"] == "ridge_weekly"]
    cv_fit = cv_ridge[cv_ridge["eval_season"].isin([2023, 2024])]
    cv_test = cv_ridge[cv_ridge["eval_season"] == 2025]

    bands = calibrate_bands(cv_fit, panel)
    print("\nBands fit on 2023-2024 (p20/p80 residual offsets, PPR points):")
    print(bands.to_string(index=False))

    cov_in = coverage_check(cv_fit, panel, bands)
    cov_out = coverage_check(cv_test, panel, bands)
    print(f"\nCoverage in-sample (2023-2024): {cov_in:.3f}")
    print(f"Coverage out-of-sample (2025):  {cov_out:.3f}  (target ~0.60)")

    print("\n=== 5. Production bands: fit on all 3 seasons (2023-2025) ===")
    bands_full = calibrate_bands(cv_ridge, panel)
    print(bands_full.to_string(index=False))

    out_dir = EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bands_full.to_csv(out_dir / "weekly_variance_bands.csv", index=False)
    persistence.to_csv(out_dir / "weekly_variance_persistence.csv", index=False)
    err_corr.to_csv(out_dir / "weekly_variance_error_correlation.csv", index=False)
    print(f"\nWrote calibration artifacts -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
