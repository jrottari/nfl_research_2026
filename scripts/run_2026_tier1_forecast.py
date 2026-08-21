"""Forecast 2026 PPR fantasy points with the tier-1 XGBoost ablation model.

Fits ``XGBoostModel(max_tier=1)`` — lag/age/draft-capital features plus
FantasyPros preseason ECR and ffverse opportunity/usage features — on every
real player-season where tier-1 coverage actually exists (2021-2025, the same
restricted panel used in the Run 0 ablation and ``extract_model_weights.py``),
then predicts 2026 PPR totals for every returning player using their actual
2025 (and earlier) history plus the live 2026 preseason ECR pull.

Players whose only NFL season on record is 2025 (true rookies) have no lag
history and are out of scope for this lag-based pipeline, same as every other
model in this project — see RESULTS.md.

Usage
-----
    python scripts/run_2026_tier1_forecast.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from nfl_research.config import EXPORT_DIR
from nfl_research.forecasting.data import load_multi_season
from nfl_research.forecasting.features import (
    build_historical_wide_panel,
    build_panel,
    build_wide_panel,
    make_forecast_row,
)
from nfl_research.forecasting.models import XGBoostModel
from nfl_research.forecasting.source_cache import load_tier_sources

FIRST_SEASON = 1999
LAST_SEASON = 2025
FORECAST_SEASON = 2026
ECR_FIRST_SEASON = 2021  # tier-1's real (non-synthetic) coverage start


def main() -> int:
    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))

    print(f"Loading season totals {seasons[0]}-{seasons[-1]}...")
    totals = load_multi_season(seasons)
    base = build_panel(totals)
    print(f"  base panel: {len(base):,} player-seasons (>=1 year of history)")

    print("Loading tier-1 sources (cached; includes today's live ECR pull)...")
    sources = load_tier_sources(seasons, max_tier=1, include_pbp=False)

    # ---- Training panel: real tier-1 coverage only (2021-2025) ----------------
    print("Building tier-1 training panel...")
    wide = build_historical_wide_panel(base, max_tier=1, sources=sources)
    train = wide[pd.to_numeric(wide["season"]) >= ECR_FIRST_SEASON].dropna(subset=["target"])
    print(f"  training rows: {len(train):,} ({ECR_FIRST_SEASON}-{LAST_SEASON})")

    print("Fitting XGBoostModel(max_tier=1)...")
    model = XGBoostModel(max_tier=1)
    model.fit(train.drop(columns=["target"]), train["target"])
    print("  top feature importances:")
    print(model.feature_importance().head(10).to_string())

    # ---- Forecast rows for 2026: every player active in 2024 or 2025 ----------
    # Excludes players whose two most recent seasons on record are >2 years apart:
    # build_panel()'s lag1/lag2/lag3 are positional (the player's own previous ROWS),
    # not calendar-aware, so a multi-year gap silently blends a stale season into
    # "last year." In practice this catches a handful of nflverse data anomalies
    # (e.g. a since-retired veteran with one erroneous late season row) rather than
    # real, informative comebacks.
    seasons_by_player = base.groupby("player_id")["season"].apply(lambda s: sorted(s.tolist()))
    latest_season = base.groupby("player_id")["season"].max()
    candidate_ids = latest_season[latest_season >= LAST_SEASON - 1].index
    gap_ok = candidate_ids.map(
        lambda pid: len(seasons_by_player[pid]) < 2
        or seasons_by_player[pid][-1] - seasons_by_player[pid][-2] <= 2
    )
    eligible_ids = candidate_ids[gap_ok].tolist()
    n_excluded = len(candidate_ids) - len(eligible_ids)
    if n_excluded:
        print(f"  excluding {n_excluded} player(s) with a >2-season gap (likely data anomalies)")
    print(f"\nBuilding {FORECAST_SEASON} forecast rows for {len(eligible_ids):,} players...")

    rows = [make_forecast_row(pid, base, FORECAST_SEASON) for pid in eligible_ids]
    rows = [r for r in rows if r is not None]
    forecast_meta = pd.concat(rows, ignore_index=True)

    # team is informational only — not a model feature — carry it through separately
    team_lookup = (
        base.sort_values("season").groupby("player_id")["team"].last()
        if "team" in base.columns
        else pd.Series(dtype=object)
    )

    print("Attaching tier-0/tier-1 features (age, draft capital, ECR, opportunity, usage)...")
    forecast_wide = build_wide_panel(
        forecast_meta,
        max_tier=1,
        as_of=datetime(FORECAST_SEASON, 8, 1),
        sources=sources,
    )

    matched_ecr = forecast_wide["ecr_rank"].notna().sum() if "ecr_rank" in forecast_wide else 0
    print(f"  {matched_ecr:,}/{len(forecast_wide):,} players matched to a 2026 preseason ECR rank")

    print("Predicting...")
    predicted = model.predict(forecast_wide)

    out = pd.DataFrame(
        {
            "player_id": forecast_wide["player_id"],
            "player_name": forecast_wide["player_name"],
            "position": forecast_wide["position"],
            "team": forecast_wide["player_id"].map(team_lookup),
            "predicted_ppr_2026": predicted.round(1),
            "points_ppr_2025": forecast_wide["points_ppr_lag1"].round(1),
            "age_2026": forecast_wide.get("age"),
            "ecr_rank_2026": forecast_wide.get("ecr_rank"),
            "ecr_pos_rank_2026": forecast_wide.get("ecr_pos_rank"),
        }
    )
    out = out.sort_values("predicted_ppr_2026", ascending=False).reset_index(drop=True)
    out.insert(0, "overall_rank", out.index + 1)
    out["position_rank"] = out.groupby("position")["predicted_ppr_2026"].rank(
        ascending=False, method="min"
    ).astype(int)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / f"{FORECAST_SEASON}_ppr_forecast_tier1_xgboost.csv"
    out.to_csv(out_path, index=False)

    print(f"\n--- TOP 25, {FORECAST_SEASON} tier-1 XGBoost forecast ---")
    print(
        out[
            ["overall_rank", "player_name", "position", "team", "predicted_ppr_2026", "ecr_rank_2026"]
        ]
        .head(25)
        .to_string(index=False)
    )
    print(f"\nForecast saved -> {out_path}  ({len(out):,} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
