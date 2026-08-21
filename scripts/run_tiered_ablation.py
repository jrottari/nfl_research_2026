"""Fetch/cache tier sources, build a point-in-time panel, and run ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_research.forecasting.data import load_multi_season  # noqa: E402
from nfl_research.forecasting.evaluation import ablation_table, run_ablation  # noqa: E402
from nfl_research.forecasting.features import (  # noqa: E402
    build_historical_wide_panel,
    build_panel,
)
from nfl_research.forecasting.source_cache import load_tier_sources  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--max-tier", type=int, choices=range(5), default=4)
    parser.add_argument("--no-pbp", action="store_true", help="skip the large play-by-play source")
    parser.add_argument("--min-train-seasons", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seasons = list(range(args.first_season, args.last_season + 1))
    print(f"Loading season totals {seasons[0]}-{seasons[-1]}...")
    totals = load_multi_season(seasons)
    base = build_panel(totals)
    print(f"Base panel: {len(base):,} player-seasons")

    print(f"Fetching/caching sources through Tier {args.max_tier}...")
    sources = load_tier_sources(
        seasons,
        max_tier=args.max_tier,
        include_pbp=not args.no_pbp,
    )
    wide = build_historical_wide_panel(base, max_tier=args.max_tier, sources=sources)
    print(f"Wide panel: {wide.shape[0]:,} rows x {wide.shape[1]:,} columns")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_dir / "tiered_forecasting_panel.parquet"
    wide.to_parquet(panel_path, index=False)

    raw, scores = run_ablation(
        wide,
        tiers=range(args.max_tier + 1),
        min_train_seasons=args.min_train_seasons,
    )
    raw.to_csv(args.out_dir / "tiered_ablation_predictions.csv", index=False)
    scores.to_csv(args.out_dir / "tiered_ablation_scores.csv", index=False)
    for metric in (
        "mae",
        "rmse",
        "spearman_within_position",
        "top24_precision",
        "vorp_weighted_mae",
    ):
        table = ablation_table(scores, metric=metric)
        table.to_csv(args.out_dir / f"tiered_ablation_{metric}.csv")
        print(f"\n{metric.upper()}\n{table.to_string()}")
    print(f"\nArtifacts written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
