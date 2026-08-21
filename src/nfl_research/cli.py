"""Command-line entry point.

python -m nfl_research --season 2025 --top-n 250 --out ~/Desktop/fantasy_2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import exports, pipeline
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfl_research",
        description="Build season-end fantasy football boards from nflverse data.",
    )
    parser.add_argument("--season", type=int, default=2025, help="season to analyze")
    parser.add_argument(
        "--top-n", type=int, default=250, help="size of the overall PPR board (default: 250)"
    )
    parser.add_argument(
        "--min-games", type=int, default=4, help="minimum games to qualify for per-game ranks"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="export folder (default: data/exports)"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="positions to include (default: QB RB WR TE FB)",
    )
    parser.add_argument("--only", nargs="+", default=None, help="export only these filenames")
    parser.add_argument(
        "--no-export", action="store_true", help="build and summarize without writing CSVs"
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="also cache the full board to data/processed as parquet",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = Settings(
        season=args.season,
        top_n=args.top_n,
        min_games=args.min_games,
        positions=tuple(args.positions) if args.positions else Settings.positions,
    )
    if args.out is not None:
        settings.export_dir = Path(args.out).expanduser()

    print(f"Building {settings.season} boards (top {settings.top_n})...")
    boards = pipeline.build_season(settings)

    overall = boards["overall"]
    top_n = boards["top_n"]
    print(f"\n{len(overall):,} qualifying players; board depth {len(top_n)}")
    if len(top_n) < settings.top_n:
        print(f"WARNING: only {len(top_n)} players available, asked for {settings.top_n}")
    print("\nPosition mix in the board:")
    print(top_n["position"].value_counts().to_string())
    print("\nReplacement level (PPR PPG):")
    print(boards["replacement"].to_string(index=False))

    if args.parquet:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        path = settings.processed_dir / f"{settings.season}_overall.parquet"
        overall.to_parquet(path, index=False)
        print(f"\ncached -> {path}")

    if not args.no_export:
        print()
        exports.export_all(
            pipeline.to_export_map(boards, settings), settings.export_dir, only=args.only
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
