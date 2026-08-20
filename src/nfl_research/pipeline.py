"""End-to-end: raw nflverse data in, finished boards out."""

from __future__ import annotations

import pandas as pd

from . import loaders, metrics, rankings
from .config import Settings


def build_season(
    settings: Settings,
    weekly: pd.DataFrame | None = None,
    season_totals: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build every board for one season.

    Pass ``weekly`` / ``season_totals`` to work from frames you already have
    (that's how the tests run without a network call); leave them as None to
    download via nflreadpy.

    Returns a dict with keys: ``overall``, ``top_n``, ``flex``, ``QB``, ``RB``,
    ``WR``, ``TE``, ``game_log``, ``gainers``, ``replacement``.
    """
    if weekly is None:
        weekly = loaders.load_weekly(settings.season, season_type="REG")
    if season_totals is None:
        season_totals = loaders.load_season_totals(settings.season, summary_level="reg")

    weekly = weekly[weekly["position"].isin(settings.positions)].copy()
    season = season_totals[season_totals["position"].isin(settings.positions)].copy()
    if "season" in season.columns:
        season = season[season["season"] == settings.season]

    # nflverse ships a season-level `games` column, but it is not guaranteed
    # across schema versions - and we want games_active and mean target share
    # regardless, so derive from the weekly frame either way.
    per_player = metrics.games_played(weekly)
    season = season.drop(columns=[c for c in per_player.columns if c != "player_id"],
                         errors="ignore")
    season = season.merge(per_player, on="player_id", how="inner")

    season = metrics.add_scoring(season)
    season = metrics.add_volume(season)
    season = metrics.add_efficiency(season)

    season = season.merge(metrics.weekly_consistency(weekly, settings),
                          on="player_id", how="left")
    season = metrics.add_consistency_rates(season)

    season = rankings.add_ranks(season, settings)
    season = rankings.add_vor(season, settings)
    season = rankings.add_tiers(season, settings)

    overall = rankings.build_overall(season)
    player_ids = overall.attrs["player_ids"]
    top_n = overall.head(settings.top_n).copy()

    boards: dict[str, pd.DataFrame] = {
        "overall": overall,
        "top_n": top_n,
        "flex": rankings.flex_board(overall, settings.flex_board_size),
        "game_log": metrics.weekly_game_log(weekly, player_ids[: settings.top_n]),
        "gainers": rankings.value_gainers(season, settings),
    }
    for pos, size in settings.position_board_sizes.items():
        boards[pos] = rankings.position_board(overall, pos, size)

    boards["replacement"] = pd.DataFrame(
        [
            {"position": pos, "replacement_rank": settings.replacement_rank[pos],
             "replacement_ppg": ppg}
            for pos, ppg in rankings.replacement_levels(season, settings).items()
        ]
    )
    return boards


def export_filenames(settings: Settings) -> dict[str, str]:
    """Board key -> output filename."""
    season = settings.season
    return {
        "top_n": f"{season}_overall_top{settings.top_n}_ppr.csv",
        "flex": f"{season}_flex_top{settings.flex_board_size}.csv",
        "QB": f"{season}_qb_rankings.csv",
        "RB": f"{season}_rb_rankings.csv",
        "WR": f"{season}_wr_rankings.csv",
        "TE": f"{season}_te_rankings.csv",
        "game_log": f"{season}_weekly_game_log_top{settings.top_n}.csv",
        "overall": f"{season}_full_season_all_players.csv",
    }


def to_export_map(boards: dict[str, pd.DataFrame], settings: Settings) -> dict[str, pd.DataFrame]:
    """Rekey the boards dict by output filename, ready for ``exports.export_all``."""
    names = export_filenames(settings)
    return {filename: boards[key] for key, filename in names.items() if key in boards}
