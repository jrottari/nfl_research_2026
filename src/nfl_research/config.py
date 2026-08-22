"""Central configuration. Edit the defaults here, not in the analysis code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# All generated/cached data lives outside the repo, on Google Drive, so it's
# backed up without bloating git (data/ is gitignored anyway).
DATA_DIR = Path(r"G:\My Drive\repos\nfl_research_2026\data")
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "exports"


def _replacement_rank() -> dict[str, int]:
    # Positional rank treated as replacement level in a 12-team league
    # running 1QB / 2RB / 3WR / 1TE / 1FLEX.
    return {"QB": 12, "RB": 30, "WR": 42, "TE": 12}


def _boom_bust() -> dict[str, tuple[float, float]]:
    # (boom threshold, bust threshold) in weekly PPR points.
    return {
        "QB": (24.0, 14.0),
        "RB": (20.0, 10.0),
        "WR": (20.0, 10.0),
        "TE": (15.0, 7.0),
    }


def _starter_slots() -> dict[str, int]:
    # Weekly positional finish that counts as a startable week (12-team).
    return {"QB": 12, "RB": 24, "WR": 36, "TE": 12}


def _tier_gap() -> dict[str, float]:
    # PPG gap that starts a new tier within a position.
    return {"QB": 1.5, "RB": 1.5, "WR": 1.5, "TE": 1.2}


@dataclass
class Settings:
    """Everything the pipeline needs to know."""

    season: int = 2025
    top_n: int = 250
    min_games: int = 4

    # nflverse now ships every position (K, OL, DL...) in one player-stats file.
    # Fantasy points are only meaningful for the skill positions, so filter.
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE", "FB")

    replacement_rank: dict[str, int] = field(default_factory=_replacement_rank)
    boom_bust: dict[str, tuple[float, float]] = field(default_factory=_boom_bust)
    starter_slots: dict[str, int] = field(default_factory=_starter_slots)
    tier_gap: dict[str, float] = field(default_factory=_tier_gap)

    export_dir: Path = EXPORT_DIR
    processed_dir: Path = PROCESSED_DIR

    # Board sizes for the per-position files.
    position_board_sizes: dict[str, int] = field(
        default_factory=lambda: {"QB": 40, "RB": 90, "WR": 110, "TE": 50}
    )
    flex_board_size: int = 200

    def __post_init__(self) -> None:
        self.export_dir = Path(self.export_dir).expanduser()
        self.processed_dir = Path(self.processed_dir).expanduser()


POS_COLORS = {
    "QB": "#2b6cb0",
    "RB": "#2f855a",
    "WR": "#c05621",
    "TE": "#6b46c1",
    "FB": "#718096",
}
