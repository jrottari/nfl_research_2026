"""Season-end NFL fantasy football research, built on nflreadpy."""

from .config import POS_COLORS, Settings
from .pipeline import build_season, export_filenames, to_export_map

__all__ = [
    "Settings",
    "POS_COLORS",
    "build_season",
    "export_filenames",
    "to_export_map",
]
__version__ = "0.1.0"
