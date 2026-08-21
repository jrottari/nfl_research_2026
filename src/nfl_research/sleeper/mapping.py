"""Sleeper player_id <-> nflverse (gsis) player_id crosswalk.

Sleeper rosters are lists of Sleeper's own numeric-string player ids. Our
forecasting pipeline is keyed on nflverse's ``gsis_id``. ``nflreadpy``'s
``load_ff_playerids()`` (the dynastyprocess crosswalk) carries both, so that's
the join key — cached locally since it changes rarely and the source is
"soft" (nflreadpy will complain but keep working across seasons where the
file lags a rookie class by a few days).
"""

from __future__ import annotations

import pandas as pd

from ..forecasting.source_cache import cached_loader


def load_crosswalk() -> pd.DataFrame:
    """gsis_id / sleeper_id / name / position / team, one row per player.

    ``sleeper_id`` is cast to a zero-padded-free string to match the string
    ids Sleeper returns in roster player lists (``"4046"``, not ``4046.0``).
    """
    import nflreadpy as nfl

    df = cached_loader("raw_ff_playerids", nfl.load_ff_playerids)
    df = df[["gsis_id", "sleeper_id", "name", "position", "team"]].copy()
    df = df.dropna(subset=["sleeper_id"])
    df["sleeper_id"] = df["sleeper_id"].astype("Int64").astype(str)
    return df.drop_duplicates(subset=["sleeper_id"])


def sleeper_to_gsis_map(crosswalk: pd.DataFrame | None = None) -> dict[str, str]:
    crosswalk = crosswalk if crosswalk is not None else load_crosswalk()
    valid = crosswalk.dropna(subset=["gsis_id"])
    return dict(zip(valid["sleeper_id"], valid["gsis_id"], strict=True))
