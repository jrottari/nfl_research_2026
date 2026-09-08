"""Join a Sleeper roster to weekly FantasyPros ECR rows.

FantasyPros' weekly rankings have no gsis_id/sleeper_id, only player_name +
team + position — so, like the preseason ECR join in
``forecasting/tier1.py``, this is a normalized-name join. Expect a small
number of misses (name variants, recent signings) that show up as NaN rank.
"""

from __future__ import annotations

import re

import pandas as pd

_NAME_CLEAN = re.compile(r"[^a-z ]+")
_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def normalize_name(name: str) -> str:
    cleaned = _NAME_CLEAN.sub("", str(name).lower().strip())
    return _SUFFIX.sub("", cleaned).strip()


def attach_ecr(roster_players: pd.DataFrame, ecr: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``rank``/``ecr``/``sd``/``best``/``worst``/``pos_rank`` onto
    ``roster_players`` (needs ``player_name``, ``position``) by normalized
    name + position. Unmatched rows keep NaN in the ECR columns.
    """
    left = roster_players.copy()
    left["_name_norm"] = left["player_name"].apply(normalize_name)

    right = ecr.copy()
    right["_name_norm"] = right["player_name"].apply(normalize_name)
    right = right.drop_duplicates(subset=["_name_norm", "position"], keep="first")
    right = right.drop(columns=["player_name", "team"], errors="ignore")

    out = left.merge(right, on=["_name_norm", "position"], how="left")
    return out.drop(columns=["_name_norm"])
