"""Synthetic Sleeper API fixtures, shaped like the real endpoints' JSON.

Mirrors ``tests/fake_nflverse.py``'s approach: build data that matches the
real service's shape closely enough to exercise the real parsing/merge code,
without any network access.
"""

from __future__ import annotations

import pandas as pd

USER = {"user_id": "111", "username": "testuser", "display_name": "Test User"}

LEAGUE = {
    "league_id": "L1",
    "name": "Test League",
    "season": "2025",
    "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "BN"],
    "scoring_settings": {"rec": 1.0, "pass_td": 4, "rush_td": 6, "rec_td": 6},
}

LEAGUE_HALF_PPR = {
    **LEAGUE, "league_id": "L2", "name": "Half PPR League", "scoring_settings": {"rec": 0.5}
}

ROSTERS = [
    {
        "roster_id": 1,
        "owner_id": "111",
        "players": ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "2001", "3001"],
        "starters": ["1001", "1002", "1003", "1004", "1005", "1006", "2001", "3001"],
        "reserve": [],
        "taxi": [],
    }
]

ROSTERS_HALF_PPR = [{**ROSTERS[0], "roster_id": 1}]

# A league that hasn't drafted yet: real Sleeper shape ("pre_draft" leagues
# return an empty players list, not null) — caught a real KeyError bug where
# an empty roster produced a DataFrame with zero columns.
ROSTERS_UNDRAFTED = [
    {"roster_id": 2, "owner_id": "111", "players": [], "starters": [], "reserve": [], "taxi": []}
]

LEAGUE_USERS = [
    {"user_id": "111", "display_name": "TestUser", "metadata": {"team_name": "The Testers"}}
]

# Sleeper's own player dict — used as a fallback label for K/DEF (not in the crosswalk).
PLAYERS_DICT = {
    "1001": {"full_name": "QB One", "position": "QB", "team": "AAA"},
    "1002": {"full_name": "RB One", "position": "RB", "team": "BBB"},
    "1003": {"full_name": "RB Two", "position": "RB", "team": "CCC"},
    "1004": {"full_name": "WR One", "position": "WR", "team": "DDD"},
    "1005": {"full_name": "WR Two", "position": "WR", "team": "EEE"},
    "1006": {"full_name": "TE One", "position": "TE", "team": "FFF"},
    "1007": {"full_name": "RB Three (bench)", "position": "RB", "team": "GGG"},
    "2001": {"full_name": "Kicker One", "position": "K", "team": "HHH"},
    "3001": {"full_name": "Defense One", "position": "DEF", "team": "III"},
}


def make_crosswalk() -> pd.DataFrame:
    """gsis_id/sleeper_id crosswalk covering the modeled (QB/RB/WR/TE) fixture players.
    K (2001) and DEF (3001) are deliberately absent, matching the real crosswalk
    (it's player-level, not team-defense-level).
    """
    rows = [
        ("00-1001", "1001", "QB One", "QB", "AAA"),
        ("00-1002", "1002", "RB One", "RB", "BBB"),
        ("00-1003", "1003", "RB Two", "RB", "CCC"),
        ("00-1004", "1004", "WR One", "WR", "DDD"),
        ("00-1005", "1005", "WR Two", "WR", "EEE"),
        ("00-1006", "1006", "TE One", "TE", "FFF"),
        ("00-1007", "1007", "RB Three (bench)", "RB", "GGG"),
    ]
    return pd.DataFrame(rows, columns=["gsis_id", "sleeper_id", "name", "position", "team"])
