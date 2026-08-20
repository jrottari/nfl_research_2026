"""Synthetic data shaped like the current nflverse player-stats schema.

Deliberately uses the *new* column names (``team``, ``passing_interceptions``,
``sacks_suffered``) plus a couple of old ones, so the schema layer is exercised
rather than bypassed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE", "FB", "K", "DE", "OT"]
TEAMS = [f"T{i:02d}" for i in range(32)]


def make_weekly(season: int = 2025, n_players: int = 520, seed: int = 11) -> pd.DataFrame:
    # 520 players across 8 positions leaves ~325 at skill positions, enough to
    # fill a 250-deep board after filtering.
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        pos = POSITIONS[min(len(POSITIONS) - 1, int(rng.integers(0, len(POSITIONS))))]
        team = TEAMS[p % 32]
        n_games = int(rng.integers(1, 18))
        for week in range(1, n_games + 1):
            opp = TEAMS[(p + week) % 32]
            if opp == team:
                opp = TEAMS[(p + week + 1) % 32]
            att = int(rng.integers(20, 45)) if pos == "QB" else 0
            car = int(rng.integers(0, 25)) if pos in ("RB", "QB", "FB") else int(rng.integers(0, 3))
            tgt = int(rng.integers(0, 14)) if pos in ("WR", "TE", "RB") else 0
            rec = int(rng.integers(0, tgt + 1))
            rows.append(
                {
                    "player_id": f"00-{p:07d}",
                    "player_name": f"P.Player{p:03d}",
                    "player_display_name": f"Player {p:03d}",
                    "position": pos,
                    "position_group": pos,
                    "headshot_url": "",
                    "season": season,
                    "week": week,
                    "season_type": "REG" if week <= 17 else "POST",
                    "game_id": f"{season}_{week:02d}_{team}_{opp}",
                    "team": team,
                    "opponent_team": opp,
                    "completions": int(att * 0.63),
                    "attempts": att,
                    "passing_yards": att * rng.uniform(5, 9),
                    "passing_tds": int(rng.integers(0, 4)) if att else 0,
                    "passing_interceptions": int(rng.integers(0, 3)) if att else 0,
                    "sacks_suffered": int(rng.integers(0, 5)) if att else 0,
                    "sack_yards_lost": float(rng.integers(0, 30)) if att else 0.0,
                    "sack_fumbles": 0.0,
                    "sack_fumbles_lost": float(rng.integers(0, 2)),
                    "passing_air_yards": att * rng.uniform(6, 11),
                    "passing_yards_after_catch": att * rng.uniform(2, 4),
                    "passing_first_downs": float(int(att * 0.35)),
                    "passing_2pt_conversions": 0.0,
                    "carries": car,
                    "rushing_yards": car * rng.uniform(2, 6),
                    "rushing_tds": float(rng.integers(0, 2)),
                    "rushing_fumbles": 0.0,
                    "rushing_fumbles_lost": float(rng.integers(0, 2)),
                    "rushing_first_downs": float(int(car * 0.25)),
                    "rushing_2pt_conversions": 0.0,
                    "receptions": rec,
                    "targets": tgt,
                    "receiving_yards": rec * rng.uniform(6, 16),
                    "receiving_tds": float(rng.integers(0, 2)),
                    "receiving_fumbles": 0.0,
                    "receiving_fumbles_lost": 0.0,
                    "receiving_air_yards": tgt * rng.uniform(5, 14),
                    "receiving_yards_after_catch": rec * rng.uniform(2, 7),
                    "receiving_first_downs": float(int(rec * 0.5)),
                    "receiving_2pt_conversions": 0.0,
                    "target_share": float(rng.uniform(0, 0.32)),
                    "air_yards_share": float(rng.uniform(0, 0.35)),
                    "special_teams_tds": 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df["fantasy_points"] = (
        df["passing_yards"] / 25 + df["passing_tds"] * 4 - df["passing_interceptions"] * 2
        + df["rushing_yards"] / 10 + df["rushing_tds"] * 6
        + df["receiving_yards"] / 10 + df["receiving_tds"] * 6
    )
    df["fantasy_points_ppr"] = df["fantasy_points"] + df["receptions"]
    return df


SUM_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "sack_yards_lost", "sack_fumbles",
    "sack_fumbles_lost", "passing_air_yards", "passing_yards_after_catch",
    "passing_first_downs", "passing_2pt_conversions", "carries", "rushing_yards",
    "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_first_downs", "rushing_2pt_conversions", "receptions", "targets",
    "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_fumbles_lost", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs",
    "receiving_2pt_conversions", "special_teams_tds", "fantasy_points",
    "fantasy_points_ppr",
]


def make_season_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    """Mimic ``load_player_stats(summary_level='reg')``."""
    reg = weekly[weekly["season_type"] == "REG"]
    keys = ["player_id", "player_name", "player_display_name", "position",
            "position_group", "season"]
    out = reg.groupby(keys, as_index=False)[SUM_COLS].sum()
    out["games"] = reg.groupby("player_id")["week"].nunique().reindex(out["player_id"]).values
    out["team"] = reg.groupby("player_id")["team"].last().reindex(out["player_id"]).values
    return out


def install_fake_module(monkeypatch=None, season: int = 2025):
    """Register a fake ``nflreadpy`` in sys.modules returning Polars-ish frames."""
    import sys
    import types

    weekly = make_weekly(season=season)
    totals = make_season_totals(weekly)

    class FakePolars:
        # Stand-in for a Polars frame: the loader only ever calls .to_pandas()
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df.copy()

    module = types.ModuleType("nflreadpy")

    def load_player_stats(seasons=None, summary_level="week"):
        return FakePolars(weekly if summary_level == "week" else totals)

    module.load_player_stats = load_player_stats
    module.load_schedules = lambda seasons=None: FakePolars(pd.DataFrame())
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "nflreadpy", module)
    else:
        sys.modules["nflreadpy"] = module
    return module
