"""Optimal starting lineup given a roster's projections and a league's
required starting slots (including FLEX-type multi-position slots).

Solved as a linear assignment problem (players x slots) via
``scipy.optimize.linear_sum_assignment`` — exact optimum, and the problem is
tiny (a roster's worth of skill-position players against a handful of
slots), so there's no reason to reach for a heuristic.

Scope: only QB/RB/WR/TE are modeled (same as the rest of the weekly
forecasting system). K/DEF and any IDP slots are not solved here — the
caller (``scripts/sleeper_lineup.py``) passes those through unchanged from
the actual Sleeper roster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MODELED_POSITIONS = {"QB", "RB", "WR", "TE"}

SLOT_ELIGIBILITY: dict[str, set[str]] = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "K": {"K"},
    "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "WRTE_FLEX": {"WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}

_BIG = 1e6


def modeled_slots(roster_positions: list[str]) -> list[str]:
    """Starting slots this optimizer can actually fill (overlap QB/RB/WR/TE)."""
    return [s for s in roster_positions if SLOT_ELIGIBILITY.get(s, {s}) & MODELED_POSITIONS]


def optimize_lineup(
    players: pd.DataFrame,
    slots: list[str],
    objective_col: str = "proj_points",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assign players to slots to maximize total ``objective_col``.

    ``players`` needs ``position`` and ``objective_col`` columns; rows should
    already be restricted to modeled positions (QB/RB/WR/TE) with a real
    projection — see ``modeled_slots`` for restricting ``slots`` the same way.

    Returns ``(starters_df, bench_df, empty_slots)``. ``starters_df`` has one
    row per filled slot with an added ``slot`` column; ``empty_slots`` lists
    any slot that had no eligible player left to fill it (e.g. only one RB
    on the whole roster but two RB slots required).
    """
    players = players.reset_index(drop=True).copy()
    players[objective_col] = players[objective_col].fillna(0.0)

    n_players, n_slots = len(players), len(slots)
    size = max(n_players, n_slots, 1)

    cost = np.zeros((size, size))
    for i in range(n_players):
        for j in range(n_slots):
            eligible = SLOT_ELIGIBILITY.get(slots[j], {slots[j]})
            is_eligible = players.loc[i, "position"] in eligible
            cost[i, j] = -players.loc[i, objective_col] if is_eligible else _BIG

    row_ind, col_ind = linear_sum_assignment(cost)

    slot_to_player: dict[int, int] = {
        j: i
        for i, j in zip(row_ind, col_ind, strict=True)
        if i < n_players and j < n_slots and cost[i, j] < _BIG / 2
    }

    starters_rows, empty_slots = [], []
    for j, slot in enumerate(slots):
        if j in slot_to_player:
            row = players.loc[slot_to_player[j]].to_dict()
            row["slot"] = slot
            starters_rows.append(row)
        else:
            empty_slots.append(slot)

    filled_idx = set(slot_to_player.values())
    bench_idx = [i for i in range(n_players) if i not in filled_idx]
    bench_df = players.loc[bench_idx].reset_index(drop=True)
    starters_df = pd.DataFrame(starters_rows)
    return starters_df, bench_df, empty_slots


def adjust_for_scoring(board: pd.DataFrame, scoring_settings: dict) -> pd.DataFrame:
    """Rescale ``proj_points``/``floor``/``ceiling`` for a league's actual
    reception value (Sleeper's ``scoring_settings['rec']``: 0 standard, 0.5
    half-PPR, 1.0 full PPR — our model's native format).

    This is a first-order correction only: passing/rushing/receiving TD point
    values, yardage bonuses, and IDP scoring are not modeled. Directionally
    correct for the single biggest driver of cross-format point differences,
    not an exact re-scoring.
    """
    rec_value = scoring_settings.get("rec", 1.0)
    if rec_value == 1.0 or "receptions_ma3" not in board.columns:
        return board

    out = board.copy()
    shift = out["receptions_ma3"].fillna(0.0) * (1.0 - rec_value)
    for col in ("proj_points", "floor", "ceiling"):
        if col in out.columns:
            out[col] = (out[col] - shift).clip(lower=0).round(1)
    return out
