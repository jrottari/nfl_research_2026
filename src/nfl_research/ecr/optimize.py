"""Optimal starting lineup from FantasyPros weekly ECR, across every slot
type Sleeper leagues in this repo actually use — including K/DST and IDP,
which the Ridge-model optimizer in ``sleeper/optimize.py`` can't cover.

Solved the same way as the model-based optimizer: a players x slots linear
assignment problem via ``scipy.optimize.linear_sum_assignment``.

**The cross-position normalization problem (v1, meant to be workshopped):**
FantasyPros' weekly ECR is positional only (RB1, WR1, ...) — there's no
overall cross-position rank, so a FLEX/SUPER_FLEX/IDP_FLEX slot can't just
compare raw ranks between an RB and a WR. This module converts each
player's positional rank into a replacement-level fraction:

    normalized_value = 1 - (rank / REPLACEMENT_RANK[position])

i.e. how far above (positive) or below (negative) replacement level that
rank is, as a fraction of the replacement rank itself. This reuses the same
replacement-rank convention as ``config.Settings.replacement_rank`` (already
the project's VOR yardstick) rather than inventing a new scale. It's a
reasonable first cut, not a validated model — the K/DST/IDP replacement
ranks below are rough single-slot guesses, not derived from anything. Swap
this function out first if the lineup calls it produces look wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
    "IDP_FLEX": {"DB", "DL", "LB"},
    "DB": {"DB"},
    "DL": {"DL"},
    "LB": {"LB"},
}

# Rough "replacement level" positional rank per slot, 12-team-league scale.
# QB/RB/WR/TE mirror config.Settings.replacement_rank; K/DEF/IDP are
# placeholder single-slot guesses (12 teams x ~1 slot each) pending real
# validation.
REPLACEMENT_RANK: dict[str, int] = {
    "QB": 12, "RB": 30, "WR": 42, "TE": 12,
    "K": 12, "DEF": 12, "DB": 24, "DL": 24, "LB": 24,
}

# Which ECR column each risk objective reads. "consensus" uses FantasyPros'
# own positional ``rank`` (not the ``ecr`` decimal, which is a plain mean of
# individual expert ranks and can diverge sharply from ``rank`` when experts
# disagree a lot — verified in-session: gap between rank and ecr correlates
# 0.77 with ``sd`` across the full weekly pull, e.g. a TE with rank=36 but
# ecr=17.62 at sd=7.71/best=10/worst=36. ``rank`` tracks ``pos_rank`` exactly
# and looks like FantasyPros' actual (more robust) consensus algorithm).
# "ceiling"/"floor" use the single best/worst individual-expert rank —
# straight from FantasyPros' own columns, not a separately fitted variance
# model.
OBJECTIVE_COLUMNS = {"consensus": "rank", "ceiling": "best", "floor": "worst"}

_BIG = 1e6


def add_normalized_value(df: pd.DataFrame, mode: str = "consensus") -> pd.DataFrame:
    """Add ``objective_value`` (higher = better, comparable across positions)
    from the ECR column ``OBJECTIVE_COLUMNS[mode]`` selects.
    """
    if mode not in OBJECTIVE_COLUMNS:
        raise ValueError(f"mode must be one of {list(OBJECTIVE_COLUMNS)}, got {mode!r}")
    rank_col = OBJECTIVE_COLUMNS[mode]

    out = df.copy()
    replacement = out["position"].map(REPLACEMENT_RANK)
    out["objective_value"] = 1.0 - (out[rank_col] / replacement)
    return out


def modeled_slots(roster_positions: list[str]) -> list[str]:
    return [s for s in roster_positions if s in SLOT_ELIGIBILITY]


def optimize_ecr_lineup(
    players: pd.DataFrame, slots: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Assign players to slots maximizing total ``objective_value``.

    ``players`` needs ``position`` and ``objective_value`` (see
    ``add_normalized_value``); a player missing an ECR match (NaN objective)
    is treated as unstartable-by-recommendation rather than silently ranked
    worst-but-eligible, so it never displaces a real ranked player.

    Returns ``(starters_df, bench_df, empty_slots)`` — same shape as
    ``sleeper.optimize.optimize_lineup``.
    """
    players = players.reset_index(drop=True).copy()
    eligible_mask = players["objective_value"].notna()

    n_players, n_slots = len(players), len(slots)
    size = max(n_players, n_slots, 1)

    cost = np.full((size, size), _BIG)
    for i in range(n_players):
        if not eligible_mask[i]:
            continue
        for j in range(n_slots):
            if players.loc[i, "position"] in SLOT_ELIGIBILITY.get(slots[j], {slots[j]}):
                cost[i, j] = -players.loc[i, "objective_value"]

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
