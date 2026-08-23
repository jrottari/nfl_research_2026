"""Turn raw fantasy-point forecasts into redraft ranks that respect roster construction.

The tier-1 XGBoost forecast (``2026_ppr_forecast_tier1_xgboost.csv``) ranks
players by *predicted points alone*. That's not the same thing as a draft
board: a 12-team league only starts 1 QB but 2-4 RB/WR/TE (dedicated slots
plus FLEX), so the "edge" a top player provides over the guy replacing him
on waivers is wildly different by position. QBs stack deep — the QB12 in
this league is still a low-end starter — so even the #1 QB's edge over
replacement is small. RB/WR are scarcer and compete for FLEX slots too, so
the top players there carry much more value-over-replacement.

This script runs a standard value-based-drafting (VBD) simulation:

1. Fill each team's dedicated slots (QB/RB/RB/WR/WR/TE) from the top of each
   position's projection, league-wide.
2. Fill the league's FLEX slots (RB/WR/TE-eligible) from whoever's left,
   sorted by projected points — this is what actually determines how many
   extra RBs/WRs/TEs are "startable" beyond the dedicated slots.
3. Replacement level for a position = the projection of the next player at
   that position after all its dedicated + FLEX-won starters are accounted
   for (i.e., the best player NOT drafted as a starter — the top waiver
   option).
4. Value over replacement (VORP) = predicted points - that replacement level.
5. Re-rank every player by VORP instead of raw points.

League assumed: 12 teams, starting QB, RB, RB, WR, WR, TE, FLEX, FLEX
(FLEX = RB/WR/TE). Change N_TEAMS / SLOTS / N_FLEX below for a different league.

Usage
-----
    python scripts/build_lineup_adjusted_ranks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from nfl_research.config import EXPORT_DIR

SOURCE_FILE = EXPORT_DIR / "2026_ppr_forecast_tier1_xgboost.csv"
OUT_FILE = EXPORT_DIR / "2026_ppr_ranks_lineup_adjusted.csv"

N_TEAMS = 12
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}  # dedicated starting slots/team
N_FLEX = 2  # FLEX slots/team, eligible positions below
FLEX_ELIGIBLE = ("RB", "WR", "TE")
POINTS_COL = "predicted_ppr_2026"


def simulate_starters(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    """Return (startable-player-count, replacement-level points) per position."""
    pools = {
        pos: df.loc[df["position"] == pos, ["player_id", POINTS_COL]]
        .sort_values(POINTS_COL, ascending=False)
        .reset_index(drop=True)
        for pos in df["position"].unique()
    }

    dedicated_n = {pos: SLOTS.get(pos, 0) * N_TEAMS for pos in pools}
    dedicated_ids = {
        pos: set(pools[pos].iloc[: dedicated_n[pos]]["player_id"]) for pos in pools
    }

    flex_candidates = pd.concat(
        [
            pools[pos].iloc[dedicated_n[pos] :]
            for pos in FLEX_ELIGIBLE
            if pos in pools
        ],
        ignore_index=True,
    ).sort_values(POINTS_COL, ascending=False)

    n_flex_slots = N_FLEX * N_TEAMS
    flex_winners = flex_candidates.head(n_flex_slots)
    flex_ids_by_pos = {
        pos: set(flex_winners.loc[flex_winners["player_id"].isin(pools[pos]["player_id"]), "player_id"])
        for pos in FLEX_ELIGIBLE
        if pos in pools
    }

    startable_n: dict[str, int] = {}
    replacement_pts: dict[str, float] = {}
    for pos, pool in pools.items():
        n_flex_won = len(flex_ids_by_pos.get(pos, set()))
        n_starters = dedicated_n[pos] + n_flex_won
        startable_n[pos] = n_starters
        if len(pool) > n_starters:
            replacement_pts[pos] = float(pool.iloc[n_starters][POINTS_COL])
        elif len(pool):
            replacement_pts[pos] = float(pool.iloc[-1][POINTS_COL])
        else:
            replacement_pts[pos] = 0.0

    return startable_n, replacement_pts


def main() -> int:
    df = pd.read_csv(SOURCE_FILE)
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)

    startable_n, replacement_pts = simulate_starters(df)

    print(f"League: {N_TEAMS} teams, slots={SLOTS}, FLEX={N_FLEX} ({'/'.join(FLEX_ELIGIBLE)})")
    print("\nStartable players & replacement level (season pts) by position:")
    for pos in ["QB", "RB", "WR", "TE"]:
        print(f"  {pos}: {startable_n[pos]:>3} startable  |  replacement = {replacement_pts[pos]:.1f} pts")

    out = df.copy()
    out["replacement_pts"] = out["position"].map(replacement_pts)
    out["vorp"] = out[POINTS_COL] - out["replacement_pts"]

    out = out.sort_values("vorp", ascending=False).reset_index(drop=True)
    out.insert(0, "adjusted_rank", out.index + 1)
    out["adjusted_pos_rank"] = out.groupby("position")["vorp"].rank(
        ascending=False, method="min"
    ).astype(int)
    out["raw_points_rank"] = out["overall_rank"]

    starter_ids: set[str] = set()
    for pos in SLOTS:
        starter_ids |= set(
            out.loc[out["position"] == pos]
            .sort_values(POINTS_COL, ascending=False)
            .head(SLOTS[pos] * N_TEAMS)["player_id"]
        )
    flex_pool = (
        out[out["position"].isin(FLEX_ELIGIBLE) & ~out["player_id"].isin(starter_ids)]
        .sort_values(POINTS_COL, ascending=False)
        .head(N_FLEX * N_TEAMS)["player_id"]
    )
    starter_ids |= set(flex_pool)
    out["starter_type"] = out["player_id"].apply(
        lambda pid: "starter" if pid in starter_ids else "bench/waiver"
    )

    cols = [
        "adjusted_rank",
        "player_id",
        "player_name",
        "position",
        "team",
        "vorp",
        POINTS_COL,
        "replacement_pts",
        "adjusted_pos_rank",
        "starter_type",
        "raw_points_rank",
        "position_rank",
        "points_ppr_2025",
        "age_2026",
        "ecr_rank_2026",
        "ecr_pos_rank_2026",
    ]
    out = out.loc[:, [c for c in cols if c in out.columns]]

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"\n--- TOP 30, lineup-adjusted (VORP) ranks ---")
    print(
        out[["adjusted_rank", "player_name", "position", "team", "vorp", POINTS_COL, "raw_points_rank"]]
        .head(30)
        .to_string(index=False)
    )
    print(f"\nSaved -> {OUT_FILE}  ({len(out):,} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
