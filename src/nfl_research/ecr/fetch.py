"""Live weekly FantasyPros Expert Consensus Rank (ECR), via nflreadpy.

Unlike ``load_ff_rankings(type="all")`` (a historical preseason archive, see
``forecasting/tier1.py``), ``type="week"`` scrapes FantasyPros' *current*
rankings pages live — every call returns today's snapshot, not a specific
past week. There is no historical archive of weekly ECR to backtest against;
this is a live decision tool, not something we can validate on past seasons
without archiving our own snapshots going forward.

FantasyPros publishes separate pages per position: QB, K, DST (team defense),
and IDP (DB/DL/LB) rankings are scoring-format-agnostic, while RB/WR/TE have
a PPR-specific page (``ppr-rb``/``ppr-wr``/``ppr-te``) we use since this
project's leagues are all PPR. ``rank``/``pos_rank`` are positional (RB1,
WR1, ...) — there is no FantasyPros overall cross-position weekly page, so
comparing e.g. an RB to a WR for a FLEX decision needs its own normalization
(see ``optimize.py``).
"""

from __future__ import annotations

import time

import pandas as pd

from ..config import CACHE_DIR

CACHE_PATH = CACHE_DIR / "ecr_weekly.parquet"
# Short TTL: unlike Sleeper's players endpoint, FantasyPros rankings move
# during the week (injury news, snap counts) and there's no stated rate
# limit — but we still don't want to re-scrape on every script invocation.
MAX_AGE_SECONDS = 3 * 60 * 60  # 3h

# FantasyPros page_pos -> our roster position codes.
POSITION_ALIAS = {"DST": "DEF"}

KEEP_COLS = [
    "page_pos", "player_name", "team", "rank", "ecr", "sd", "best", "worst", "pos_rank",
]


def _fetch_live() -> pd.DataFrame:
    import nflreadpy as nfl

    df = nfl.load_ff_rankings(type="week").to_pandas()
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    df = df.rename(columns={"page_pos": "position"})
    df["position"] = df["position"].replace(POSITION_ALIAS)
    for col in ("rank", "ecr", "sd", "best", "worst"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_weekly_ecr(force_refresh: bool = False) -> pd.DataFrame:
    """This week's FantasyPros ECR across QB/RB/WR/TE/K/DST/DB/DL/LB.

    Disk-cached for ``MAX_AGE_SECONDS``; pass ``force_refresh=True`` right
    before setting a lineup if you want the freshest possible snapshot.
    """
    if not force_refresh and CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < MAX_AGE_SECONDS:
            return pd.read_parquet(CACHE_PATH)

    df = _fetch_live()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df
