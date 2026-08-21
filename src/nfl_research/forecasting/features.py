"""Feature engineering: turns long-format season totals into a model-ready panel."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .feature_registry import cols_for_tier

POSITION_CODES: dict[str, int] = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a panel with lag features from long-format season totals.

    Each row is a (player_id, season) pair. Rows with no lag-1 history are
    dropped - you cannot make a data-driven forecast with zero prior seasons.

    Columns added:
        points_ppr_lag{1,2,3} - PPR totals from prior seasons
        ppg_lag{1,2}          - PPG from prior seasons
        games_lag{1,2}        - games played from prior seasons
        trend_1               - lag1 - lag2  (1-year change)
        trend_2               - lag2 - lag3  (2-year change)
        career_season         - how many seasons prior to this one (0 = rookie in data)
        exp_smooth            - exponentially-weighted past totals (alpha=0.5)
        pos_code              - integer position code for tree models
        target                - fantasy_points_ppr this season (what we predict)
    """
    # Deduplicate: some nflverse pulls return multiple rows per player per season
    # (e.g., after a trade). Sum counting stats, keep last string fields.
    agg: dict[str, tuple] = {
        "player_name": ("player_name", "last"),
        "position": ("position", "last"),
        "fantasy_points_ppr": ("fantasy_points_ppr", "sum"),
        "fantasy_points": ("fantasy_points", "sum"),
    }
    if "team" in df.columns:
        agg["team"] = ("team", "last")
    elif "recent_team" in df.columns:
        agg["team"] = ("recent_team", "last")
    if "games" in df.columns:
        agg["games"] = ("games", "max")
    if "ppg_ppr" in df.columns:
        # ppg must be recomputed after dedup, not averaged
        pass

    panel = df.groupby(["player_id", "season"], as_index=False).agg(**agg)

    if "games" not in panel.columns:
        panel["games"] = 17  # fallback

    panel["ppg_ppr"] = (
        panel["fantasy_points_ppr"] / panel["games"].replace(0, float("nan"))
    ).fillna(0.0)

    panel = panel.sort_values(["player_id", "season"]).reset_index(drop=True)

    g = panel.groupby("player_id", sort=False)

    panel["points_ppr_lag1"] = g["fantasy_points_ppr"].shift(1)
    panel["points_ppr_lag2"] = g["fantasy_points_ppr"].shift(2)
    panel["points_ppr_lag3"] = g["fantasy_points_ppr"].shift(3)
    panel["ppg_lag1"] = g["ppg_ppr"].shift(1)
    panel["ppg_lag2"] = g["ppg_ppr"].shift(2)
    panel["games_lag1"] = g["games"].shift(1)
    panel["games_lag2"] = g["games"].shift(2)

    # trend: recent trajectory
    panel["trend_1"] = panel["points_ppr_lag1"] - panel["points_ppr_lag2"]
    panel["trend_2"] = panel["points_ppr_lag2"] - panel["points_ppr_lag3"]

    # how many seasons this player has appeared in before the current one
    panel["career_season"] = g.cumcount()

    # exponentially-weighted blended history (weights 0.5 / 0.3 / 0.2 normalised)
    w1 = panel["points_ppr_lag1"].notna().astype(float) * 0.5
    w2 = panel["points_ppr_lag2"].notna().astype(float) * 0.3
    w3 = panel["points_ppr_lag3"].notna().astype(float) * 0.2
    total_w = (w1 + w2 + w3).replace(0, float("nan"))
    panel["exp_smooth"] = (
        panel["points_ppr_lag1"].fillna(0) * 0.5
        + panel["points_ppr_lag2"].fillna(0) * 0.3
        + panel["points_ppr_lag3"].fillna(0) * 0.2
    ) / total_w

    panel["pos_code"] = panel["position"].map(POSITION_CODES).fillna(-1).astype(int)

    # Rename target for clarity
    panel = panel.rename(columns={"fantasy_points_ppr": "target"})

    # Drop rows without at least 1 year of history
    panel = panel.dropna(subset=["points_ppr_lag1"]).copy()

    return panel.reset_index(drop=True)


# Columns fed to sklearn / xgboost models
_FEATURE_COLS = [
    "points_ppr_lag1",
    "points_ppr_lag2",
    "ppg_lag1",
    "ppg_lag2",
    "games_lag1",
    "games_lag2",
    "trend_1",
    "trend_2",
    "career_season",
    "exp_smooth",
    "pos_code",
]


def feature_cols(max_tier: int = 0) -> list[str]:
    """Registered numeric features through ``max_tier``."""
    return cols_for_tier(max_tier)


def feature_matrix(
    panel: pd.DataFrame,
    fill_value: float = 0.0,
    max_tier: int = 0,
) -> pd.DataFrame:
    """Slice the wide panel using the feature registry and zero-fill values."""
    wanted = feature_cols(max_tier)
    X = panel[[c for c in wanted if c in panel.columns]].copy()
    X = X.fillna(fill_value)
    return X


def build_wide_panel(
    panel: pd.DataFrame,
    *,
    max_tier: int,
    as_of: datetime,
    sources: dict[str, pd.DataFrame] | None = None,
    fetch_tier0: bool = False,
) -> pd.DataFrame:
    """Apply tier builders to an existing lag panel with an explicit cutoff.

    External frames may be injected through ``sources`` for an offline,
    reproducible build. Tier-0 static loaders are fetched only when explicitly
    requested; missing-lag and rookie indicators are always constructed.
    """
    from .tier0 import build_tier0_features
    from .tier1 import build_tier1_features
    from .tier2 import build_tier2_features
    from .tier3 import build_tier3_features
    from .tier4 import build_tier4_features

    sources = sources or {}
    out = build_tier0_features(
        panel,
        as_of=as_of,
        include_age=fetch_tier0 or "players" in sources,
        include_draft=fetch_tier0 or "draft_picks" in sources,
        sources=sources,
    )
    if max_tier >= 1:
        out = build_tier1_features(out, as_of=as_of, sources=sources)
    if max_tier >= 2:
        out = build_tier2_features(out, as_of=as_of, sources=sources)
    if max_tier >= 3:
        out = build_tier3_features(out, as_of=as_of, sources=sources)
    if max_tier >= 4:
        out = build_tier4_features(out, as_of=as_of, sources=sources)
    return out.reset_index(drop=True)


def build_historical_wide_panel(
    panel: pd.DataFrame,
    *,
    max_tier: int,
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build every forecast season using its own August 1 information cutoff."""
    from datetime import datetime

    parts = []
    for season in sorted(int(value) for value in panel["season"].dropna().unique()):
        season_panel = panel[panel["season"] == season].copy()
        built = build_wide_panel(
            season_panel,
            max_tier=max_tier,
            as_of=datetime(season, 8, 1),
            sources=sources,
        )
        if not built.empty:
            parts.append(built)
    return pd.concat(parts, ignore_index=True) if parts else panel.iloc[:0].copy()


def make_forecast_row(
    player_id: str,
    panel: pd.DataFrame,
    forecast_season: int,
) -> pd.DataFrame | None:
    """Build a single-row feature vector for a player to forecast `forecast_season`.

    Looks back into `panel` for their lag history. Returns None if the player
    has no data in the season immediately before `forecast_season`.
    """
    history = panel[panel["player_id"] == player_id].sort_values("season")
    seasons_before = history[history["season"] < forecast_season]
    if seasons_before.empty:
        return None

    latest = seasons_before.iloc[-1]
    row: dict = {
        "player_id": player_id,
        "player_name": latest.get("player_name", player_id),
        "position": latest.get("position", "UNK"),
        "season": forecast_season,
        "points_ppr_lag1": latest["target"] if "target" in latest else float("nan"),
        "ppg_lag1": latest.get("ppg_ppr", float("nan")),
        "games_lag1": latest.get("games", float("nan")),
        "career_season": int(latest.get("career_season", 0)) + 1,
    }

    # 2 seasons back
    if len(seasons_before) >= 2:
        prev2 = seasons_before.iloc[-2]
        row["points_ppr_lag2"] = prev2["target"] if "target" in prev2 else float("nan")
        row["ppg_lag2"] = prev2.get("ppg_ppr", float("nan"))
        row["games_lag2"] = prev2.get("games", float("nan"))
    else:
        row["points_ppr_lag2"] = float("nan")
        row["ppg_lag2"] = float("nan")
        row["games_lag2"] = float("nan")

    # 3 seasons back
    row["points_ppr_lag3"] = (
        seasons_before.iloc[-3]["target"]
        if len(seasons_before) >= 3 and "target" in seasons_before.iloc[-3]
        else float("nan")
    )

    row["trend_1"] = (
        row["points_ppr_lag1"] - row["points_ppr_lag2"]
        if pd.notna(row["points_ppr_lag2"])
        else float("nan")
    )
    row["trend_2"] = (
        row["points_ppr_lag2"] - row["points_ppr_lag3"]
        if pd.notna(row.get("points_ppr_lag3"))
        else float("nan")
    )

    # exp_smooth
    vals = [
        (row["points_ppr_lag1"], 0.5),
        (row["points_ppr_lag2"], 0.3),
        (row.get("points_ppr_lag3", float("nan")), 0.2),
    ]
    num = sum(v * w for v, w in vals if pd.notna(v))
    denom = sum(w for v, w in vals if pd.notna(v))
    row["exp_smooth"] = num / denom if denom > 0 else float("nan")

    row["pos_code"] = POSITION_CODES.get(row["position"], -1)

    return pd.DataFrame([row])
