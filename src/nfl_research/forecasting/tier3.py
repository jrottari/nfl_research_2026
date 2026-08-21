"""Tier-3 (2016+) Next Gen Stats and participation features."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

NGS_ALIASES = {
    "separation_avg": ("avg_separation", "separation"),
    "cushion_avg": ("avg_cushion", "cushion"),
    "intended_air_yards": ("avg_intended_air_yards", "avg_air_yards"),
    "rush_yards_oe": ("rush_yards_over_expected_per_att", "ryoe_per_att"),
    "efficiency": ("efficiency",),
    "time_to_los": ("avg_time_to_los", "time_to_los"),
    "completion_pct_oe": ("completion_percentage_above_expectation", "cpoe"),
}


def _identity(df: pd.DataFrame) -> str | None:
    return next((c for c in ("player_id", "gsis_id", "player_gsis_id") if c in df), None)


def build_nextgen_features(
    panel: pd.DataFrame,
    frames: list[pd.DataFrame],
    *,
    as_of: datetime,
) -> pd.DataFrame:
    """Join prior-season NGS receiving, rushing, and passing summaries."""
    parts: list[pd.DataFrame] = []
    for raw in frames:
        df = raw.copy()
        pid = _identity(df)
        if not pid or "season" not in df:
            continue
        df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year - 1]
        selected: dict[str, str] = {}
        for canonical, aliases in NGS_ALIASES.items():
            actual = next((c for c in aliases if c in df), None)
            if actual:
                selected[actual] = canonical
        if not selected:
            continue
        keep = df[[pid, "season", *selected]].rename(columns={pid: "player_id", **selected})
        keep = keep.groupby(["player_id", "season"], as_index=False).mean(numeric_only=True)
        parts.append(keep)
    if not parts:
        return panel
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(part, on=["player_id", "season"], how="outer", suffixes=("", "_dup"))
        wide = wide.drop(columns=[c for c in wide if c.endswith("_dup")])
    wide["season"] += 1
    return panel.merge(wide, on=["player_id", "season"], how="left")


def build_participation_features(
    panel: pd.DataFrame,
    participation: pd.DataFrame,
    *,
    as_of: datetime,
) -> pd.DataFrame:
    """Aggregate prior-season participation; silently accepts a lagging empty source."""
    df = participation.copy()
    pid = _identity(df)
    if df.empty or not pid or "season" not in df:
        return panel
    value = next(
        (c for c in ("offense_pct", "participation", "route_participation") if c in df), None
    )
    if not value:
        return panel
    df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year - 1]
    agg = (
        df.groupby([pid, "season"], as_index=False)[value]
        .mean()
        .rename(columns={pid: "player_id", value: "participation_rate"})
    )
    agg["season"] += 1
    return panel.merge(agg, on=["player_id", "season"], how="left")


def build_tier3_features(
    panel: pd.DataFrame,
    *,
    as_of: datetime,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    sources = sources or {}
    ngs = [sources[k] for k in ("ngs_receiving", "ngs_rushing", "ngs_passing") if k in sources]
    out = build_nextgen_features(panel, ngs, as_of=as_of) if ngs else panel.copy()
    if "participation" in sources:
        out = build_participation_features(out, sources["participation"], as_of=as_of)
    return out
