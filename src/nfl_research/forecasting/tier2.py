"""Tier-2 (2012+) situation and usage features with point-in-time guards."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _col(df: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _prior_join(panel: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["season"] = pd.to_numeric(features["season"], errors="coerce") + 1
    return panel.merge(features, on=["player_id", "season"], how="left")


def build_snap_features(
    panel: pd.DataFrame, snaps: pd.DataFrame, *, as_of: datetime
) -> pd.DataFrame:
    """Aggregate prior-season offensive snap share and trajectory."""
    df = snaps.copy()
    pid = _col(df, "player_id", "gsis_id")
    share = _col(df, "offense_pct", "offense_snap_pct", "snap_pct")
    if not pid or not share or not {"season", "week"}.issubset(df.columns):
        return panel
    df = df.rename(columns={pid: "player_id", share: "snap_pct"})
    df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year - 1]
    df["snap_pct"] = pd.to_numeric(df["snap_pct"], errors="coerce")
    df["half"] = np.where(pd.to_numeric(df["week"], errors="coerce") <= 9, "h1", "h2")
    base = (
        df.groupby(["player_id", "season"])
        .agg(
            snap_pct_lag1=("snap_pct", "mean"),
            wks_above_50pct_snaps=("snap_pct", lambda s: int((s >= 0.50).sum())),
            wks_above_75pct_snaps=("snap_pct", lambda s: int((s >= 0.75).sum())),
        )
        .reset_index()
    )
    halves = df.pivot_table(
        index=["player_id", "season"], columns="half", values="snap_pct", aggfunc="mean"
    )
    base = base.merge(halves.reset_index(), on=["player_id", "season"], how="left")
    base["snap_pct_trend"] = base.get("h2", np.nan) - base.get("h1", np.nan)
    return _prior_join(panel, base.drop(columns=["h1", "h2"], errors="ignore"))


def build_depth_features(
    panel: pd.DataFrame, depth: pd.DataFrame, *, as_of: datetime
) -> pd.DataFrame:
    """Use only the latest depth-chart record published on or before ``as_of``."""
    df = depth.copy()
    pid = _col(df, "player_id", "gsis_id")
    date = _col(df, "dt", "date", "timestamp", "as_of")
    rank = _col(df, "depth_position", "depth_team", "depth", "pos_rank")
    if not pid or not rank:
        return panel
    df = df.rename(columns={pid: "player_id", rank: "depth_chart_pos"})
    if date:
        df[date] = pd.to_datetime(df[date], errors="coerce")
        df = df[df[date] <= pd.Timestamp(as_of)].sort_values(date)
    if "season" in df:
        df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year]
        keys = ["player_id", "season"]
    else:
        df["season"] = as_of.year
        keys = ["player_id", "season"]
    df["depth_chart_pos"] = pd.to_numeric(df["depth_chart_pos"], errors="coerce")
    latest = df.drop_duplicates(keys, keep="last")[keys + ["depth_chart_pos"]]
    return panel.merge(latest, on=keys, how="left")


def build_pbp_features(panel: pd.DataFrame, pbp: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """Derive prior-year air-yards, WOPR, and red-zone opportunity shares."""
    df = pbp.copy()
    season = pd.to_numeric(df.get("season"), errors="coerce")
    df = df[season <= as_of.year - 1].copy()
    receiver = _col(df, "receiver_player_id", "receiver_id")
    rusher = _col(df, "rusher_player_id", "rusher_id")
    team = _col(df, "posteam", "team")
    if not team or (not receiver and not rusher):
        return panel
    parts: list[pd.DataFrame] = []
    if receiver:
        rec = df[df[receiver].notna()].copy()
        rec["air"] = pd.to_numeric(rec.get("air_yards", 0), errors="coerce").fillna(0)
        rec["target"] = 1.0
        rec["rz_target"] = (
            pd.to_numeric(rec.get("yardline_100", 100), errors="coerce") <= 20
        ).astype(float)
        player = (
            rec.groupby([receiver, "season", team])
            .agg(air=("air", "sum"), targets=("target", "sum"), rz=("rz_target", "sum"))
            .reset_index()
        )
        totals = (
            rec.groupby(["season", team])
            .agg(
                team_air=("air", "sum"),
                team_targets=("target", "sum"),
                team_rz=("rz_target", "sum"),
            )
            .reset_index()
        )
        player = player.merge(totals, on=["season", team])
        player["air_yards_share"] = player["air"] / player["team_air"].replace(0, np.nan)
        tgt_share = player["targets"] / player["team_targets"].replace(0, np.nan)
        player["wopr"] = 1.5 * tgt_share + 0.7 * player["air_yards_share"]
        player["rz_target_share"] = player["rz"] / player["team_rz"].replace(0, np.nan)
        parts.append(
            player.rename(columns={receiver: "player_id"})[
                ["player_id", "season", "air_yards_share", "wopr", "rz_target_share"]
            ]
        )
    if rusher:
        rush = df[df[rusher].notna()].copy()
        rush["rz"] = (pd.to_numeric(rush.get("yardline_100", 100), errors="coerce") <= 20).astype(
            float
        )
        player = rush.groupby([rusher, "season", team])["rz"].sum().reset_index()
        total = rush.groupby(["season", team])["rz"].sum().reset_index(name="team_rz")
        player = player.merge(total, on=["season", team])
        player["rz_carry_share"] = player["rz"] / player["team_rz"].replace(0, np.nan)
        parts.append(
            player.rename(columns={rusher: "player_id"})[["player_id", "season", "rz_carry_share"]]
        )
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["player_id", "season"], how="outer")
    out = _prior_join(panel, merged)
    pass_oe = _col(df, "pass_oe", "pass_rate_over_expected")
    if pass_oe and "team" in panel.columns:
        team_roe = (
            df.groupby(["season", team])[pass_oe]
            .mean()
            .reset_index(name="team_pass_roe")
            .rename(columns={team: "team"})
        )
        team_roe["season"] += 1
        out = out.merge(team_roe, on=["team", "season"], how="left")
    return out


def build_situation_features(
    panel: pd.DataFrame,
    rosters: pd.DataFrame,
    *,
    as_of: datetime,
    draft_picks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add team-change and same-position rookie competition features."""
    df = rosters.copy()
    pid = _col(df, "player_id", "gsis_id")
    team = _col(df, "team", "recent_team")
    if not pid or not team or "season" not in df:
        return panel
    df = df.rename(columns={pid: "player_id", team: "roster_team"})
    date = _col(df, "date", "as_of", "timestamp")
    if date:
        df[date] = pd.to_datetime(df[date], errors="coerce")
        df = df[df[date] <= pd.Timestamp(as_of)]
    latest = df.sort_values(date if date else "season").drop_duplicates(
        ["player_id", "season"], keep="last"
    )
    prior = latest[["player_id", "season", "roster_team"]].copy()
    prior["season"] += 1
    prior = prior.rename(columns={"roster_team": "prior_team"})
    current = latest[["player_id", "season", "roster_team"]]
    out = panel.merge(current, on=["player_id", "season"], how="left").merge(
        prior, on=["player_id", "season"], how="left"
    )
    out["changed_team"] = (
        (out["roster_team"].notna())
        & (out["prior_team"].notna())
        & (out["roster_team"] != out["prior_team"])
    ).astype(int)
    out["new_starting_qb"] = 0
    out["new_offensive_coordinator"] = 0
    out["competition_count"] = 0
    out["competition_draft_capital"] = 0.0
    if draft_picks is not None and not draft_picks.empty:
        picks = draft_picks.copy()
        pteam = _col(picks, "team", "recent_team")
        ppos = _col(picks, "position", "pos")
        pick = _col(picks, "pick", "draft_pick")
        if pteam and ppos and pick and "season" in picks:
            picks = picks[pd.to_numeric(picks["season"], errors="coerce") <= as_of.year]
            comp = (
                picks.groupby(["season", pteam, ppos])
                .agg(
                    competition_count=(pick, "count"),
                    competition_draft_capital=(
                        pick,
                        lambda s: float(
                            np.maximum(0, 33 - pd.to_numeric(s, errors="coerce")).sum()
                        ),
                    ),
                )
                .reset_index()
                .rename(columns={pteam: "roster_team", ppos: "position"})
            )
            out = out.drop(columns=["competition_count", "competition_draft_capital"]).merge(
                comp, on=["season", "roster_team", "position"], how="left"
            )
    return out.drop(columns=["roster_team", "prior_team"], errors="ignore")


def build_tier2_features(
    panel: pd.DataFrame,
    *,
    as_of: datetime,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build all available Tier-2 features from injected, point-in-time sources.

    ``sources`` keeps tests and offline builds deterministic. Missing sources are
    tolerated and simply leave their registered columns absent.
    """
    sources = sources or {}
    out = panel.copy()
    if "snap_counts" in sources:
        out = build_snap_features(out, sources["snap_counts"], as_of=as_of)
    if "depth_charts" in sources:
        out = build_depth_features(out, sources["depth_charts"], as_of=as_of)
    if "pbp" in sources:
        out = build_pbp_features(out, sources["pbp"], as_of=as_of)
    if "rosters" in sources:
        out = build_situation_features(
            out, sources["rosters"], as_of=as_of, draft_picks=sources.get("draft_picks")
        )
    return out
