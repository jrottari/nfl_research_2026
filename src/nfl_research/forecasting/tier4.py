"""Tier-4 advanced, injury, contract, and FTN features."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, *names: str) -> str | None:
    return next((c for c in names if c in df), None)


def _prior_summary(
    panel: pd.DataFrame, raw: pd.DataFrame, aliases: dict[str, tuple[str, ...]], as_of: datetime
) -> pd.DataFrame:
    df = raw.copy()
    pid = _col(df, "player_id", "gsis_id", "player_gsis_id")
    if not pid or "season" not in df:
        return panel
    df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year - 1]
    rename: dict[str, str] = {pid: "player_id"}
    for canonical, choices in aliases.items():
        actual = _col(df, *choices)
        if actual:
            rename[actual] = canonical
    values = [v for v in rename.values() if v != "player_id"]
    if not values:
        return panel
    df = df[[pid, "season", *[c for c in rename if c != pid]]].rename(columns=rename)
    agg = df.groupby(["player_id", "season"], as_index=False)[values].mean(numeric_only=True)
    agg["season"] += 1
    return panel.merge(agg, on=["player_id", "season"], how="left")


def build_injury_features(
    panel: pd.DataFrame, injuries: pd.DataFrame, *, as_of: datetime
) -> pd.DataFrame:
    """Use only injury reports strictly before the cutoff and from prior seasons."""
    df = injuries.copy()
    pid = _col(df, "player_id", "gsis_id")
    date = _col(df, "date", "report_date")
    if not pid or "season" not in df:
        return panel
    if date:
        df[date] = pd.to_datetime(df[date], errors="coerce")
        df = df[df[date] < pd.Timestamp(as_of)]
    df = df[pd.to_numeric(df["season"], errors="coerce") <= as_of.year - 1]
    status = _col(df, "report_status", "game_status", "status")
    df["missed"] = (
        df[status].astype(str).str.lower().isin({"out", "inactive", "ir"}).astype(int)
        if status
        else 0
    )
    agg = (
        df.groupby([pid, "season"])
        .agg(injury_weeks_lag1=("missed", "size"), games_missed_lag1=("missed", "sum"))
        .reset_index()
        .rename(columns={pid: "player_id"})
    )
    agg["season"] += 1
    return panel.merge(agg, on=["player_id", "season"], how="left")


def build_contract_features(
    panel: pd.DataFrame, contracts: pd.DataFrame, *, as_of: datetime
) -> pd.DataFrame:
    """Join contracts only when an explicit signing date proves they existed at cutoff."""
    df = contracts.copy()
    pid = _col(df, "player_id", "gsis_id")
    signed = _col(df, "signed_date", "date_signed", "contract_date")
    if not pid or not signed:
        return panel  # no snapshot semantics means no admissible feature
    df[signed] = pd.to_datetime(df[signed], errors="coerce")
    df = df[df[signed] <= pd.Timestamp(as_of)].sort_values(signed).drop_duplicates(pid, keep="last")
    guaranteed = _col(df, "guaranteed", "guaranteed_money")
    end = _col(df, "contract_end", "end_year")
    keep = pd.DataFrame({"player_id": df[pid]})
    keep["contract_guaranteed"] = (
        pd.to_numeric(df[guaranteed], errors="coerce") if guaranteed else np.nan
    )
    if end:
        end_year = (
            pd.to_datetime(df[end], errors="coerce").dt.year
            if not pd.api.types.is_numeric_dtype(df[end])
            else pd.to_numeric(df[end], errors="coerce")
        )
        keep["contract_years_remaining"] = (end_year - as_of.year).clip(lower=0)
    else:
        keep["contract_years_remaining"] = np.nan
    return panel.merge(keep, on="player_id", how="left")


def build_tier4_features(
    panel: pd.DataFrame, *, as_of: datetime, sources: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    sources = sources or {}
    out = panel.copy()
    if "pfr" in sources:
        out = _prior_summary(
            out,
            sources["pfr"],
            {
                "broken_tackles": ("broken_tackles", "brk_tkl"),
                "drop_rate": ("drop_pct", "drop_rate"),
                "yac_per_rec": ("yac_per_rec",),
                "pressure_rate": ("pressure_pct", "pressure_rate"),
            },
            as_of,
        )
    if "ftn" in sources:
        out = _prior_summary(
            out,
            sources["ftn"],
            {
                "ftn_xyac": ("xyac", "x_yac"),
                "ftn_no_huddle_rate": ("no_huddle_rate", "no_huddle"),
            },
            as_of,
        )
    if "injuries" in sources:
        out = build_injury_features(out, sources["injuries"], as_of=as_of)
    if "contracts" in sources:
        out = build_contract_features(out, sources["contracts"], as_of=as_of)
    return out
