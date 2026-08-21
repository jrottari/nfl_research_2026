"""CSV export, formatted so each file converts cleanly to a Google Sheets Table.

The rules that matter for `Format > Convert to table`: exactly one header row
starting at A1, no index column, no blank rows or columns, no totals row, and
one consistent type per column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRETTY: dict[str, str] = {
    "rank_ppr": "Rank (PPR)",
    "rank_half": "Rank (Half)",
    "rank_std": "Rank (Std)",
    "rank_ppg": "Rank (PPG)",
    "rank_delta": "Rank Delta",
    "flex_rank": "FLEX Rank",
    "player_name": "Player",
    "player_id": "Player ID",
    "position": "Pos",
    "pos_label": "Pos Rank Label",
    "team": "Team",
    "teams": "Teams",
    "games": "G",
    "games_active": "G Active",
    "points_ppr": "PPR Pts",
    "ppg_ppr": "PPR PPG",
    "points_half": "Half PPR Pts",
    "ppg_half": "Half PPR PPG",
    "points_std": "Std Pts",
    "ppg_std": "Std PPG",
    "pos_rank": "Pos Rank",
    "pos_rank_ppg": "Pos Rank (PPG)",
    "vor_ppg": "VOR PPG",
    "vorp_total": "VORP Total",
    "tier": "Tier",
    "replacement_ppg": "Replacement PPG",
    "floor": "Floor (P10)",
    "ceiling": "Ceiling (P90)",
    "median_week": "Median Wk",
    "stdev": "Std Dev",
    "cv": "CV",
    "best_week": "Best Wk",
    "worst_week": "Worst Wk",
    "boom_weeks": "Boom Wks",
    "boom_rate": "Boom Rate",
    "bust_weeks": "Bust Wks",
    "bust_rate": "Bust Rate",
    "starter_weeks": "Starter Wks",
    "starter_week_rate": "Starter Wk Rate",
    "top5_weeks": "Top 5 Wks",
    "best_pos_finish": "Best Pos Finish",
    "touches": "Touches",
    "touch_pg": "Touches/G",
    "opportunities": "Opportunities",
    "opp_pg": "Opp/G",
    "target_share": "Target Share (Avg)",
    "scrim_yards": "Scrim Yds",
    "scrim_yds_pg": "Scrim Yds/G",
    "total_yards": "Total Yds",
    "total_yds_pg": "Total Yds/G",
    "total_tds": "Total TDs",
    "tds_pg": "TDs/G",
    "first_downs": "First Downs",
    "turnovers": "Turnovers",
    "fumbles_lost": "Fumbles Lost",
    "two_pt": "2PT Conv",
    "pts_per_touch": "Pts/Touch",
    "pts_per_opp": "Pts/Opp",
    "yards_per_touch": "Yds/Touch",
    "ypc": "YPC",
    "ypr": "YPR",
    "ypt": "YPT",
    "catch_rate": "Catch Rate",
    "td_rate": "TD Rate",
    "adot": "aDOT",
    "yac_per_rec": "YAC/Rec",
    "completions": "Comp",
    "attempts": "Att",
    "passing_yards": "Pass Yds",
    "passing_tds": "Pass TD",
    "interceptions": "INT",
    "comp_pct": "Comp %",
    "ypa": "YPA",
    "air_yards_pa": "Air Yds/Att",
    "td_int_ratio": "TD:INT",
    "sack_rate": "Sack Rate",
    "carries": "Carries",
    "rushing_yards": "Rush Yds",
    "rushing_tds": "Rush TD",
    "targets": "Targets",
    "receptions": "Rec",
    "receiving_yards": "Rec Yds",
    "receiving_tds": "Rec TD",
    "week": "Week",
    "opponent": "Opp",
    "week_pos_rank": "Wk Pos Rank",
    "fantasy_points_ppr": "PPR Pts",
    "fantasy_points": "Std Pts",
}

# Exported as decimals (0.6512) so Sheets' percent format is exact.
RATE_COLS = frozenset(
    {
        "catch_rate",
        "boom_rate",
        "bust_rate",
        "starter_week_rate",
        "comp_pct",
        "td_rate",
        "target_share",
        "cv",
        "sack_rate",
    }
)


def sheets_ready(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Flatten, round and rename a frame for import into Google Sheets."""
    out = df.copy().reset_index(drop=True)

    for col in out.columns:
        if col in RATE_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)

    # Stop Sheets reading a leading =, +, - or @ as a formula.
    for col in out.select_dtypes(include="object").columns:
        text = out[col].astype(str).replace({"nan": "", "None": "", "<NA>": ""})
        out[col] = text.where(~text.str.startswith(("=", "+", "@")), "'" + text)

    out.columns = [PRETTY.get(c, c.replace("_", " ").title()) for c in out.columns]
    return out


def export_csv(df: pd.DataFrame, filename: str, folder: Path | str) -> Path:
    """Write one Sheets-ready CSV and return the path."""
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    sheets_ready(df).to_csv(path, index=False, encoding="utf-8", na_rep="")
    return path


def export_all(
    boards: dict[str, pd.DataFrame],
    folder: Path | str,
    only: list[str] | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Write every board in ``boards`` (or just the filenames listed in ``only``)."""
    written: list[Path] = []
    for filename, frame in boards.items():
        if only and filename not in only:
            continue
        path = export_csv(frame, filename, folder)
        written.append(path)
        if not quiet:
            print(f"{len(frame):>6,} rows  ->  {path}")
    return written


def choose_export_folder(default: Path | str, use_dialog: bool = True) -> Path:
    """Ask for an export folder: native dialog -> typed input -> default."""
    default = Path(default).expanduser()
    if use_dialog:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            picked = filedialog.askdirectory(
                title="Choose an export folder", initialdir=str(default.parent)
            )
            root.destroy()
            if picked:
                return Path(picked)
        except Exception as exc:
            print(f"(folder dialog unavailable: {exc.__class__.__name__} - type it instead)")
    try:
        typed = input(f"Export folder [{default}]: ").strip()
    except Exception:
        typed = ""
    return Path(typed).expanduser() if typed else default
