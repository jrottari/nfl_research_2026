"""Scoring, comparison tables, and reporting for CV results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_model(cv: pd.DataFrame, model_name: str) -> dict:
    """Aggregate CV metrics for one model across all eval seasons."""
    sub = cv[cv["model"] == model_name]
    rw = cv[cv["model"] == "random_walk"]

    if sub.empty:
        return {}

    mae = float(sub["abs_error"].mean())
    rmse = float(np.sqrt(sub["sq_error"].mean()))
    bias = float(sub["error"].mean())
    corr = float(sub[["actual", "predicted"]].corr().iloc[0, 1])

    rw_mae = float(rw["lag1_abs_error"].mean()) if not rw.empty else float("nan")
    skill = 1.0 - mae / rw_mae if rw_mae > 0 else float("nan")

    # Year-by-year MAE
    yearly = sub.groupby("eval_season")["abs_error"].mean()

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson_r": corr,
        "rw_mae": rw_mae,
        "skill_score": skill,  # positive = better than random walk
        "n_player_seasons": len(sub),
        "n_eval_seasons": sub["eval_season"].nunique(),
        "yearly_mae": yearly.to_dict(),
    }


def summary_table(cv: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-model summary suitable for printing or export."""
    rows = []
    for name in cv["model"].unique():
        m = score_model(cv, name)
        rows.append(
            {
                "Model": m["model"],
                "MAE": round(m["mae"], 2),
                "RMSE": round(m["rmse"], 2),
                "Bias": round(m["bias"], 2),
                "Pearson r": round(m["pearson_r"], 3),
                "Skill vs RW": round(m["skill_score"], 3),
                "N player-seasons": m["n_player_seasons"],
            }
        )
    df = pd.DataFrame(rows).sort_values("MAE")
    df["Rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def position_breakdown(cv: pd.DataFrame) -> pd.DataFrame:
    """MAE per model per position."""
    return (
        cv.groupby(["model", "position"])["abs_error"]
        .mean()
        .round(2)
        .unstack("position")
        .reset_index()
        .sort_values("model")
    )


def yearly_mae(cv: pd.DataFrame) -> pd.DataFrame:
    """MAE per model per eval season - useful for spotting anomaly years."""
    return (
        cv.groupby(["model", "eval_season"])["abs_error"]
        .mean()
        .round(2)
        .unstack("eval_season")
        .reset_index()
    )


def print_report(cv: pd.DataFrame) -> None:
    """Print a human-readable CV report to stdout."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD CV SUMMARY")
    print("=" * 70)

    tbl = summary_table(cv)
    print(tbl.to_string(index=False))

    print("\nPOSITION BREAKDOWN (MAE):")
    print(position_breakdown(cv).to_string(index=False))

    print("\nYEAR-BY-YEAR MAE:")
    print(yearly_mae(cv).to_string(index=False))

    # Best model
    best = tbl.iloc[0]["Model"]
    best_skill = tbl.iloc[0]["Skill vs RW"]
    rw_mae = tbl[tbl["Model"] == "random_walk"]["MAE"].values
    rw_str = f" (RW MAE={rw_mae[0]:.1f})" if len(rw_mae) else ""
    print(f"\nBest model: {best}  |  skill score vs random walk: {best_skill:+.3f}{rw_str}")
    print("=" * 70)


def make_forecast_df(
    predictions: dict[str, np.ndarray],
    meta: pd.DataFrame,
    best_model: str | None = None,
) -> pd.DataFrame:
    """Assemble a clean forecast DataFrame for export.

    Args:
        predictions: {model_name: array of predictions}, same order as ``meta``.
        meta:        DataFrame with player_id, player_name, position, lag1 columns.
        best_model:  If given, that model's predictions appear as the primary column.
    """
    out = meta[["player_id", "player_name", "position"]].copy().reset_index(drop=True)
    out["rw_forecast"] = meta["points_ppr_lag1"].fillna(0).values

    for name, preds in predictions.items():
        col = "forecast" if name == best_model else f"forecast_{name}"
        out[col] = np.round(preds, 1)

    if best_model and "forecast" in out.columns:
        out["vs_rw"] = (out["forecast"] - out["rw_forecast"]).round(1)

    return out.sort_values("rw_forecast", ascending=False).reset_index(drop=True)
