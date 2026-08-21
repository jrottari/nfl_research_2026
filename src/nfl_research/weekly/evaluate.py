"""Scoring and reporting for weekly CV results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_summary_table(cv: pd.DataFrame) -> pd.DataFrame:
    """One row per model, sorted by MAE."""
    if cv.empty:
        return pd.DataFrame()

    baseline_mae = cv[cv["model"] == cv["model"].iloc[0]]["baseline_abs_error"].mean()

    rows = []
    for name, grp in cv.groupby("model"):
        mae = float(grp["abs_error"].mean())
        rmse = float(np.sqrt(grp["sq_error"].mean()))
        bias = float(grp["error"].mean())
        r = float(grp[["actual", "predicted"]].corr().iloc[0, 1])
        skill = 1.0 - mae / baseline_mae if baseline_mae > 0 else float("nan")
        rows.append(
            {
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "Bias": round(bias, 2),
                "Pearson r": round(r, 3),
                "Skill vs 3G-Avg": round(skill, 3),
                "N game-weeks": len(grp),
            }
        )

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def weekly_position_breakdown(cv: pd.DataFrame) -> pd.DataFrame:
    return (
        cv.groupby(["model", "position"])["abs_error"]
        .mean()
        .round(2)
        .unstack("position")
        .reset_index()
    )


def weekly_by_week(cv: pd.DataFrame) -> pd.DataFrame:
    """MAE by week number — shows if models degrade late in season."""
    return cv.groupby(["model", "week"])["abs_error"].mean().round(2).unstack("week").reset_index()


def boom_bust_accuracy(cv: pd.DataFrame, boom_thresh: float = 20.0) -> pd.DataFrame:
    """For each model, how well does it identify boom weeks (> boom_thresh PPR)?"""
    rows = []
    for name, grp in cv.groupby("model"):
        actual_boom = grp["actual"] >= boom_thresh
        predicted_boom = grp["predicted"] >= boom_thresh
        tp = (actual_boom & predicted_boom).sum()
        fp = (~actual_boom & predicted_boom).sum()
        fn = (actual_boom & ~predicted_boom).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "Model": name,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
                "Boom N": int(actual_boom.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def print_weekly_report(cv: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("WEEKLY GAME FORECAST — WALK-FORWARD CV SUMMARY")
    print("=" * 65)
    print(weekly_summary_table(cv).to_string(index=False))
    print("\nPOSITION BREAKDOWN (MAE):")
    print(weekly_position_breakdown(cv).to_string(index=False))
    print("\nBOOM WEEK DETECTION (>=20 PPR):")
    print(boom_bust_accuracy(cv).to_string(index=False))
    print("=" * 65)
