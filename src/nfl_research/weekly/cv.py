"""Walk-forward weekly CV.

For each eval season Y and each eval week W (≥ min_prior_games + 1):
  Training:
    - All prior seasons (every week)
    - Current season Y, weeks 1 .. W-1
  Test:
    - Current season Y, week W
    - Restricted to players who were top-N PPR in the prior full season

This mirrors real in-season decisions: you know everything through last week.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from .features import weekly_feature_matrix


def walk_forward_weekly_cv(
    panel: pd.DataFrame,
    models: list,
    eval_seasons: list[int],
    top_n_filter: int = 150,
    min_prior_games: int = 2,
    min_train_rows: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward CV over weekly game predictions.

    Args:
        panel:            Output of ``build_weekly_panel()``.
        models:           Weekly model instances.
        eval_seasons:     Seasons to evaluate (e.g. [2022, 2023, 2024]).
        top_n_filter:     Score only players who were top-N PPR the prior season.
        min_prior_games:  Skip weeks where the player has fewer than this many
                          prior games this season.
        min_train_rows:   Skip eval weeks with fewer training rows.
        verbose:          Print progress.

    Returns:
        Long-format DataFrame: model, eval_season, week, player fields,
        actual, predicted, baseline (rolling_mean_3g).
    """
    all_rows: list[dict] = []

    for eval_season in sorted(eval_seasons):
        prior_season = eval_season - 1

        # Who was top-N last full season?
        if top_n_filter > 0:
            prior = panel[panel["season"] == prior_season].copy()
            prior_totals = (
                prior.groupby("player_id")["target"].sum()
                .nlargest(top_n_filter)
            )
            eligible_ids = set(prior_totals.index)
        else:
            eligible_ids = set(panel["player_id"].unique())

        eval_weeks = sorted(
            panel[panel["season"] == eval_season]["week"].dropna().unique()
        )

        for week in eval_weeks:
            # Training: all prior seasons + current season up to week-1
            train_prior   = panel[panel["season"] < eval_season]
            train_current = panel[(panel["season"] == eval_season) & (panel["week"] < week)]
            train = pd.concat([train_prior, train_current], ignore_index=True)

            # Test: current season, this week, eligible players with enough history
            test = panel[
                (panel["season"] == eval_season) &
                (panel["week"] == week) &
                (panel["player_id"].isin(eligible_ids)) &
                (panel["games_played"] >= min_prior_games)
            ].copy()

            if len(train) < min_train_rows or test.empty:
                continue

            X_train = weekly_feature_matrix(train)
            y_train = train["target"]
            X_test  = weekly_feature_matrix(test)
            y_test  = test["target"]

            for model in models:
                t0 = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                except Exception as exc:
                    if verbose and week == eval_weeks[0]:
                        print(f"    [{model.name}] ERROR wk{week}: {exc}")
                    continue

                baseline = X_test["ppr_ma3"].fillna(X_test["ppr_lag1"].fillna(0)).values

                for i, (_, row) in enumerate(test.iterrows()):
                    all_rows.append({
                        "model":       model.name,
                        "eval_season": eval_season,
                        "week":        int(week),
                        "player_id":   row["player_id"],
                        "player_name": row.get("player_name", ""),
                        "position":    row.get("position", ""),
                        "actual":      float(y_test.iloc[i]),
                        "predicted":   float(preds[i]),
                        "baseline":    float(baseline[i]),
                    })

            if verbose and week % 4 == 0:
                n_models = len({r["model"] for r in all_rows if r["eval_season"] == eval_season and r["week"] == week})
                print(f"  [{eval_season} wk{week:02d}]  train={len(train):,}  test={len(test)}  models={n_models}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["error"]             = df["predicted"] - df["actual"]
    df["abs_error"]         = df["error"].abs()
    df["sq_error"]          = df["error"] ** 2
    df["baseline_error"]    = df["baseline"] - df["actual"]
    df["baseline_abs_error"] = df["baseline_error"].abs()
    return df
