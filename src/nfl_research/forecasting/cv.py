"""Walk-forward cross-validation for fantasy football forecasting models.

For each evaluation year T:
  - Training panel: all rows where season < T, optionally restricted to players
                    who were top-N PPR in the season before each training row
                    (``top_n_train_filter``).  This keeps the model focused on
                    the elite-player distribution we actually care about.
  - Test panel:     rows where season == T, restricted to players who were
                    in the top-N PPR finishers in season T-1 (our real target
                    population)
  - Each model is fit on training, scored on test

Returns a long-format DataFrame with one row per (model, eval_season, player).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import top_n_in_season
from .features import feature_matrix


@dataclass
class CVResult:
    model_name: str
    eval_season: int
    player_id: str
    player_name: str
    position: str
    actual: float
    predicted: float
    lag1: float  # random-walk prediction, for reference


def _filter_train_to_top_n(train: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep only training rows where the player was top-N PPR in the prior season.

    Each training row already carries ``points_ppr_lag1`` (the player's total
    from the season before).  Ranking those within each training season gives us
    the set of players who were elite heading into that year — exactly the
    population we want the model to learn from.
    """
    parts = []
    for _season, group in train.groupby("season", sort=False):
        top = group.nlargest(n, "points_ppr_lag1")
        parts.append(top)
    return pd.concat(parts, ignore_index=True) if parts else train.iloc[:0]


def walk_forward_cv(
    panel: pd.DataFrame,
    models: list,
    eval_seasons: list[int],
    top_n_filter: int = 200,
    top_n_train_filter: int = 0,
    min_train_seasons: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk-forward CV across ``eval_seasons``.

    Args:
        panel:               Output of ``features.build_panel()``.
        models:              List of model instances (fit/predict interface).
        eval_seasons:        Years to evaluate. Training uses all earlier seasons.
        top_n_filter:        Only score players who were top-N PPR in season T-1.
                             Set to 0 to score all players with data.
        top_n_train_filter:  Also restrict *training* rows to players who were
                             top-N in the season before each training row.
                             0 = disabled (use all players).
        min_train_seasons:   Skip eval years where there are fewer distinct
                             training seasons than this.
        verbose:             Print progress per eval year.

    Returns:
        DataFrame with columns: model, eval_season, player_id, player_name,
        position, actual, predicted, lag1, error, abs_error, sq_error.
    """
    all_rows: list[dict] = []

    for eval_season in sorted(eval_seasons):
        train = panel[panel["season"] < eval_season].copy()
        test = panel[panel["season"] == eval_season].copy()

        if train["season"].nunique() < min_train_seasons:
            if verbose:
                print(f"  [skip {eval_season}] only {train['season'].nunique()} training seasons")
            continue

        if top_n_train_filter > 0:
            train = _filter_train_to_top_n(train, top_n_train_filter)

        if test.empty:
            if verbose:
                print(f"  [skip {eval_season}] no test data")
            continue

        # Filter to players who were top-N in the prior season
        if top_n_filter > 0:
            prior_season = eval_season - 1
            top_ids = top_n_in_season(
                panel[["player_id", "season", "target"]].rename(
                    columns={"target": "fantasy_points_ppr"}
                ),
                prior_season,
                n=top_n_filter,
            )
            # Also accept players who appear in the raw long-form with prior season data
            # (covers players who played but weren't on the panel yet due to lag reqs)
            test = test[test["player_id"].isin(top_ids)].copy()

        if test.empty:
            if verbose:
                print(f"  [skip {eval_season}] no test players after top-{top_n_filter} filter")
            continue

        X_train = feature_matrix(train)
        y_train = train["target"]
        X_test = feature_matrix(test)
        y_test = test["target"]

        if verbose:
            print(f"\n[{eval_season}]  train rows: {len(train)}  test players: {len(test)}")

        for model in models:
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception as exc:
                if verbose:
                    print(f"    [{model.name}] ERROR: {exc}")
                continue

            elapsed = time.time() - t0
            if verbose:
                err = np.mean(np.abs(preds - y_test.values))
                print(f"    [{model.name:25s}]  MAE={err:.1f}   ({elapsed:.1f}s)")

            lag1_vals = X_test["points_ppr_lag1"].fillna(0).values

            for i, (_idx, row) in enumerate(test.iterrows()):
                all_rows.append(
                    {
                        "model": model.name,
                        "eval_season": eval_season,
                        "player_id": row["player_id"],
                        "player_name": row.get("player_name", ""),
                        "position": row.get("position", ""),
                        "actual": float(y_test.iloc[i]),
                        "predicted": float(preds[i]),
                        "lag1": float(lag1_vals[i]),
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"] = df["error"] ** 2
    df["lag1_error"] = df["lag1"] - df["actual"]
    df["lag1_abs_error"] = df["lag1_error"].abs()
    return df
