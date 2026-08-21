"""Explosiveness / week-to-week variance features for lineup (start-sit) decisions.

A player's median weekly forecast is only half of a start/sit decision — two
players projected for the same PPR points can carry very different risk. One
might be a steady 14-16 every week (a "floor" play); the other might drop 4
one week and 28 the next (a "ceiling"/boom-bust play). Which one you should
start depends on whether you need to protect a lead or need upside to
catch up.

Everything here is computed from *prior* games only (``shift(1)`` before any
rolling window), same no-look-ahead discipline as ``features.py``.

Key columns added by ``add_variance_features``
-----------------------------------------------
ppr_std3 / ppr_std5   Rolling std of the last 3 / 5 games (NaN -> 0, needs >=2
                      games to be defined).
ppr_cv5               Coefficient of variation: ppr_std5 / max(ppr_ma5, floor).
                      Floor avoids blowing up CV for players who are barely
                      used (small denominator).
boom_rate5            Share of the last <=5 games at/above the position's
                      boom threshold (``config.Settings.boom_bust``).
bust_rate5            Share of the last <=5 games at/below the bust threshold.
explosiveness_score   0-100 composite: average of the within-position
                      percentile rank of ppr_cv5 and boom_rate5. Higher =
                      spikier role / more boom-bust; lower = steadier floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

_DEFAULT_BOOM_BUST = Settings().boom_bust
_MEAN_FLOOR = 3.0  # PPR points; guards ppr_cv5 for near-zero-usage players


def _rolling_std_prior(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=2).std()


def _rolling_rate_prior(series: pd.Series, window: int = 5) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def add_variance_features(
    panel: pd.DataFrame,
    boom_bust: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Add rolling variance / explosiveness columns to a weekly panel.

    ``panel`` must already be sorted the way ``build_weekly_panel`` leaves it
    (one row per player-game, grouped by player/season) and carry a
    ``fantasy_points_ppr``-equivalent column named ``target`` plus
    ``ppr_ma5``/``position``. Safe to call on the output of
    ``build_weekly_panel`` directly.
    """
    df = panel.copy()
    thresholds = boom_bust or _DEFAULT_BOOM_BUST

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby(["player_id", "season"], sort=False)

    df["ppr_std3"] = g["target"].apply(lambda s: _rolling_std_prior(s, 3)).values
    df["ppr_std5"] = g["target"].apply(lambda s: _rolling_std_prior(s, 5)).values
    df["ppr_std3"] = df["ppr_std3"].fillna(0.0)
    df["ppr_std5"] = df["ppr_std5"].fillna(0.0)

    if "ppr_ma5" in df.columns:
        ma5 = df["ppr_ma5"].fillna(0.0)
    else:
        ma5 = pd.Series(
            g["target"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values,
            index=df.index,
        ).fillna(0.0)
    df["ppr_cv5"] = df["ppr_std5"] / ma5.clip(lower=_MEAN_FLOOR)

    boom_hi = df["position"].map(lambda p: thresholds.get(p, (20.0, 10.0))[0])
    boom_lo = df["position"].map(lambda p: thresholds.get(p, (20.0, 10.0))[1])
    df["_boom_flag"] = (df["target"] >= boom_hi).astype(float)
    df["_bust_flag"] = (df["target"] <= boom_lo).astype(float)

    g2 = df.groupby(["player_id", "season"], sort=False)
    df["boom_rate5"] = g2["_boom_flag"].apply(lambda s: _rolling_rate_prior(s, 5)).values
    df["bust_rate5"] = g2["_bust_flag"].apply(lambda s: _rolling_rate_prior(s, 5)).values
    df["boom_rate5"] = df["boom_rate5"].fillna(0.0)
    df["bust_rate5"] = df["bust_rate5"].fillna(0.0)
    df = df.drop(columns=["_boom_flag", "_bust_flag"])

    df["cv5_pct"] = df.groupby("position")["ppr_cv5"].rank(pct=True)
    df["boom_pct"] = df.groupby("position")["boom_rate5"].rank(pct=True)
    df["explosiveness_score"] = (50.0 * (df["cv5_pct"] + df["boom_pct"])).round(1)

    return df.drop(columns=["cv5_pct", "boom_pct"])


def game_log_variance_snapshot(
    values: list[float] | np.ndarray,
    position: str,
    boom_bust: dict[str, tuple[float, float]] | None = None,
    window: int = 5,
) -> dict[str, float]:
    """Compute std/cv/boom-rate/bust-rate from a raw chronological point-total
    history (most recent last), for a single upcoming-week forecast row.

    Used at inference time where there's no panel to run a groupby over —
    e.g. scoring a player's history-to-date for next week's lineup call, or a
    prior-season game log used as a cold-start fallback in week 1.
    """
    thresholds = boom_bust or _DEFAULT_BOOM_BUST
    boom_hi, boom_lo = thresholds.get(position, (20.0, 10.0))
    arr = np.asarray(values[-window:], dtype=float)

    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    mean = float(np.mean(arr)) if len(arr) else 0.0
    cv = std / max(mean, _MEAN_FLOOR)
    boom_rate = float(np.mean(arr >= boom_hi)) if len(arr) else 0.0
    bust_rate = float(np.mean(arr <= boom_lo)) if len(arr) else 0.0

    return {"ppr_std5": std, "ppr_cv5": cv, "boom_rate5": boom_rate, "bust_rate5": bust_rate}


def fit_explosiveness_scaler(
    panel: pd.DataFrame, min_games_played: int = 3
) -> dict[str, dict[str, np.ndarray]]:
    """Fit per-position sorted reference distributions of cv5/boom_rate5,
    used to score a single new (cv5, boom_rate5) pair against the same
    percentile scale ``add_variance_features`` used at training time.

    Only rows with at least ``min_games_played`` prior games are used, so the
    reference distribution isn't dominated by early-season low-sample noise.
    """
    stable = panel[panel["games_played"] >= min_games_played]
    scaler: dict[str, dict[str, np.ndarray]] = {}
    for pos, grp in stable.groupby("position"):
        scaler[pos] = {
            "cv5": np.sort(grp["ppr_cv5"].dropna().to_numpy()),
            "boom_rate5": np.sort(grp["boom_rate5"].dropna().to_numpy()),
        }
    return scaler


def score_explosiveness(
    cv5: float, boom_rate5: float, position: str, scaler: dict[str, dict[str, np.ndarray]]
) -> float:
    """Map a single (cv5, boom_rate5) pair to the 0-100 explosiveness scale
    fit by ``fit_explosiveness_scaler``. Falls back to the 50th percentile
    for positions with no reference distribution.
    """
    ref = scaler.get(position)
    if not ref or len(ref["cv5"]) == 0:
        return 50.0
    cv_pct = np.searchsorted(ref["cv5"], cv5, side="right") / len(ref["cv5"])
    boom_pct = np.searchsorted(ref["boom_rate5"], boom_rate5, side="right") / len(ref["boom_rate5"])
    return round(50.0 * (cv_pct + boom_pct), 1)


DEFAULT_BAND_OFFSETS = {
    # Fallback if data/exports/weekly_variance_bands.csv hasn't been (re)generated
    # by scripts/analyze_weekly_variance.py. Fit on real 2023-2025 walk-forward
    # residuals; see reports/weekly_forecast_report.md for the calibration and
    # out-of-sample coverage check (0.607 vs a 0.60 target).
    "Low": (-5.21, 6.32),
    "Medium": (-5.64, 5.20),
    "High": (-6.26, 6.47),
}


def load_band_offsets() -> dict[str, tuple[float, float]]:
    from ..config import EXPORT_DIR

    path = EXPORT_DIR / "weekly_variance_bands.csv"
    if not path.exists():
        return DEFAULT_BAND_OFFSETS
    bands_df = pd.read_csv(path)
    return {
        row["tercile"]: (row["floor_offset"], row["ceiling_offset"])
        for _, row in bands_df.iterrows()
    }


def fit_tercile_thresholds(
    panel: pd.DataFrame, col: str = "explosiveness_score"
) -> dict[str, tuple[float, float]]:
    """Fixed, per-position Low/Medium/High cutoffs for ``col``, fit once on the
    full training panel. Applying these fixed cutoffs at inference time (rather
    than re-quantiling a small board of a handful of players) keeps risk_tier
    labels comparable across boards of any size, including a 12-15 player
    fantasy roster where a position group might have only 1-2 players.
    """
    thresholds: dict[str, tuple[float, float]] = {}
    for pos, grp in panel.groupby("position"):
        q = grp[col].quantile([1 / 3, 2 / 3])
        thresholds[pos] = (float(q.iloc[0]), float(q.iloc[1]))
    return thresholds


def apply_tercile(score: float, position: str, thresholds: dict[str, tuple[float, float]]) -> str:
    lo, hi = thresholds.get(position, (33.3, 66.7))
    if score < lo:
        return "Low"
    if score > hi:
        return "High"
    return "Medium"


def attach_risk_bands(
    board: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
    bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Attach ``risk_tier``/``floor``/``ceiling`` to a projected board that already
    has ``proj_points`` and ``explosiveness_score`` columns.
    """
    bands = bands or load_band_offsets()
    out = board.copy()
    out["risk_tier"] = [
        apply_tercile(score, pos, thresholds)
        for score, pos in zip(out["explosiveness_score"], out["position"], strict=True)
    ]
    default_offset = (-5.5, 6.5)
    offsets = out["risk_tier"].map(lambda t: bands.get(t, default_offset))
    out["floor"] = (out["proj_points"] + offsets.map(lambda t: t[0])).clip(lower=0).round(1)
    out["ceiling"] = (out["proj_points"] + offsets.map(lambda t: t[1])).round(1)
    return out


def explosiveness_tercile(panel: pd.DataFrame, col: str = "explosiveness_score") -> pd.Series:
    """Label each row Low/Medium/High explosiveness, within position, via terciles."""

    def _label(group: pd.Series) -> pd.Series:
        try:
            return pd.qcut(group, 3, labels=["Low", "Medium", "High"])
        except ValueError:
            # Not enough distinct values to cut into 3 bins.
            return pd.Series(["Medium"] * len(group), index=group.index)

    return panel.groupby("position")[col].transform(_label).astype(str)


def validate_variance_persistence(panel: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Split-half check: is a player's early-season CV correlated with their
    late-season CV? A real "explosiveness" trait should persist within a
    season; if the split-half correlation is ~0, the metric is just noise.

    Returns one row per position with n (qualifying player-seasons) and the
    Spearman correlation between first-half and second-half coefficient of
    variation.
    """
    rows = []
    df = panel.copy()
    for pos, g in df.groupby("position"):
        pair_rows = []
        for (_pid, _season), sub in g.groupby(["player_id", "season"]):
            sub = sub.sort_values("week")
            if len(sub) < min_games:
                continue
            mid = len(sub) // 2
            first, second = sub.iloc[:mid], sub.iloc[mid:]
            f_mean, s_mean = first["target"].mean(), second["target"].mean()
            f_std, s_std = first["target"].std(), second["target"].std()
            if pd.isna(f_std) or pd.isna(s_std):
                continue
            f_cv = f_std / max(f_mean, _MEAN_FLOOR)
            s_cv = s_std / max(s_mean, _MEAN_FLOOR)
            pair_rows.append((f_cv, s_cv))

        if len(pair_rows) < 8:
            rows.append({"position": pos, "n": len(pair_rows), "spearman_r": float("nan")})
            continue

        pairs = pd.DataFrame(pair_rows, columns=["first_half_cv", "second_half_cv"])
        r = pairs["first_half_cv"].corr(pairs["second_half_cv"], method="spearman")
        rows.append({"position": pos, "n": len(pairs), "spearman_r": round(float(r), 3)})

    return pd.DataFrame(rows)


def validate_variance_predicts_error(cv: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Does pre-game explosiveness correlate with that week's forecast error?

    Joins the walk-forward CV predictions (``cv`` from ``walk_forward_weekly_cv``)
    back to the panel's ``explosiveness_score`` as of the same (player, season,
    week) row (that score is already computed from strictly prior games) and
    reports, per model, the Spearman correlation between explosiveness and
    absolute error. A positive correlation validates using wider floor/ceiling
    bands for high-explosiveness players.
    """
    key_cols = ["player_id", "season", "week"]
    score_cols = key_cols + ["explosiveness_score"]
    merged = cv.merge(
        panel[score_cols].rename(columns={"season": "eval_season"}),
        on=["player_id", "eval_season", "week"],
        how="inner",
    )
    rows = []
    for name, grp in merged.groupby("model"):
        r = grp["explosiveness_score"].corr(grp["abs_error"], method="spearman")
        rows.append(
            {"Model": name, "N": len(grp), "Spearman(explosiveness, abs_error)": round(float(r), 3)}
        )
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


def calibrate_bands(
    cv: pd.DataFrame,
    panel: pd.DataFrame,
    lo_q: float = 0.20,
    hi_q: float = 0.80,
) -> pd.DataFrame:
    """Empirically calibrate floor/ceiling multipliers per explosiveness tercile.

    For each tercile, fits the residual quantiles of (actual - predicted) using
    the best-performing model's out-of-sample CV rows, so the bands reflect
    real historical spread rather than an assumed Gaussian shape. Returns one
    row per tercile with the additive floor/ceiling offsets (PPR points) to
    apply on top of a point projection.
    """
    key_cols = ["player_id", "season", "week"]
    tiers = panel[key_cols + ["explosiveness_score"]].copy()
    tiers["tercile"] = explosiveness_tercile(panel)
    merged = cv.merge(
        tiers.rename(columns={"season": "eval_season"}),
        on=["player_id", "eval_season", "week"],
        how="inner",
    )
    rows = []
    for tercile, grp in merged.groupby("tercile"):
        resid = grp["actual"] - grp["predicted"]
        rows.append(
            {
                "tercile": tercile,
                "n": len(grp),
                "floor_offset": round(float(resid.quantile(lo_q)), 2),
                "ceiling_offset": round(float(resid.quantile(hi_q)), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("tercile").reset_index(drop=True)


def coverage_check(
    cv: pd.DataFrame,
    panel: pd.DataFrame,
    bands: pd.DataFrame,
    lo_q: float = 0.20,
    hi_q: float = 0.80,
) -> float:
    """Fraction of held-out actuals that fall within [predicted+floor_offset,
    predicted+ceiling_offset]. Should land near ``hi_q - lo_q`` (e.g. 0.60 for
    the p20/p80 default) if the calibration in ``calibrate_bands`` generalizes
    rather than overfitting the same rows it was fit on.
    """
    key_cols = ["player_id", "season", "week"]
    tiers = panel[key_cols + ["explosiveness_score"]].copy()
    tiers["tercile"] = explosiveness_tercile(panel)
    merged = cv.merge(
        tiers.rename(columns={"season": "eval_season"}),
        on=["player_id", "eval_season", "week"],
        how="inner",
    ).merge(bands, on="tercile", how="left")
    lo = merged["predicted"] + merged["floor_offset"]
    hi = merged["predicted"] + merged["ceiling_offset"]
    inside = (merged["actual"] >= lo) & (merged["actual"] <= hi)
    return float(inside.mean())
