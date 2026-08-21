# Within-Season Weekly Forecast — Model Conclusions & Explosiveness Metric

*Real, cached nflverse data, 2021-2025 (`data/cache/raw_player_weekly_*.parquet`,
`raw_player_season_*.parquet`) — not synthetic fixtures. Reproduce with
`python scripts/run_weekly_forecast.py` and `python scripts/analyze_weekly_variance.py`.*

This closes out the within-season (week-to-week) forecasting work that sits
alongside the preseason tiered forecasting system (`RESULTS.md`,
`reports/model_report_2026.md`). It answers three things:

1. Which model should actually generate the weekly point projection?
2. Is a player's week-to-week volatility ("explosiveness") a real, usable
   signal — and does it actually help lineup decisions, or is it noise?
3. What's the one command to run each week? → `scripts/weekly_lineup.py`

---

## 1. Model comparison — walk-forward CV, 2023-2025

Six models, three eval seasons (2023/2024/2025), trained walk-forward (all
prior seasons + current-season weeks before the test week), scored against
the top-150 PPR players from the prior season — 5,108 player-weeks total.

| Model | MAE | RMSE | Bias | Pearson r | Skill vs 3G-avg |
|---|---:|---:|---:|---:|---:|
| **Ridge (full feature set)** | **5.88** | 7.66 | -0.25 | 0.474 | **+7.4%** |
| XGBoost (full feature set) | 5.91 | 7.65 | -0.11 | 0.476 | +6.9% |
| Season-to-date average | 6.04 | 7.89 | -0.04 | 0.452 | +4.9% |
| Rolling 3-game average (baseline) | 6.35 | 8.32 | +0.08 | 0.424 | 0.0% |
| Opponent-adjusted season avg | 6.41 | 8.45 | +0.09 | 0.406 | -0.8% |
| Weighted rolling blend (0.5/0.3/0.2) | 6.55 | 8.62 | +0.16 | 0.399 | -3.0% |

**Winner: Ridge**, statistically indistinguishable from XGBoost (5.88 vs 5.91
MAE — well inside noise at this sample size). `scripts/weekly_lineup.py` ships
Ridge as the production model: same accuracy as XGBoost, no extra dependency,
and coefficients are inspectable if something looks wrong on a given week.

**Two results worth flagging because they're counter to the naive
prior-work assumption:**

- **The opponent-matchup adjustment makes things worse, not better**
  (-0.8% skill — *below* the plain 3-game-average baseline). Scaling a
  player's season average by "how many PPR points has this defense allowed
  to this position so far" adds noise rather than signal at this level of
  aggregation. Defense-vs-position through a handful of games is itself a
  noisy, small-sample statistic, and multiplying two noisy signals together
  compounds the noise rather than cancelling it. This isn't in the production
  model.
- **The hand-tuned recency blend (0.5×lag1 + 0.3×ma3 + 0.2×season-avg)
  underperforms both a plain 3-game average and a plain season average.**
  Guessing weights by intuition loses to either extreme (recency-only or
  full-season-only) once you're not fitting the blend to actual data — which
  is exactly what Ridge does instead of hand-picking coefficients.

### Position breakdown (MAE)

| Model | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| Ridge | 6.35 | 5.78 | 5.94 | **5.31** |
| XGBoost | 6.37 | 5.79 | 5.98 | 5.36 |
| Season avg | 6.51 | 5.92 | 6.10 | 5.46 |
| Rolling 3-game | 6.78 | 6.15 | 6.47 | 5.84 |
| Opp-adjusted | 7.17 | 6.18 | 6.40 | 5.93 |
| Weighted rolling | 7.05 | 6.23 | 6.69 | 6.11 |

QB is the hardest position to pin down week to week (highest MAE everywhere);
TE is the easiest in absolute terms, though that's partly a low-scoring-floor
artifact (smaller point totals → smaller absolute errors), not necessarily
"more predictable."

### Boom-week detection (≥20 PPR) — the gap this report exists to close

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Weighted rolling | 0.367 | 0.275 | 0.315 |
| Rolling 3-game | 0.387 | 0.265 | 0.315 |
| Opponent-adjusted | 0.359 | 0.221 | 0.274 |
| Season avg | 0.415 | 0.201 | 0.271 |
| **XGBoost** | **0.481** | 0.093 | 0.156 |
| **Ridge** | 0.476 | 0.082 | 0.139 |

This is the headline reason a separate explosiveness metric earns its place.
**The model with the lowest average error (Ridge) is also the worst at
flagging boom weeks** — 8.2% recall vs 27.5% for a dumb 3-game average.
Minimizing MAE trains a model to shrink toward the mean, which is *correct*
for the median week but actively suppresses the outlier weeks that decide
whether a boom-or-bust flex play should start. A point forecast alone cannot
tell you which of two similarly-projected players is the one likely to
either blow up or disappear — you need a second, explicitly variance-shaped
signal for that. That's section 2.

---

## 2. Explosiveness / variance metric

Implemented in `nfl_research.weekly.variance`. Everything is computed from
*prior* games only (no look-ahead), same discipline as the point-forecast
features.

| Column | Definition |
|---|---|
| `ppr_std5` | Rolling std of the last ≤5 games (needs ≥2 games to be defined) |
| `ppr_cv5` | Coefficient of variation: `ppr_std5 / max(ppr_ma5, 3.0)` — the floor keeps near-zero-usage players from producing an exploding ratio |
| `boom_rate5` / `bust_rate5` | Share of the last ≤5 games at/above the position's boom threshold, or at/below the bust threshold (league's own `Settings.boom_bust`) |
| `explosiveness_score` | 0-100: `50 × (percentile_rank(cv5, within position) + percentile_rank(boom_rate5, within position))` |

### Is it real, or noise? Split-half persistence within a season

For every player-season with ≥8 games, correlate the coefficient of
variation computed on the first half of their games against the second half.
A real trait should survive the split; pure noise correlates near zero.

| Position | n (player-seasons) | Spearman r (1st half CV vs 2nd half CV) |
|---|---:|---:|
| RB | 455 | **0.296** |
| WR | 742 | **0.269** |
| TE | 372 | 0.073 |
| QB | 170 | 0.072 |

**Explosiveness is a real, moderately persistent trait for RB and WR** — a
running back or receiver who has been boom-or-bust through the first half of
a season tends to keep being boom-or-bust in the second half. It's much
weaker for QB and TE: QB variance looks driven more by that week's game
script/opponent than by a stable player identity, and TE variance is noisier
at the smaller sample sizes TEs get. **Practical takeaway: trust the
explosiveness score more for RB/WR start-sit calls than for QB/TE ones.**

### Does it explain what the point models miss?

Spearman correlation between pre-game `explosiveness_score` and that week's
absolute forecast error, across all three eval seasons:

| Model | N | Spearman(explosiveness, abs_error) |
|---|---:|---:|
| Season avg | 5,108 | 0.082 |
| Ridge | 5,108 | 0.080 |
| XGBoost | 5,108 | 0.075 |

Small but positive and consistent across every model type — high-explosiveness
players really are harder to forecast, and no model (including the tree
model that could in principle learn nonlinear volatility patterns) has
absorbed that into its point estimate. This is exactly why floor/ceiling
bands should widen for high-explosiveness players rather than applying one
fixed band to everyone.

### Floor/ceiling calibration — fit on real residuals, validated out-of-sample

Rather than assume a Gaussian shape, floor (p20) and ceiling (p80) offsets
are the empirical 20th/80th percentile of `(actual - predicted)` from Ridge's
walk-forward residuals, computed separately per explosiveness tercile
(Low/Medium/High, within position).

**Out-of-sample check** — fit the bands on 2023-2024 residuals only, then
check how often 2025's actual points landed inside `[predicted + floor_offset,
predicted + ceiling_offset]`:

| | Coverage |
|---|---:|
| In-sample (2023-2024, the fitting data) | 0.599 |
| **Out-of-sample (2025, held out)** | **0.607** |
| Target (p80 - p20) | 0.60 |

The bands generalize — out-of-sample coverage lands within 0.7 points of the
nominal target, essentially the same as in-sample. This is the strongest
result in this analysis: the calibration isn't overfit to the seasons it was
built on.

**Production bands** (refit on all three seasons, 2023-2025, shipped as
`scripts/weekly_lineup.py`'s defaults and regeneratable via
`data/exports/weekly_variance_bands.csv`):

| Tercile | n | Floor offset | Ceiling offset | Band width |
|---|---:|---:|---:|---:|
| Low | 925 | -5.21 | +6.32 | 11.53 |
| Medium | 1,776 | -5.64 | +5.20 | 10.84 |
| High | 2,407 | -6.26 | +6.47 | 12.73 |

**Honest caveat:** the tercile effect on band *width* is real but modest
(High is ~1.9 points wider than Medium, not dramatically wider) — most of a
player's forecast uncertainty comes from the general difficulty of weekly
fantasy prediction, not from their individual explosiveness label. Use
`explosiveness_score` primarily as a **ranking/tie-breaker signal** between
two similarly-projected players (which one is more likely to boom, which is
the safer floor play), not as proof that high-explosiveness players carry a
categorically different range of outcomes.

---

## 3. The system: `scripts/weekly_lineup.py`

One command, run any week during the season:

```bash
# Auto-detects season + the next unplayed week
python scripts/weekly_lineup.py

# Force season/week
python scripts/weekly_lineup.py --season 2026 --week 5

# Head-to-head start/sit call
python scripts/weekly_lineup.py --compare "Bijan Robinson" "James Cook"
```

What it does:
1. Loads 2021→current-season weekly data (cached seasons read from disk,
   the live/current season pulled fresh from nflverse every run).
2. Trains Ridge on every available player-week.
3. Projects the next unplayed week for every player with current-season
   history, tagged `data_source=current_season`.
4. **Cold-start fallback**: for players with zero games so far this
   season (week 1, or a return from injury with no snaps yet), falls back to
   last season's per-game average as a rough prior, tagged
   `data_source=prior_season_only` — restricted to players who were
   top-150 PPR last season, so the board isn't full of irrelevant names.
   This is intentionally a rough prior, not a real projection: for a proper
   week-1/preseason board, use `scripts/run_2026_tier1_forecast.py`, which is
   built for that (market consensus + draft capital, no in-season history
   required).
5. Attaches `proj_points`, `floor`, `ceiling`, `explosiveness_score`, and
   `risk_tier` (Low/Medium/High, within position) to every row.
6. Prints the top 20 per position and exports the full board to
   `data/exports/{season}_wk{week:02d}_lineup.csv`.

`--compare` prints the two players side by side and calls out which one has
the higher floor (protect a lead) vs the higher ceiling (need upside to
catch up) — the two are often different players, which is the whole point of
carrying a variance metric alongside the point projection.

### Known limitations

- **Rookies and players with no prior-season data are not covered** by either
  the current-season or cold-start path — they need the preseason tiered
  model's draft-capital/market-consensus features instead.
- **QB/TE explosiveness scores are a much weaker signal** than RB/WR (see the
  persistence table above) — treat High/Low tags for those positions with
  more skepticism.
- **Tier band widths are modestly, not dramatically, differentiated** — see
  the honest caveat in section 2.
- The opponent-matchup feature (`opp_ppr_allowed_avg`) is retained in the
  feature set for XGBoost/Ridge to weigh as they see fit, but the standalone
  `OpponentAdjustedModel` is not used in production — see section 1.
