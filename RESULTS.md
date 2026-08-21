# Tiered feature ablation results

The nested walk-forward ablation lives in `nfl_research.forecasting.evaluation`
(`run_ablation`, `walk_forward_evaluate`, `score_predictions`) and is driven end
to end by `scripts/run_tiered_ablation.py`, which loads season totals, builds
the tiered feature panel with an August 1 point-in-time cutoff for every
forecast season, and reports MAE, RMSE, within-position Spearman correlation,
top-24 precision, VORP-weighted MAE, CRPS (for models that emit a predictive
distribution), and skill relative to `MarketConsensusModel`.

Tiers 0 and 1 below are **real results** from real, cached nflverse/ffverse
data (`data/cache/*.parquet`), not synthetic fixtures. Tiers 2-4 are built,
point-in-time-guarded, and unit-tested, but have not yet been run end to end
against real historical data in this pass — see "What's not done yet" below.

## Two real bugs found and fixed while producing these numbers

1. **`load_ff_opportunity`'s `season` column round-trips through parquet as a
   string**, not an int. Every `as_of` cutoff comparison against it
   (`opp_raw["season"] <= cutoff_season`) raised `TypeError` before this fix.
   Now coerced with `pd.to_numeric` in `tier1.py`.
2. **`MarketConsensusModel`'s FantasyPros join was ranking a polluted pool.**
   FantasyPros reuses the same `ecr_type` code (`'ro'`) across three different,
   incompatible ranking universes: `redraft-overall` (the standard 1-QB board
   this league needs), `redraft-idp` (individual defensive players), and
   `redraft-offense`. Filtering on `ecr_type` alone silently mixed all three
   before computing rank, which is why the first version of this ablation
   showed Tom Brady's 2020 preseason rank as 677 and every model — including
   plain random-walk — "beating the market." Fixed to filter on
   `page_type == "redraft-overall"` (verified against real preseason boards:
   McCaffrey #1 overall 2021, Jonathan Taylor #1 2022, Puka Nacua #1 2025 —
   all correct). This also revealed that `load_ff_rankings(type="all")`'s
   `redraft-overall` history only goes back to 2021, not 2019 as originally
   assumed; the registry's declared first season for `ecr_rank`/`ecr_pos_rank`/
   `ecr_projection` has been corrected to 2021.

Both are covered by regression tests (`test_tiered_features.py`).

## Tier 0 — full panel (1999-2025, 23 walk-forward eval seasons 2003-2025)

| Model | MAE | RMSE | Spearman (within pos) | Top-24 precision | VORP-wtd MAE | CRPS |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 43.87 | 59.03 | 0.710 | 0.677 | 83.95 | - |
| XGBoost | 43.97 | 59.49 | 0.708 | 0.669 | 82.99 | - |
| Exp. smoothing | 45.08 | 63.21 | 0.695 | 0.662 | 72.00 | - |
| Hierarchical Bayes | 45.81 | 61.50 | 0.684 | 0.650 | 90.47 | 33.6 |
| Regression to mean | 45.99 | 61.74 | 0.681 | 0.646 | 91.26 | - |
| Random walk | 46.99 | 67.09 | 0.681 | 0.646 | 74.59 | - |
| Position mean | 70.38 | 86.59 | - | 0.265 | 167.78 | - |

At tier 0, Ridge and XGBoost are statistically indistinguishable (43.87 vs
43.97 MAE) — exactly the outcome the original build prompt predicted:
gradient boosting has no information advantage when every feature is a
transform of the same underlying PPR history.

## Tier 1 — market consensus + opportunity/usage features

Restricting to seasons where every tier-1 column is actually available
(`ecr_rank`'s true coverage starts 2021, not the originally-assumed 2019)
leaves only **2 walk-forward eval seasons: 2024 and 2025** — a genuinely
small sample, consistent with the build prompt's own warning that richer
sources come with shorter usable panels. Treat this table as directional,
not conclusive.

| Model | MAE | RMSE | Spearman (within pos) | Top-24 precision | VORP-wtd MAE | CRPS |
|---|---:|---:|---:|---:|---:|---:|
| **Market consensus** | **35.66** | **50.90** | **0.819** | 0.708 | **65.93** | - |
| XGBoost | 35.90 | 51.21 | 0.805 | **0.734** | 76.51 | - |
| Ridge | 40.21 | 56.27 | 0.755 | 0.708 | 88.47 | - |
| Ridge (tier-0 features, same restricted panel) | 40.29 | 57.11 | 0.759 | 0.698 | 91.58 | - |
| Hierarchical Bayes | 42.65 | 60.12 | 0.733 | 0.661 | 99.43 | 32.1 |
| Regression to mean | 42.91 | 60.33 | 0.733 | 0.656 | 100.65 | - |
| Random walk | 44.34 | 65.44 | 0.733 | 0.651 | 80.72 | - |
| Exp. smoothing | 43.33 | 61.02 | 0.747 | 0.688 | 75.91 | - |
| Position mean | 71.29 | 88.57 | - | 0.219 | 182.41 | - |

### Separating the "more seasons" effect from the "more features" effect

The prompt asked for exactly this control: report tier 0 on the *same*
season-restricted panel used for tier 1, so a tier's apparent gain isn't
confounded with a shrinking, easier-to-predict sample.

- Ridge, full 1999-2025 panel (23 seasons): **43.87** MAE
- Ridge, tier-0 features only, restricted to 2024-2025: **40.29** MAE
  → **3.58 pts** of "improvement" is just the restricted panel (recent,
  smaller, arguably easier seasons) — nothing to do with new features.
- Ridge, tier-1 features, same restricted panel: **40.21** MAE
  → real incremental lift from tier-1 features for Ridge: **~0.08 MAE**,
  indistinguishable from noise at n=2 seasons.
- XGBoost, full panel: 43.97 → XGBoost, tier-1 features, restricted panel:
  **35.90** MAE. Applying Ridge's ~3.6-point era effect as a rough baseline,
  roughly **4.4 of those 8.1 points** are attributable to the tier-1 features
  themselves (expected points, points-over-expected, ECR rank, usage share) —
  not to the shorter panel.

**This is the headline finding**: once the feature matrix actually has
heterogeneous, non-lag information to interact (tier 1), XGBoost separates
from Ridge for the first time (35.90 vs 40.21 MAE) and closes most of the gap
to market consensus, even beating it outright on top-24 precision (0.734 vs
0.708) and in the 2025 eval season specifically (34.95 vs 35.83 MAE). Ridge,
by contrast, gets almost nothing out of tier 1 — it can't exploit the
opportunity/expected-points and usage-share features the way a tree model
can, which is consistent with those features carrying most of their signal
through interactions (e.g. usage share × efficiency, POE × position) rather
than as additive linear effects.

Market consensus still wins on MAE, RMSE, Spearman, and VORP-weighted MAE.
Nobody in this project "beats the market" outright — the corrected result
looks the way professional consensus rankings should: hard to beat, with
XGBoost the closest challenger. (Note: `ecr_projection`, the FantasyPros
point-projection column, is never populated — `load_ff_rankings(type="all")`
doesn't carry a `projected_points`/`fantasy_points`/`projection` column in the
cached pull — so `MarketConsensusModel` runs entirely on its log(rank)
regression fallback, not the direct-projection shortcut. It still wins.)

## Model internals: features and weights (production fit on full data)

Everything above is out-of-sample walk-forward evaluation. This section is
different: each model below is fit **once, on every available real
player-season** (tier 0: 2000-2025, `n=11,157`; tier 1: 2021-2025 restricted
panel, `n=2,415`) to show what each model actually learned — the weights you'd
ship for a 2026 forecast, not a held-out score. Position codes: `QB=0, RB=1,
WR=2, TE=3`. Reproduced with `python scripts/extract_model_weights.py` (fits
`RidgeModel`, `XGBoostModel`, `HierarchicalBayesModel`, `MarketConsensusModel`
directly from `data/processed_tier1/tiered_forecasting_panel.parquet`).

### Ridge — standardized coefficients

Ridge runs on a `StandardScaler → Ridge(alpha=10)` pipeline with one-hot
position dummies (`drop='first'`, so `pos_code_0`/QB is the reference level),
so coefficients are directly comparable in magnitude.

**Tier 0** (top magnitude, of 22 features):

| Feature | Coef | Feature | Coef |
|---|---:|---|---:|
| `exp_smooth` | **+51.87** | `pos_code_1` (RB) | -3.68 |
| `age` | -40.19 | `games_lag2` | -3.28 |
| `age_squared` | +29.93 | `pos_code_3` (TE) | -2.81 |
| `draft_pick` | -13.31 | `points_ppr_lag1` | +2.60 |
| `ppg_lag1` | +9.78 | `pos_code_2` (WR) | -2.35 |
| `draft_round` | +4.78 | `points_ppr_lag2` | +1.96 |
| `trend_1` | +4.25 | `games_lag1` | -1.35 |
| `trend_2` | +3.96 | `ppg_lag2` | -1.31 |
| `is_undrafted` | +3.68 | `points_ppr_lag3` | +1.30 |

The exponentially-smoothed lag blend (`exp_smooth`, weights 0.5/0.3/0.2 on
lag1/2/3) dominates every other lag feature combined — Ridge mostly learns
"trust the smoothed history," with the individual lag columns fighting each
other for small residual signal (multicollinearity: they're all built from
the same three numbers). `age` and `age_squared` together trace the expected
concave career-arc curve (rises, peaks, declines) rather than a monotonic
effect, and draft capital (`draft_pick` negative — lower pick number, i.e.
earlier draft slot, pushes the forecast up) matters more than any single
year-over-year trend term.

**Tier 1** (top magnitude, of 33 features):

| Feature | Coef | Feature | Coef |
|---|---:|---|---:|
| `age` | -27.07 | `pos_code_1` (RB) | -2.65 |
| `exp_fantasy_pts` | **+26.14** | `trend_2` | -2.41 |
| `exp_smooth` | +25.93 | `games_lag2` | -2.33 |
| `age_squared` | +17.12 | `poe_ppg` | +2.21 |
| `exp_fantasy_ppg` | +13.10 | `ppg_lag1` | -2.03 |
| `draft_pick` | -12.10 | `ecr_pos_rank` | +1.35 |
| `ecr_rank` | **-10.54** | `carry_share_lag1` | -1.19 |
| `points_over_expected` | +4.79 | `ppg_lag2` | +0.84 |
| `trend_1` | -4.06 | `points_ppr_lag1` | -0.60 |
| `pos_code_2` (WR) | -3.96 | *(19 more, all \|coef\| < 0.6)* | |

Once the opportunity-model features are available, `exp_fantasy_pts`
(expected PPR points from ffverse's opportunity model) becomes roughly as
important as `exp_smooth`, and the raw lag columns (`points_ppr_lag1`,
`ppg_lag1`) collapse toward zero or flip sign — Ridge is substituting
"expected production given usage" for "what actually happened last year"
wherever it can. `ecr_rank` (market consensus rank) is the third-largest
single feature and carries a clean negative sign (worse/higher rank number ⇒
lower forecast), which is exactly the direction a legitimate market signal
should point. `ecr_projection` gets a coefficient of exactly 0.0 because, as
noted above, that column is never populated in the cached data.

### XGBoost — feature importances (gain-normalized)

**Tier 0**: `exp_smooth` 0.471, `ppg_lag1` 0.140, `points_ppr_lag1` 0.116,
`lag3_missing` 0.045, `games_lag1` 0.024, `age` 0.022, `age_squared` 0.021,
`draft_pick` 0.020, `draft_round` 0.019 — the remaining 13 features split
0.061 between them. XGBoost concentrates almost all of its splits on three
correlated summaries of recent history (`exp_smooth`, `ppg_lag1`,
`points_ppr_lag1`); age and draft capital get used but only for the tails
the smoothed history can't already explain.

**Tier 1**: `exp_smooth` 0.307, `exp_fantasy_pts` 0.190, `ecr_pos_rank`
0.064, `points_ppr_lag1` 0.058, `ppg_lag2` 0.050, `ecr_rank` 0.046,
`exp_fantasy_ppg` 0.041, `target_share_lag1` 0.033, `ppg_lag1` 0.020,
`carry_share_lag1` 0.018 — the top 10 of 32 features already account for
0.826 of total gain. This is the mechanism behind the headline finding above:
XGBoost puts real weight (0.19) directly on `exp_fantasy_pts` and pulls
`ecr_pos_rank`/`ecr_rank`/`target_share_lag1`/`carry_share_lag1` into the top
10 — i.e. it's building interactions between usage/opportunity and market
rank that Ridge's linear form structurally can't express, which is exactly
why XGBoost separates from Ridge only once tier-1 features exist.

### Hierarchical Bayes — position-level parameters (tier 1, MAP fit)

Model: `points ~ Normal(alpha[pos] + beta[pos] * (lag1 - mean(lag1)) + gamma * ppg_lag1, sigma)`

| Position | alpha (intercept) | beta (lag1 slope) |
|---|---:|---:|
| QB | 64.64 | 0.568 |
| RB | 63.44 | 0.555 |
| WR | 61.02 | 0.585 |
| TE | 61.57 | 0.595 |

`gamma` (shared `ppg_lag1` slope) = **2.863**; `sigma` (residual sd, the same
number reported as the model's CRPS predictive spread) = **59.39**. Unlike
Ridge/XGBoost, the four positions end up with nearly identical partial-pooling
estimates (alpha within a 3.6-point band, beta within 0.04) — the model is
telling us position doesn't add much once lag1 and ppg_lag1 are in, which is
consistent with Hierarchical Bayes scoring closest to Regression-to-mean
rather than to XGBoost/Ridge in the tier-0/tier-1 tables above.

### Market consensus — per-position log(rank) regression

`predicted = intercept + slope * log1p(ecr_rank)`, fit per position (falls
back to the global fit below 20 players at a position):

| Position | Intercept | Slope |
|---|---:|---:|
| Global | 432.73 | -63.17 |
| QB | 825.64 | **-126.00** |
| RB | 401.37 | -62.03 |
| WR | 422.93 | -61.57 |
| TE | 425.20 | -61.61 |

QB's slope is roughly double every other position's — the QB1-vs-QB12
spread in preseason consensus rank maps to a much steeper points gradient
than the equivalent spread at RB/WR/TE, which matches the real shape of
one-QB-league fantasy scoring (a small number of streamable, closely-bunched
non-elite QBs vs. a long, steep drop-off at the other skill positions).

## What's not done yet

- **Tiers 2-4** are implemented (`tier2.py`/`tier3.py`/`tier4.py`), point-in-time
  guarded, and unit-tested, but have not been run against real historical data
  in this pass — each tier-0/tier-1 real run took ~10 minutes, dominated by
  refitting `HierarchicalBayesModel` via ADVI once per walk-forward season.
  Run them with:
  ```
  python scripts/run_tiered_ablation.py --max-tier 2   # add --no-pbp to skip the large play-by-play pull
  python scripts/run_tiered_ablation.py --max-tier 3
  python scripts/run_tiered_ablation.py --max-tier 4
  ```
  Per the build prompt's own expectation, tiers 3-4 have ~10 and ~4 seasons of
  coverage respectively and are likely to show gains indistinguishable from
  noise — that would be a legitimate finding, not a reason to tune until they
  look better.
- **In-fold hyperparameter tuning** (Part 5.5 — Ridge `alpha`, XGBoost
  `max_depth`/`learning_rate`/`reg_lambda` tuned via nested walk-forward
  `GridSearchCV` inside each training fold) is implemented
  (`evaluation.tune_hyperparameters`, `walk_forward_evaluate(..., tune=True)`,
  `--tune` CLI flag) but was not used for the numbers above, which use each
  model's hardcoded constructor defaults. Worth re-running tier 1 with
  `--tune` once tiers 2-4 are in, since a wider matrix is exactly when
  hardcoded hyperparameters stop being a safe default.
- **PBP-derived tier-2 features** (air yards share, WOPR, red-zone shares,
  team pass-rate-over-expected) require `load_pbp(seasons=True)`, which was
  not cached in this environment (the 1999-2025 play-by-play pull is large);
  `source_cache.load_tier_sources(..., include_pbp=False)` / `--no-pbp` skips
  it. The rest of tier 2 (snap counts, depth charts, schedules, competition
  churn) does not depend on it.

## Reproducing

```
python scripts/run_tiered_ablation.py --max-tier 1 --no-pbp --out-dir data/processed_tier1
```

Writes `tiered_forecasting_panel.parquet`, `tiered_ablation_predictions.csv`,
`tiered_ablation_scores.csv`, and one `tiered_ablation_<metric>.csv` per
metric family to `--out-dir`. Raw nflverse/ffverse pulls are cached to
`data/cache/*.parquet` on first fetch (`source_cache.py`), so subsequent runs
are network-free and reproducible offline, per Part 3's requirements.
