# Tiered feature ablation status

The nested walk-forward ablation is implemented in
`nfl_research.forecasting.evaluation`. It reports MAE, RMSE, within-position
Spearman correlation, top-24 precision, VORP-weighted MAE, CRPS when a model
provides a predictive distribution, and skill relative to market consensus.

## Results

| Model | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---:|---:|---:|---:|---:|
| Random walk | pending | pending | pending | pending | pending |
| Position mean | pending | pending | pending | pending | pending |
| Exponential smoothing | pending | pending | pending | pending | pending |
| Regression to mean | pending | pending | pending | pending | pending |
| Ridge | pending | pending | pending | pending | pending |
| XGBoost | pending | pending | pending | pending | pending |
| Hierarchical Bayes | n/a | pending | pending | pending | pending |
| Market consensus | n/a | pending | pending | pending | pending |

No higher-tier result is claimed yet. The repository does not contain the raw,
point-in-time parquet snapshots needed to run a defensible historical ablation,
and generating values from final-state nflverse tables would defeat the leak
guards this work adds. Populate `data/cache` with dated raw snapshots, build the
wide panel with an August 1 cutoff for each forecast season, run
`run_ablation()`, and replace this table with the generated metrics.

This is intentional: short Tier-3/Tier-4 panels make apparent gains especially
easy to manufacture through leakage or favorable evaluation windows. A
`pending` result is preferable to a fabricated or non-point-in-time result.
