"""Fit tier-0/tier-1 models on the full real panel and print feature weights.

Not part of the walk-forward evaluation harness (`run_tiered_ablation.py`) --
this is a single production fit on every available season, purely to inspect
what each model learned (Ridge coefficients, XGBoost feature importances,
Hierarchical Bayes position params, MarketConsensus per-position rank fit).
See RESULTS.md's "Model internals: features and weights" section.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from nfl_research.forecasting.models import (
    HierarchicalBayesModel,
    MarketConsensusModel,
    RidgeModel,
    XGBoostModel,
)


def main() -> int:
    panel_path = REPO_ROOT / "data" / "processed_tier1" / "tiered_forecasting_panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel["season"] = pd.to_numeric(panel["season"])

    print("=" * 70)
    print(f"TIER 0 -- full panel, seasons {panel['season'].min()}-{panel['season'].max()}")
    print("=" * 70)

    tier0 = panel.dropna(subset=["target"])
    x0, y0 = tier0.drop(columns=["target"]), tier0["target"]

    ridge0 = RidgeModel(alpha=10.0, max_tier=0).fit(x0, y0)
    coef0 = ridge0._pipeline.named_steps["ridge"].coef_
    names0 = ridge0._encode(x0).columns.tolist()
    print("\n-- Ridge (tier 0) standardized coefficients --")
    print(pd.Series(coef0, index=names0).sort_values(key=np.abs, ascending=False).to_string())

    xgb0 = XGBoostModel(max_tier=0).fit(x0, y0)
    print("\n-- XGBoost (tier 0) feature importances --")
    print(xgb0.feature_importance().to_string())

    print("\n" + "=" * 70)
    restricted = panel[panel["season"] >= 2021].dropna(subset=["target"])
    print(
        f"TIER 1 -- restricted panel, seasons {restricted['season'].min()}-"
        f"{restricted['season'].max()} ({len(restricted)} rows)"
    )
    print("=" * 70)

    x1, y1 = restricted.drop(columns=["target"]), restricted["target"]

    ridge1 = RidgeModel(alpha=10.0, max_tier=1).fit(x1, y1)
    coef1 = ridge1._pipeline.named_steps["ridge"].coef_
    names1 = ridge1._encode(x1).columns.tolist()
    print("\n-- Ridge (tier 1) standardized coefficients --")
    print(pd.Series(coef1, index=names1).sort_values(key=np.abs, ascending=False).to_string())

    xgb1 = XGBoostModel(max_tier=1).fit(x1, y1)
    print("\n-- XGBoost (tier 1) feature importances --")
    print(xgb1.feature_importance().to_string())

    print("\n-- MarketConsensusModel (tier 1) per-position log(rank) fit --")
    mc = MarketConsensusModel().fit(x1, y1)
    print("global (intercept, slope):", mc._global_model)
    for code, params in sorted(mc._pos_models.items()):
        print(f"  pos_code={code}: intercept={params[0]:.2f}, slope={params[1]:.2f}")

    print("\n-- HierarchicalBayesModel (tier 1, MAP fit for speed) --")
    hb = HierarchicalBayesModel(use_map=True, max_tier=1).fit(x1, y1)
    print(hb.position_params().to_string(index=False))
    print("gamma (ppg1 coefficient):", hb._map.get("gamma"))
    print("sigma (residual sd):", hb._map.get("sigma"))

    pos_lookup = dict(zip(restricted["pos_code"], restricted["position"], strict=False))
    print("\npos_code -> position:", pos_lookup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
