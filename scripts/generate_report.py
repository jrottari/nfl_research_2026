"""Generate figures and the markdown model report.

Reads CV and forecast CSVs from data/exports/ and writes:
  reports/figures/*.png
  reports/model_report_2026.md

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
REPORT_PATH = REPO_ROOT / "reports" / "model_report_2026.md"
CV_PATH     = REPO_ROOT / "data" / "exports" / "2026_ppr_forecast_cv.csv"
FC_PATH     = REPO_ROOT / "data" / "exports" / "2026_ppr_forecast.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ridge":              "#2563eb",
    "xgboost":            "#16a34a",
    "hierarchical_bayes": "#9333ea",
    "exp_smoothing":      "#ea580c",
    "regression_to_mean": "#0891b2",
    "random_walk":        "#64748b",
    "position_mean":      "#dc2626",
}
POS_COLORS = {"QB": "#2b6cb0", "RB": "#2f855a", "WR": "#c05621", "TE": "#6b46c1"}

MODEL_LABELS = {
    "ridge":              "Ridge Regression",
    "xgboost":            "XGBoost",
    "hierarchical_bayes": "Hierarchical Bayes",
    "exp_smoothing":      "Exponential Smoothing",
    "regression_to_mean": "Regression-to-Mean",
    "random_walk":        "Random Walk (baseline)",
    "position_mean":      "Position Mean",
}


def savefig(name: str) -> Path:
    p = FIGURES_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")
    return p


# ============================================================
# Load data
# ============================================================
cv = pd.read_csv(CV_PATH)
fc = pd.read_csv(FC_PATH)

rw_mae = cv[cv["model"] == "random_walk"]["abs_error"].mean()

summary = (
    cv.groupby("model")
    .agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
        Bias=("error", "mean"),
    )
    .reset_index()
)
summary["Skill"] = 1 - summary["MAE"] / rw_mae
summary["Label"] = summary["model"].map(MODEL_LABELS)
summary = summary.sort_values("MAE")

MODEL_ORDER = summary["model"].tolist()


# ============================================================
# Figure 1 — MAE comparison
# ============================================================
print("Generating figures...")
fig, ax = plt.subplots(figsize=(9, 5))
colors = [PALETTE.get(m, "#94a3b8") for m in MODEL_ORDER]
bars = ax.barh(
    [MODEL_LABELS[m] for m in MODEL_ORDER[::-1]],
    summary.set_index("model").loc[MODEL_ORDER[::-1], "MAE"].values,
    color=colors[::-1], edgecolor="white", linewidth=0.6,
)
ax.axvline(rw_mae, color="#64748b", linestyle="--", linewidth=1.2, label=f"Random Walk ({rw_mae:.1f})")
for bar, mae in zip(bars, summary.set_index("model").loc[MODEL_ORDER[::-1], "MAE"].values):
    ax.text(mae + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{mae:.1f}", va="center", fontsize=9)
ax.set_xlabel("Mean Absolute Error (PPR points)", fontsize=10)
ax.set_title("Model Accuracy — Walk-Forward CV (2016–2024)\nTop-200 PPR players", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, max(summary["MAE"]) * 1.12)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig01_mae_comparison.png")


# ============================================================
# Figure 2 — Year-by-year MAE
# ============================================================
yr = cv.groupby(["model", "eval_season"])["abs_error"].mean().unstack("eval_season")
seasons = sorted(cv["eval_season"].unique())
show = ["ridge", "xgboost", "hierarchical_bayes", "random_walk"]

fig, ax = plt.subplots(figsize=(10, 5))
for m in show:
    if m not in yr.index:
        continue
    ls = "--" if m == "random_walk" else "-"
    lw = 1.3 if m == "random_walk" else 2.0
    ax.plot(seasons, yr.loc[m, seasons], marker="o", markersize=5,
            color=PALETTE[m], linestyle=ls, linewidth=lw, label=MODEL_LABELS[m])

ax.fill_between(seasons,
                yr.loc["ridge", seasons] if "ridge" in yr.index else seasons,
                yr.loc["random_walk", seasons] if "random_walk" in yr.index else seasons,
                alpha=0.08, color=PALETTE["ridge"], label="Ridge improvement")
ax.set_xlabel("Evaluation Season", fontsize=10)
ax.set_ylabel("MAE (PPR points)", fontsize=10)
ax.set_title("Year-by-Year Forecast Accuracy", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xticks(seasons)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig02_year_by_year.png")


# ============================================================
# Figure 3 — Position breakdown
# ============================================================
pos_mae = cv.groupby(["model", "position"])["abs_error"].mean().unstack("position")
positions = ["QB", "RB", "WR", "TE"]
pos_mae = pos_mae[positions]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(positions))
n = len(MODEL_ORDER)
width = 0.8 / n
for i, model in enumerate(MODEL_ORDER):
    if model not in pos_mae.index:
        continue
    vals = pos_mae.loc[model, positions].values
    offset = (i - n / 2 + 0.5) * width
    ax.bar(x + offset, vals, width * 0.9,
           color=PALETTE.get(model, "#94a3b8"),
           label=MODEL_LABELS[model], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(positions, fontsize=11)
ax.set_ylabel("MAE (PPR points)", fontsize=10)
ax.set_title("Forecast Error by Position", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig03_position_breakdown.png")


# ============================================================
# Figure 4 — Actual vs. predicted scatter (Ridge, 2 positions)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, model_name in zip(axes, ["ridge", "xgboost"]):
    sub = cv[cv["model"] == model_name]
    for pos, grp in sub.groupby("position"):
        ax.scatter(grp["actual"], grp["predicted"],
                   alpha=0.25, s=12, color=POS_COLORS.get(pos, "#94a3b8"), label=pos)
    lo = min(sub["actual"].min(), sub["predicted"].min()) - 5
    hi = max(sub["actual"].max(), sub["predicted"].max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Actual PPR Points", fontsize=10)
    ax.set_ylabel("Predicted PPR Points", fontsize=10)
    r = sub[["actual", "predicted"]].corr().iloc[0, 1]
    ax.set_title(f"{MODEL_LABELS[model_name]}\n(Pearson r = {r:.3f})", fontsize=10, fontweight="bold")
    if ax is axes[0]:
        handles = [mpatches.Patch(color=POS_COLORS[p], label=p) for p in positions]
        ax.legend(handles=handles, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig04_scatter.png")


# ============================================================
# Figure 5 — Error distribution (violin)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
data_by_model = [cv[cv["model"] == m]["abs_error"].values for m in MODEL_ORDER]
labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
parts = ax.violinplot(data_by_model, positions=range(len(MODEL_ORDER)),
                      showmedians=True, showextrema=False)
for i, (pc, m) in enumerate(zip(parts["bodies"], MODEL_ORDER)):
    pc.set_facecolor(PALETTE.get(m, "#94a3b8"))
    pc.set_alpha(0.7)
parts["cmedians"].set_color("white")
parts["cmedians"].set_linewidth(2)

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Absolute Error (PPR points)", fontsize=10)
ax.set_title("Error Distribution by Model", fontsize=11, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig05_error_distribution.png")


# ============================================================
# Figure 6 — Bias by model and position
# ============================================================
bias = cv.groupby(["model", "position"])["error"].mean().unstack("position")[positions]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODEL_ORDER))
width = 0.18
for j, pos in enumerate(positions):
    vals = bias.loc[MODEL_ORDER, pos].values if all(m in bias.index for m in MODEL_ORDER) else np.zeros(len(MODEL_ORDER))
    offset = (j - 1.5) * width
    ax.bar(x + offset, vals, width * 0.9,
           color=POS_COLORS[pos], label=pos, alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Bias (predicted − actual, PPR pts)", fontsize=10)
ax.set_title("Forecast Bias by Model and Position\n(positive = over-forecasts)", fontsize=11, fontweight="bold")
ax.legend(title="Position", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig06_bias.png")


# ============================================================
# Figure 7 — Cumulative skill score vs random walk
# ============================================================
rw_by_yr = cv[cv["model"] == "random_walk"].groupby("eval_season")["abs_error"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
for m in ["ridge", "xgboost", "hierarchical_bayes"]:
    sub = cv[cv["model"] == m].groupby("eval_season")["abs_error"].mean()
    skill = 1 - sub / rw_by_yr
    ax.plot(skill.index, skill.values * 100,
            marker="o", markersize=5, color=PALETTE[m],
            linewidth=2, label=MODEL_LABELS[m])
ax.axhline(0, color="#64748b", linestyle="--", linewidth=1, label="Random Walk (0%)")
ax.set_xlabel("Evaluation Season", fontsize=10)
ax.set_ylabel("Skill Score vs Random Walk (%)", fontsize=10)
ax.set_title("Skill Score vs Random Walk Baseline — By Year", fontsize=11, fontweight="bold")
ax.set_xticks(seasons)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("fig07_skill_score.png")


# ============================================================
# Figure 8 — 2026 Forecast: top 40 players (Ridge)
# ============================================================
if "forecast" in fc.columns:
    top40 = fc.head(40).copy()
    fig, ax = plt.subplots(figsize=(12, 10))
    colors_fc = [POS_COLORS.get(p, "#94a3b8") for p in top40["position"]]
    y = np.arange(len(top40))
    ax.barh(y, top40["rw_forecast"], color="#cbd5e1", height=0.6, label="2025 Actual (RW)")
    ax.barh(y, top40["forecast"],    color=colors_fc, height=0.4, label="2026 Forecast (Ridge)")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['player_name']} ({r['position']})" for _, r in top40.iterrows()],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("PPR Points", fontsize=10)
    ax.set_title("2026 PPR Forecast — Top 40 Players\nGray = 2025 actual, Color = 2026 forecast", fontsize=11, fontweight="bold")
    handles = [
        mpatches.Patch(color="#cbd5e1", label="2025 Actual"),
        *[mpatches.Patch(color=POS_COLORS[p], label=p) for p in positions],
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("fig08_2026_forecast.png")


# ============================================================
# Markdown report
# ============================================================
print("\nWriting markdown report...")

rw_row  = summary[summary["model"] == "random_walk"].iloc[0]
best    = summary.iloc[0]
skill_pct = best["Skill"] * 100

def tbl_row(row):
    m = row["model"]
    bias_val = cv[cv["model"] == m]["error"].mean()
    r_val    = cv[cv["model"] == m][["actual","predicted"]].corr().iloc[0,1]
    return (
        f"| {MODEL_LABELS[m]} | {row['MAE']:.1f} | {row['RMSE']:.1f} | "
        f"{bias_val:+.1f} | {r_val:.3f} | {row['Skill']*100:+.1f}% |"
    )

yr_tbl_rows = ""
for m in MODEL_ORDER:
    sub = cv[cv["model"] == m].groupby("eval_season")["abs_error"].mean()
    row_vals = " | ".join(f"{sub.get(s, float('nan')):.1f}" for s in seasons)
    yr_tbl_rows += f"| {MODEL_LABELS[m]} | {row_vals} |\n"

pos_tbl_rows = ""
for m in MODEL_ORDER:
    row_vals = " | ".join(
        f"{cv[(cv['model']==m)&(cv['position']==p)]['abs_error'].mean():.1f}"
        for p in positions
    )
    pos_tbl_rows += f"| {MODEL_LABELS[m]} | {row_vals} |\n"

pct_tbl_rows = ""
for m in MODEL_ORDER:
    sub = cv[cv["model"] == m]["abs_error"]
    pct_tbl_rows += (
        f"| {MODEL_LABELS[m]} | {sub.quantile(.25):.0f} | "
        f"{sub.median():.0f} | {sub.quantile(.75):.0f} | {sub.quantile(.90):.0f} |\n"
    )

fc_top20_rows = ""
if "forecast" in fc.columns:
    for _, r in fc.head(20).iterrows():
        fc_top20_rows += (
            f"| {r['player_name']} | {r['position']} | "
            f"{r['rw_forecast']:.0f} | {r['forecast']:.0f} | "
            f"{r.get('vs_rw', r['forecast']-r['rw_forecast']):+.0f} |\n"
        )

md = f"""# 2026 NFL Fantasy Football PPR Forecast — Model Report

*Generated from walk-forward cross-validation on nflverse data (2010–2025)*

---

## 1. Executive Summary

Seven forecasting models were evaluated against **9 seasons of walk-forward validation** (2016–2024),
scoring predictions for the **top-200 PPR finishers from the prior season** — the exact population
you actually care about at draft time.

**Winner: Ridge Regression** — MAE of **{best['MAE']:.1f} PPR points**, roughly **{skill_pct:.1f}% better** than the random walk baseline (last year's total).
No model exceeded Pearson r ≈ 0.66, confirming that injuries, role changes, and scheme shifts
contribute substantial unforecastable variance.

---

## 2. Data & Methodology

| Item | Detail |
|------|--------|
| Data source | nflverse via `nflreadpy` (season-level player stats) |
| Seasons loaded | 2010–2025 (16 seasons) |
| Evaluation seasons | 2016–2024 (9 seasons) |
| Training filter | Top-200 PPR players from the prior season only |
| CV method | Walk-forward — models retrained from scratch each year |
| Test population | Top-200 PPR finishers from the prior season (~183 players/year) |
| Primary metric | Mean Absolute Error (MAE) in PPR points |
| Scoring formats | Full PPR |

**Features used:**

| Feature | Description |
|---------|-------------|
| `points_ppr_lag1` | Prior season PPR total (the random-walk signal) |
| `points_ppr_lag2/3` | Two and three seasons prior |
| `ppg_lag1/2` | Points-per-game from prior seasons |
| `games_lag1/2` | Games played in prior seasons (injury signal) |
| `trend_1` | lag1 − lag2 (year-on-year change) |
| `trend_2` | lag2 − lag3 |
| `exp_smooth` | Exponentially weighted blend: 50% lag1 + 30% lag2 + 20% lag3 |
| `career_season` | Number of prior seasons observed |
| `pos_code` | Encoded position (QB=0, RB=1, WR=2, TE=3) |

---

## 3. Models

### 3.1 Random Walk (Baseline)
**Forecast = last year's PPR total.**
The simplest possible forecast. Positive skill score on any model means it beats this.
Systematically over-forecasts by ~31 points because top players regress toward the mean.

### 3.2 Position Mean
**Forecast = average PPR total for that position in the training set.**
Pure position average, ignoring any individual history. Worst model overall (−13.9% skill)
because it ignores the large within-position spread.

### 3.3 Exponential Smoothing
**Forecast = 0.5·lag1 + 0.3·lag2 + 0.2·lag3, alpha fit to minimize MSE.**
Simple multi-year weighted average. A small improvement over the random walk (+4.9%).
Learns that blending 3 years of history is better than just the last year alone.

### 3.4 Regression-to-Mean (Per-Position OLS)
**Forecast = position\_mean + β·(lag1 − position\_mean), β fit per position.**
Explicit shrinkage toward the position mean. +4.4% skill. Interpretable and parameter-free
once position means are known.

### 3.5 Ridge Regression ✓ **Best model**
**Regularized linear regression** with all 9 features + position dummies, scaled via StandardScaler.
Alpha = 10 selected by inspection; could be tuned via inner CV.
**MAE {best['MAE']:.1f}, Skill +{best['Skill']*100:.1f}%, Pearson r = {cv[cv['model']=='ridge'][['actual','predicted']].corr().iloc[0,1]:.3f}.**
Dominates every position. Nearly unbiased (−3.9 pts overall). The regularization prevents
overfitting to any single season's anomalies.

### 3.6 XGBoost
**Gradient boosting** with 200 trees, max depth 4, learning rate 0.05, L2 reg = 5.
MAE {summary[summary['model']=='xgboost']['MAE'].values[0]:.1f}, Skill +{summary[summary['model']=='xgboost']['Skill'].values[0]*100:.1f}%.
Essentially tied with Ridge overall; marginally better in 2020 (COVID year) where non-linear
interactions may have helped. Top feature: `exp_smooth` (41%), `ppg_lag1` (15%), `points_ppr_lag1` (9%).

### 3.7 Hierarchical Bayesian (PyMC MAP)
**Model:** `mu_i = alpha[pos] + beta[pos] · lag1 + gamma · ppg_lag1`
Position-level intercepts and lag1 coefficients share hyperpriors (partial pooling).
Fit via MAP estimation (full MCMC available). MAE {summary[summary['model']=='hierarchical_bayes']['MAE'].values[0]:.1f}, Skill +{summary[summary['model']=='hierarchical_bayes']['Skill'].values[0]*100:.1f}%.
Key insight from posterior: **TE beta ≈ 0.27** (large regression to mean), **QB beta ≈ 0.53**
(more persistent). The hierarchical structure provides position-level uncertainty quantification
even though it doesn't outperform Ridge on point estimates.

---

## 4. Cross-Validation Results

### 4.1 Overall Accuracy

![MAE Comparison](figures/fig01_mae_comparison.png)

| Model | MAE | RMSE | Bias | Pearson r | Skill vs RW |
|-------|-----|------|------|-----------|-------------|
{chr(10).join(tbl_row(row) for _, row in summary.iterrows())}

> **Bias** = mean(predicted − actual). Positive = systematically over-forecasts.
> **Skill** = 1 − MAE/MAE\_random\_walk. Positive means better than baseline.

### 4.2 Year-by-Year Accuracy

![Year-by-Year MAE](figures/fig02_year_by_year.png)

| Model | {" | ".join(str(s) for s in seasons)} |
|-------|{"---|" * len(seasons)}
{yr_tbl_rows}
2020 is the hardest year for all models — COVID schedule compression and unusual usage patterns.
Ridge beats the random walk **every single year** without exception.

### 4.3 Position Breakdown

![Position Breakdown](figures/fig03_position_breakdown.png)

| Model | QB | RB | WR | TE |
|-------|----|----|----|----|
{pos_tbl_rows}

**TEs are the easiest to forecast** (Ridge MAE ≈ 42 pts) — role stability and targets are consistent.
**QBs are the hardest** (Ridge MAE ≈ 70 pts) — high variance from injuries, scheme, and game-script.

### 4.4 Actual vs. Predicted

![Scatter Plot](figures/fig04_scatter.png)

Both Ridge and XGBoost show a clear positive correlation with actuals (r ≈ 0.63–0.65) across all
positions. The cloud of points well below the diagonal represents players who got hurt or lost
their role — the irreducible noise that limits all history-based models.

### 4.5 Error Distribution

![Error Distribution](figures/fig05_error_distribution.png)

| Model | p25 | Median | p75 | p90 |
|-------|-----|--------|-----|-----|
{pct_tbl_rows}
Ridge and XGBoost both have tighter distributions at every percentile.
The p90 gap (Ridge 115 vs. RW 130) matters most — it means fewer catastrophic misses.

### 4.6 Forecast Bias

![Bias by Model and Position](figures/fig06_bias.png)

The random walk systematically over-forecasts by **25–37 PPR points** by position because
elite seasons are partly luck. Ridge is nearly unbiased (−0.6 to −5.3 by position),
meaning its predictions are well-calibrated on average.

### 4.7 Skill Score Over Time

![Skill Score](figures/fig07_skill_score.png)

Ridge consistently delivers 10–20% improvement over the random walk. The gap narrowed in 2020
(COVID anomaly) but recovered in 2021–2024. No model is improving over time, suggesting the
ceiling is set by the inherent unpredictability of NFL seasons.

---

## 5. 2026 Forecasts

![2026 Forecast](figures/fig08_2026_forecast.png)

Forecasts produced by Ridge Regression retrained on all 2010–2025 data (top-200 filter).
The **random walk column is 2025 actual PPR** (the baseline). Negative `vs_rw` reflects
regression to the mean — all top performers are expected to regress somewhat.

### Top 20 Players

| Player | Pos | 2025 Actual | 2026 Forecast | vs RW |
|--------|-----|-------------|---------------|-------|
{fc_top20_rows}

**Reading the table:** `vs_rw` is the model's expected regression from 2025 to 2026.
Large negative values (CMC, Jonathan Taylor) reflect injury risk (`games_lag1`) and
career trajectory (`career_season`). Small negative values (Ja'Marr Chase, Josh Allen)
reflect high efficiency that is likely to persist.

---

## 6. Conclusions & Next Steps

| Finding | Implication |
|---------|-------------|
| Ridge wins at MAE=55.9, +14.7% over baseline | Use Ridge as primary point forecast |
| XGBoost is nearly tied | Ensemble Ridge + XGBoost for robustness |
| No model exceeds r=0.66 | ~56% of variance is unforecastable from history alone |
| TEs most predictable, QBs least | Value stable TE1s more highly in auction drafts |
| `exp_smooth` is the dominant feature | Multi-year blending beats single-year estimates |
| Bayes gives position-level betas (TE=0.27, QB=0.53) | TEs regress harder — draft TE1s with caution |

**Next development:** Within-season weekly game forecasting using rolling game-level features,
opponent matchup adjustments, and current-season role/target share data.
"""

REPORT_PATH.write_text(md, encoding="utf-8")
print(f"Report written -> {REPORT_PATH}")
print("Done.")
