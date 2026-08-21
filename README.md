# nfl_research_2026

Season-end NFL fantasy football research built on [nflreadpy](https://nflreadpy.nflverse.com/)
(the nflverse successor to the retired `nfl_data_py`).

Produces a top-250 PPR board with per-game scoring, efficiency, weekly consistency, value over
replacement, and tiers — exported as CSVs that drop straight into Google Sheets Tables.

## Install

```bash
# uv (recommended — nflverse's own tooling choice)
uv venv && uv pip install -e ".[notebook,dev]"

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"
```

### Requirements

| Package | Why |
|---|---|
| `nflreadpy>=0.1.5` | nflverse data loader |
| `polars>=1.0` | nflreadpy's native frame type |
| `pyarrow>=16.0` | **required** for `.to_pandas()` and parquet — easy to forget, fails at runtime |
| `pandas>=2.2` | analysis layer |
| `numpy>=2.0` | numerics |
| `matplotlib>=3.8` | notebook charts |
| `jupyterlab`, `ipykernel` | notebook (`[notebook]` extra) |
| `pytest`, `ruff`, `mypy` | dev (`[dev]` extra) |

`pip install "nflreadpy[pandas]"` pulls the pandas + pyarrow pair for you if you'd rather not
list them separately.

## Usage

```bash
# build + export with defaults (season 2025, top 250, -> data/exports)
python -m nfl_research

# pick a season, depth, and destination
python -m nfl_research --season 2025 --top-n 300 --out ~/Desktop/fantasy_2025

# build and inspect without writing files
python -m nfl_research --no-export

# only the big board, and cache the full frame as parquet
python -m nfl_research --only 2025_overall_top250_ppr.csv --parquet
```

Or open `notebooks/01_season_review_2025.ipynb` for the same pipeline with charts and tables.

### Weekly lineup helper (in-season start/sit)

```bash
# Auto-detects season + next unplayed week, exports a lineup board
python scripts/weekly_lineup.py

# Head-to-head start/sit call
python scripts/weekly_lineup.py --compare "Bijan Robinson" "James Cook"
```

Trains Ridge (the walk-forward CV winner — see `reports/weekly_forecast_report.md`)
on 2021→current-season weekly data and projects the next week for every rostered
player, with a floor/ceiling band and an `explosiveness_score` (0-100) so two
players with the same median projection can still be told apart by risk. Week-1 /
zero-game players fall back to last season's per-game average, tagged
`prior_season_only` — for a real preseason board use `scripts/run_2026_tier1_forecast.py`
instead. Re-run `scripts/analyze_weekly_variance.py` periodically to refresh the
floor/ceiling calibration in `data/exports/weekly_variance_bands.csv`.

## Structure

```
nfl_research_2026/
├── pyproject.toml            # deps, entry point, ruff/mypy/pytest config
├── requirements.txt          # same deps, for pip-only workflows
├── src/nfl_research/
│   ├── config.py             # Settings: season, depth, replacement ranks, tier gaps
│   ├── schema.py             # canonical column names — absorbs nflverse renames
│   ├── loaders.py            # nflreadpy wrappers; the ONLY Polars→pandas boundary
│   ├── metrics.py            # scoring formats, volume, efficiency, consistency
│   ├── rankings.py           # ranks, value over replacement, tiers, board layouts
│   ├── exports.py            # Sheets-ready CSV writers
│   ├── pipeline.py           # build_season(): raw data in, finished boards out
│   └── cli.py                # argparse entry point
├── notebooks/
│   └── 01_season_review_2025.ipynb
├── scripts/build_boards.py   # run the CLI without -m
├── tests/
│   ├── fake_nflverse.py      # synthetic data in the current nflverse schema
│   └── test_pipeline.py      # 22 offline tests
└── data/
    ├── exports/              # CSV output (gitignored)
    └── processed/            # parquet cache (gitignored)
```

The layering rule: `schema` → `loaders` → `metrics` → `rankings` → `pipeline` → `cli`/notebook.
Nothing lower imports anything higher, so each layer is testable on its own.

## Schema drift

nflverse rebuilt player stats on `nflfastR::calculate_stats()`, and the file keeps changing —
the docs show 115 columns in one build and 145 in a later one. Renames already in effect:

| Old (`nfl_data_py` era) | Current |
|---|---|
| `recent_team` | `team` |
| `interceptions` | `passing_interceptions` |
| `sacks` | `sacks_suffered` |
| `sack_yards` | `sack_yards_lost` |

`schema.py` maps canonical names onto whatever the loaded file uses, and every other module asks
for canonical names only. **When something breaks after an nflverse update, add an alias to
`schema.ALIASES` and nothing else needs to change.** `schema.report(df)` shows what resolved.

Two more things worth knowing about the current data:

- The player-stats file now contains **every position**, kickers and linemen included. `Settings.positions`
  filters to skill positions; widen it if you want K or IDP.
- Season totals come from `summary_level="reg"` rather than a hand-rolled groupby, so the old
  bug where `target_share` got summed across weeks can't recur. Games played and mean target
  share are still derived from the weekly frame.

## Testing

```bash
pytest                      # 22 tests, no network required
ruff check . && mypy src
```

Tests run against `tests/fake_nflverse.py`, which generates data in the current schema (including
the renamed columns) and installs itself as a fake `nflreadpy` module. That means the suite
covers the real `standardize` → `build_season` → `export_csv` path without downloading anything.

## Exporting to Google Sheets

Every CSV has one header row in `A1`, no index column, no blank rows, and one type per column —
the shape `Format ▸ Convert to table` expects.

1. **File ▸ Import ▸ Upload** → *Insert new sheet* → separator **Comma**.
2. Click any data cell → **Format ▸ Convert to table**.
3. Rate columns (`Catch Rate`, `Boom Rate`, `Comp %`, `Target Share (Avg)`, …) export as decimals,
   so set them to **Format ▸ Number ▸ Percent** for exact display.

Files produced:

| File | Contents |
|---|---|
| `2025_overall_top250_ppr.csv` | main board, 250 rows |
| `2025_flex_top200.csv` | RB/WR/TE re-ranked as FLEX |
| `2025_{qb,rb,wr,te}_rankings.csv` | one per position |
| `2025_weekly_game_log_top250.csv` | week-by-week rows for the top 250 |
| `2025_full_season_all_players.csv` | every skill-position player, uncapped |

## Data credit

NFL data via [nflverse](https://github.com/nflverse), licensed CC-BY 4.0 (FTN charting data is
CC-BY-SA 4.0).
