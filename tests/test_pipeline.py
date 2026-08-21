"""Tests that run entirely offline against the synthetic fixture."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_nflverse import install_fake_module, make_season_totals, make_weekly  # noqa: E402

from nfl_research import exports, loaders, pipeline, rankings, schema  # noqa: E402
from nfl_research.config import Settings  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    weekly_raw = make_weekly()
    totals_raw = make_season_totals(weekly_raw)
    weekly = schema.coerce_numeric(schema.standardize(weekly_raw))
    weekly = weekly[weekly["season_type"] == "REG"].reset_index(drop=True)
    totals = schema.coerce_numeric(schema.standardize(totals_raw))
    return weekly, totals


@pytest.fixture(scope="module")
def boards(frames):
    weekly, totals = frames
    return pipeline.build_season(Settings(season=2025, top_n=250), weekly, totals)


# --------------------------------------------------------------------- schema
def test_new_schema_names_resolve():
    resolved = schema.resolve(make_weekly().columns)
    assert resolved["team"] == "team"
    assert resolved["interceptions"] == "passing_interceptions"
    assert resolved["sacks_taken"] == "sacks_suffered"
    assert resolved["player_name"] == "player_display_name"


def test_old_schema_names_still_resolve():
    old = pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "position",
            "season",
            "recent_team",
            "interceptions",
            "sacks",
            "fantasy_points",
            "fantasy_points_ppr",
        ]
    )
    resolved = schema.resolve(old.columns)
    assert resolved["team"] == "recent_team"
    assert resolved["interceptions"] == "interceptions"


def test_standardize_does_not_duplicate_columns():
    out = schema.standardize(make_weekly())
    assert not out.columns.duplicated().any()
    assert "player_name" in out.columns and "player_display_name" not in out.columns


def test_standardize_raises_on_missing_required():
    with pytest.raises(KeyError, match="missing required columns"):
        schema.standardize(pd.DataFrame(columns=["week", "team"]))


# -------------------------------------------------------------------- loaders
def test_loader_converts_polars_and_filters(monkeypatch):
    install_fake_module(monkeypatch)
    weekly = loaders.load_weekly(2025, season_type="REG", positions=("QB", "RB", "WR", "TE"))
    assert isinstance(weekly, pd.DataFrame)
    assert set(weekly["season_type"]) == {"REG"}
    assert set(weekly["position"]) <= {"QB", "RB", "WR", "TE"}
    assert "team" in weekly.columns


# --------------------------------------------------------------------- boards
def test_board_reaches_requested_depth(boards):
    assert len(boards["top_n"]) == 250
    assert boards["top_n"]["rank_ppr"].iloc[0] == 1


def test_points_are_monotonically_decreasing(boards):
    points = boards["top_n"]["points_ppr"]
    assert points.is_monotonic_decreasing


def test_scoring_formats_are_ordered(boards):
    board = boards["top_n"]
    assert (board["points_ppr"] >= board["points_half"] - 1e-9).all()
    assert (board["points_half"] >= board["points_std"] - 1e-9).all()


def test_ppg_matches_points_over_games(boards):
    board = boards["top_n"]
    expected = board["points_ppr"] / board["games"]
    pd.testing.assert_series_equal(board["ppg_ppr"], expected, check_names=False)


def test_no_division_by_zero_infinities(boards):
    numeric = boards["overall"].select_dtypes("number")
    assert not numeric.isin([float("inf"), float("-inf")]).any().any()


def test_target_share_is_a_mean_not_a_sum(boards):
    shares = boards["overall"]["target_share"].dropna()
    assert shares.max() <= 1.0, "target share should be a per-game mean"


def test_positions_filtered_to_skill_only(boards):
    assert set(boards["overall"]["position"]) <= set(Settings().positions)


def test_flex_board_excludes_qbs(boards):
    assert "QB" not in set(boards["flex"]["position"])
    assert boards["flex"]["flex_rank"].tolist() == list(range(1, len(boards["flex"]) + 1))


def test_game_log_covers_only_board_players(boards):
    log_players = set(boards["game_log"]["player_name"])
    board_players = set(boards["top_n"]["player_name"])
    assert log_players <= board_players


def test_replacement_levels_present_for_each_position(boards):
    assert set(boards["replacement"]["position"]) == {"QB", "RB", "WR", "TE"}
    assert boards["replacement"]["replacement_ppg"].notna().all()


def test_tiers_increase_as_ppg_falls(boards):
    rbs = boards["RB"].dropna(subset=["tier"]).sort_values("ppg_ppr", ascending=False)
    assert rbs["tier"].is_monotonic_increasing


def test_min_games_gate_on_per_game_ranks(frames):
    weekly, totals = frames
    settings = Settings(season=2025, min_games=6)
    board = pipeline.build_season(settings, weekly, totals)["overall"]
    assert board.loc[board["games"] < 6, "rank_ppg"].isna().all()


# -------------------------------------------------------------------- exports
def test_sheets_ready_headers_are_pretty(boards):
    out = exports.sheets_ready(boards["top_n"])
    assert "PPR Pts" in out.columns
    assert not any(" " in c and c.islower() for c in out.columns)
    assert out.columns.duplicated().sum() == 0


def test_export_csv_is_table_shaped(boards, tmp_path):
    path = exports.export_csv(boards["top_n"], "board.csv", tmp_path)
    rows = list(csv.reader(path.open()))
    header, body = rows[0], rows[1:]
    assert len(body) == 250
    assert all(len(r) == len(header) for r in body), "ragged rows break Sheets tables"
    assert all(h.strip() for h in header), "empty header cell"
    assert not any(all(not c.strip() for c in r) for r in body), "blank row"


def test_export_all_writes_every_board(boards, tmp_path):
    settings = Settings(season=2025)
    written = exports.export_all(pipeline.to_export_map(boards, settings), tmp_path, quiet=True)
    assert len(written) == 8
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_formula_injection_is_neutralized():
    df = pd.DataFrame({"player_name": ["=cmd()", "Normal Name"]})
    out = exports.sheets_ready(df)
    assert out["Player"].iloc[0].startswith("'")
    assert out["Player"].iloc[1] == "Normal Name"


# ------------------------------------------------------------------ utilities
def test_assign_tiers_breaks_on_gap():
    values = pd.Series([20.0, 19.8, 19.7, 12.0, 11.9])
    tiers = rankings.assign_tiers(values, gap=1.5)
    assert tiers.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
