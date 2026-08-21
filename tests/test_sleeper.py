"""Offline tests for the Sleeper integration — API responses are mocked with
fixtures shaped like the real endpoints (see fake_sleeper.py); no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_sleeper as fx

from nfl_research.sleeper import rosters as rosters_mod
from nfl_research.sleeper.optimize import adjust_for_scoring, modeled_slots, optimize_lineup

# ---- resolve_user_id --------------------------------------------------------


def test_resolve_user_id_numeric_short_circuits(monkeypatch):
    called = {"n": 0}

    def _fake_get_user(u):
        called["n"] += 1
        return None

    monkeypatch.setattr(rosters_mod.api, "get_user", _fake_get_user)
    assert rosters_mod.resolve_user_id("111") == "111"
    assert called["n"] == 0


def test_resolve_user_id_by_username(monkeypatch):
    monkeypatch.setattr(rosters_mod.api, "get_user", lambda u: fx.USER if u == "testuser" else None)
    assert rosters_mod.resolve_user_id("testuser") == "111"


def test_resolve_user_id_unknown_raises(monkeypatch):
    monkeypatch.setattr(rosters_mod.api, "get_user", lambda u: None)
    with pytest.raises(ValueError):
        rosters_mod.resolve_user_id("nobody")


# ---- build_league_roster / fetch_all_league_rosters -------------------------


@pytest.fixture
def patched_api(monkeypatch):
    monkeypatch.setattr(rosters_mod.api, "get_user", lambda u: fx.USER if u == "testuser" else None)
    monkeypatch.setattr(rosters_mod.api, "get_user_leagues", lambda uid, season: [fx.LEAGUE])
    monkeypatch.setattr(
        rosters_mod.api, "get_league", lambda lid: fx.LEAGUE if lid == "L1" else None
    )
    monkeypatch.setattr(rosters_mod.api, "get_league_rosters", lambda lid: fx.ROSTERS)
    monkeypatch.setattr(rosters_mod.api, "get_league_users", lambda lid: fx.LEAGUE_USERS)
    monkeypatch.setattr(rosters_mod.api, "get_all_players", lambda: fx.PLAYERS_DICT)
    monkeypatch.setattr(rosters_mod, "load_crosswalk", fx.make_crosswalk)


def test_build_league_roster_basic(patched_api):
    lr = rosters_mod.build_league_roster("L1", "111", fx.PLAYERS_DICT, fx.make_crosswalk())
    assert lr is not None
    assert lr.team_name == "The Testers"
    assert lr.roster_positions == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert len(lr.players) == 9


def test_build_league_roster_starter_flags(patched_api):
    lr = rosters_mod.build_league_roster("L1", "111", fx.PLAYERS_DICT, fx.make_crosswalk())
    by_id = lr.players.set_index("sleeper_id")
    assert by_id.loc["1001", "is_starter"]
    assert not by_id.loc["1007", "is_starter"]  # bench RB, not in starters list


def test_build_league_roster_gsis_mapping(patched_api):
    lr = rosters_mod.build_league_roster("L1", "111", fx.PLAYERS_DICT, fx.make_crosswalk())
    by_id = lr.players.set_index("sleeper_id")
    assert by_id.loc["1001", "gsis_id"] == "00-1001"
    # K/DEF aren't in the player-level crosswalk — gsis_id stays unset.
    assert pd.isna(by_id.loc["2001", "gsis_id"])
    assert pd.isna(by_id.loc["3001", "gsis_id"])
    assert by_id.loc["3001", "position"] == "DEF"


def test_build_league_roster_no_owner_returns_none(patched_api):
    lr = rosters_mod.build_league_roster("L1", "999", fx.PLAYERS_DICT, fx.make_crosswalk())
    assert lr is None


def test_build_league_roster_undrafted_league_has_columns(patched_api, monkeypatch):
    """Regression test: an undrafted (pre_draft) league returns an empty
    players list. build_league_roster must still produce a properly-columned
    (if empty) DataFrame so downstream .isin()/boolean-mask code doesn't
    KeyError on a missing 'position' column."""
    monkeypatch.setattr(rosters_mod.api, "get_league_rosters", lambda lid: fx.ROSTERS_UNDRAFTED)
    lr = rosters_mod.build_league_roster("L1", "111", fx.PLAYERS_DICT, fx.make_crosswalk())
    assert lr is not None
    assert lr.players.empty
    assert list(lr.players.columns) == rosters_mod.ROSTER_COLUMNS
    assert lr.players["position"].isin(["QB", "RB"]).sum() == 0


def test_fetch_all_league_rosters(patched_api):
    result = rosters_mod.fetch_all_league_rosters("testuser", 2025)
    assert len(result) == 1
    assert result[0].league_name == "Test League"


# ---- optimize_lineup ---------------------------------------------------------


def _players(rows):
    return pd.DataFrame(rows, columns=["player_name", "position", "proj_points"])


def test_optimize_lineup_fills_basic_slots():
    players = _players(
        [
            ("QB1", "QB", 20.0),
            ("RB1", "RB", 15.0),
            ("WR1", "WR", 12.0),
        ]
    )
    starters, bench, empty = optimize_lineup(players, ["QB", "RB", "WR"])
    assert set(starters["player_name"]) == {"QB1", "RB1", "WR1"}
    assert bench.empty
    assert empty == []


def test_optimize_lineup_flex_takes_best_leftover():
    players = _players(
        [
            ("RB1", "RB", 20.0),
            ("RB2", "RB", 15.0),
            ("RB3", "RB", 10.0),
            ("WR1", "WR", 12.0),
            ("WR2", "WR", 8.0),
        ]
    )
    starters, bench, empty = optimize_lineup(players, ["RB", "RB", "WR", "WR", "FLEX"])
    flex_row = starters[starters["slot"] == "FLEX"].iloc[0]
    # Best two RBs and both WRs fill their own slots; FLEX should go to the
    # best leftover player (RB3, 10.0) rather than benching it for a worse pick.
    assert flex_row["player_name"] == "RB3"
    assert bench.empty
    assert starters["proj_points"].sum() == pytest.approx(20 + 15 + 10 + 12 + 8)


def test_optimize_lineup_reports_empty_slot_when_understaffed():
    players = _players([("QB1", "QB", 20.0)])
    starters, bench, empty = optimize_lineup(players, ["QB", "TE"])
    assert empty == ["TE"]
    assert list(starters["player_name"]) == ["QB1"]


def test_optimize_lineup_ineligible_player_not_assigned():
    players = _players([("QB1", "QB", 99.0), ("RB1", "RB", 5.0)])
    starters, bench, empty = optimize_lineup(players, ["RB"])
    assert list(starters["player_name"]) == ["RB1"]
    assert "QB1" in bench["player_name"].values


def test_modeled_slots_excludes_k_def_bn():
    slots = modeled_slots(fx.LEAGUE["roster_positions"])
    assert "K" not in slots
    assert "DEF" not in slots
    assert "BN" not in slots
    assert "FLEX" in slots
    assert slots.count("RB") == 2


# ---- adjust_for_scoring -------------------------------------------------------


def _scoring_board(proj: float, floor: float, ceiling: float, receptions: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "proj_points": [proj],
            "floor": [floor],
            "ceiling": [ceiling],
            "receptions_ma3": [receptions],
        }
    )


def test_adjust_for_scoring_full_ppr_is_noop():
    board = _scoring_board(10.0, 5.0, 15.0, 4.0)
    out = adjust_for_scoring(board, {"rec": 1.0})
    pd.testing.assert_frame_equal(out, board)


def test_adjust_for_scoring_half_ppr_shifts_down():
    board = _scoring_board(10.0, 5.0, 15.0, 4.0)
    out = adjust_for_scoring(board, {"rec": 0.5})
    assert out["proj_points"].iloc[0] == pytest.approx(8.0)
    assert out["floor"].iloc[0] == pytest.approx(3.0)
    assert out["ceiling"].iloc[0] == pytest.approx(13.0)


def test_adjust_for_scoring_standard_clips_at_zero():
    board = _scoring_board(2.0, 1.0, 3.0, 5.0)
    out = adjust_for_scoring(board, {"rec": 0.0})
    assert (out[["proj_points", "floor", "ceiling"]] >= 0).all().all()
