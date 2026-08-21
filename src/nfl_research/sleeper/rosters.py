"""Pull a Sleeper user's rosters across every league for a season.

One call chain: username -> user_id -> leagues -> (roster, league settings,
team name) per league, with every player resolved to a name/position and,
where possible, our internal nflverse ``gsis_id`` so the roster can be joined
straight into ``nfl_research.weekly.projection``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import api
from .mapping import load_crosswalk

BENCH_SLOTS = {"BN", "IR", "TAXI"}
ROSTER_COLUMNS = [
    "sleeper_id", "gsis_id", "player_name", "position", "team", "is_starter", "is_ir", "is_taxi"
]


@dataclass
class LeagueRoster:
    league_id: str
    league_name: str
    team_name: str
    # Starting slots only, e.g. ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"].
    roster_positions: list[str]
    scoring_settings: dict
    # Columns: sleeper_id, gsis_id, player_name, position, team, is_starter, is_ir, is_taxi.
    players: pd.DataFrame


def resolve_user_id(username_or_id: str) -> str:
    """Accepts either a Sleeper username or a numeric user_id and returns the id."""
    if username_or_id.isdigit():
        return username_or_id
    user = api.get_user(username_or_id)
    if user is None:
        raise ValueError(f"No Sleeper user found for '{username_or_id}'")
    return user["user_id"]


def fetch_user_leagues(user_id: str, season: int) -> list[dict]:
    return api.get_user_leagues(user_id, season)


def _player_label(
    sleeper_id: str, players_dict: dict, crosswalk_row: pd.Series | None
) -> tuple[str, str, str]:
    """Returns (name, position, team), preferring the crosswalk (matches
    nflverse's naming) and falling back to Sleeper's own player dict.
    """
    sp = players_dict.get(sleeper_id, {})
    if crosswalk_row is not None:
        return crosswalk_row["name"], crosswalk_row["position"], crosswalk_row.get("team", "")
    fallback_name = f"{sp.get('first_name', '')} {sp.get('last_name', '')}".strip()
    name = sp.get("full_name") or fallback_name or sleeper_id
    return name, sp.get("position", "UNK"), sp.get("team", "") or ""


def build_league_roster(
    league_id: str,
    user_id: str,
    players_dict: dict,
    crosswalk: pd.DataFrame | None = None,
) -> LeagueRoster | None:
    league = api.get_league(league_id)
    if league is None:
        return None
    rosters = api.get_league_rosters(league_id)
    users = api.get_league_users(league_id)

    my_roster = next((r for r in rosters if r.get("owner_id") == user_id), None)
    if my_roster is None:
        return None

    league_user = next((u for u in users if u.get("user_id") == user_id), {})
    team_name = (league_user.get("metadata") or {}).get("team_name")
    team_name = team_name or league_user.get("display_name", "")

    crosswalk = crosswalk if crosswalk is not None else load_crosswalk()
    crosswalk_by_sleeper = crosswalk.set_index("sleeper_id")

    starters = set(my_roster.get("starters") or []) - {"0"}
    reserve = set(my_roster.get("reserve") or [])
    taxi = set(my_roster.get("taxi") or [])
    all_player_ids = my_roster.get("players") or []

    rows = []
    for sid in all_player_ids:
        crow = crosswalk_by_sleeper.loc[sid] if sid in crosswalk_by_sleeper.index else None
        name, position, team = _player_label(sid, players_dict, crow)
        gsis_id = crow["gsis_id"] if crow is not None else None
        rows.append(
            {
                "sleeper_id": sid,
                "gsis_id": gsis_id,
                "player_name": name,
                "position": position,
                "team": team,
                "is_starter": sid in starters,
                "is_ir": sid in reserve,
                "is_taxi": sid in taxi,
            }
        )

    roster_positions = [s for s in league.get("roster_positions", []) if s not in BENCH_SLOTS]

    players_df = pd.DataFrame(rows, columns=ROSTER_COLUMNS)

    return LeagueRoster(
        league_id=league_id,
        league_name=league.get("name", league_id),
        team_name=team_name,
        roster_positions=roster_positions,
        scoring_settings=league.get("scoring_settings", {}),
        players=players_df,
    )


def fetch_all_league_rosters(username_or_id: str, season: int) -> list[LeagueRoster]:
    """Every NFL league the user is in for ``season``, each with their own roster."""
    user_id = resolve_user_id(username_or_id)
    leagues = fetch_user_leagues(user_id, season)
    players_dict = api.get_all_players()
    crosswalk = load_crosswalk()

    out = []
    for league in leagues:
        lr = build_league_roster(league["league_id"], user_id, players_dict, crosswalk)
        if lr is not None:
            out.append(lr)
    return out
