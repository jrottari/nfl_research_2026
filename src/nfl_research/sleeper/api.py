"""Thin client for Sleeper's public, unauthenticated read API.

https://docs.sleeper.com/ — no API key needed. Every function here does one
GET and returns parsed JSON (dict/list) or ``None`` on a 404 (e.g. an unknown
username). Network errors raise ``requests.RequestException`` so callers see
a real failure rather than silently getting an empty result.

The one endpoint that needs special handling is ``/players/nfl``: it returns
every player Sleeper has ever rostered (~5-10 MB of JSON) and Sleeper's own
docs ask API consumers to call it **at most once per day**. ``get_all_players``
enforces that with a disk cache in ``data/cache/sleeper_players.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.sleeper.app/v1"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
PLAYERS_CACHE_PATH = CACHE_DIR / "sleeper_players.json"
PLAYERS_CACHE_MAX_AGE_SECONDS = 20 * 60 * 60  # Sleeper: fetch at most once/day


def _get(path: str, timeout: float = 15.0) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_nfl_state() -> dict:
    """Current NFL season/week per Sleeper (season_type: pre/regular/post)."""
    return _get("/state/nfl")


def get_user(username_or_id: str) -> dict | None:
    return _get(f"/user/{username_or_id}")


def get_user_leagues(user_id: str, season: int, sport: str = "nfl") -> list[dict]:
    result = _get(f"/user/{user_id}/leagues/{sport}/{season}")
    return result or []


def get_league(league_id: str) -> dict | None:
    return _get(f"/league/{league_id}")


def get_league_rosters(league_id: str) -> list[dict]:
    result = _get(f"/league/{league_id}/rosters")
    return result or []


def get_league_users(league_id: str) -> list[dict]:
    result = _get(f"/league/{league_id}/users")
    return result or []


def get_all_players(sport: str = "nfl", force_refresh: bool = False) -> dict:
    """Every Sleeper player, keyed by sleeper player_id. Disk-cached for
    ``PLAYERS_CACHE_MAX_AGE_SECONDS`` per Sleeper's rate-limit guidance —
    this is a large, slow-changing payload, not something to refetch per run.
    """
    if not force_refresh and PLAYERS_CACHE_PATH.exists():
        age = time.time() - PLAYERS_CACHE_PATH.stat().st_mtime
        if age < PLAYERS_CACHE_MAX_AGE_SECONDS:
            with open(PLAYERS_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)

    players = _get(f"/players/{sport}", timeout=60.0) or {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLAYERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(players, f)
    return players
