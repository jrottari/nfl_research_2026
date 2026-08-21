"""Sleeper integration: pull a user's rosters across leagues and optimize
each one's starting lineup against the weekly forecasting system.

Usage::

    from nfl_research.sleeper.rosters import fetch_all_league_rosters
    from nfl_research.sleeper.optimize import optimize_lineup, modeled_slots

See ``scripts/sleeper_lineup.py`` for the end-to-end command.
"""
