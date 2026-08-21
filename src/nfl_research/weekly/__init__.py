"""Within-season weekly PPR point forecast.

After a team is drafted, this module produces a week-by-week forecast for each
rostered player, incorporating current-season role (target share, snap usage),
recent game trend, and opponent matchup strength.

Usage::

    from nfl_research.weekly.data import load_multi_season_weekly
    from nfl_research.weekly.features import build_weekly_panel
    from nfl_research.weekly.cv import walk_forward_weekly_cv
    from nfl_research.weekly.evaluate import weekly_summary_table
"""
