"""Fantasy football PPR point total forecasting for the 2026 season.

Pipeline:
    1. data.py     - multi-season nflverse loading
    2. features.py - lag features / panel construction
    3. models.py   - model classes (sklearn-compatible API)
    4. cv.py       - walk-forward cross-validation
    5. evaluate.py - scoring and reporting

Quick start::

    from nfl_research.forecasting.data import load_multi_season
    from nfl_research.forecasting.features import build_panel
    from nfl_research.forecasting.cv import walk_forward_cv
    from nfl_research.forecasting.evaluate import summary_table
"""
from .cv import walk_forward_cv
from .data import load_multi_season
from .evaluate import summary_table
from .models import MODELS

__all__ = ["load_multi_season", "walk_forward_cv", "summary_table", "MODELS"]
