"""Push lineup/ECR data into a persistent Google Sheet dashboard.

One spreadsheet, one tab per league for "what to start this week", plus a
combined summary tab and an append-only history tab for trend charts over
the season. The spreadsheet is created once and its ID cached to disk so
every later run updates the same Sheet rather than creating duplicates.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import gspread
import pandas as pd
from gspread.utils import ExportFormat

from ..config import CACHE_DIR
from .auth import get_client

SPREADSHEET_ID_PATH = CACHE_DIR / "google" / "spreadsheet_id.txt"
DEFAULT_TITLE = "Fantasy Lineup Dashboard 2026"

SUMMARY_TAB = "All Leagues Summary"
HISTORY_TAB = "History"


def get_or_create_spreadsheet(
    client: gspread.Client, title: str = DEFAULT_TITLE
) -> gspread.Spreadsheet:
    if SPREADSHEET_ID_PATH.exists():
        sheet_id = SPREADSHEET_ID_PATH.read_text().strip()
        if sheet_id:
            try:
                return client.open_by_key(sheet_id)
            except gspread.exceptions.APIError:
                pass  # cached id no longer valid (deleted?) -> fall through and recreate

    spreadsheet = client.create(title)
    SPREADSHEET_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPREADSHEET_ID_PATH.write_text(spreadsheet.id)
    return spreadsheet


def get_or_create_worksheet(
    spreadsheet: gspread.Spreadsheet, tab_name: str, rows: int = 200, cols: int = 20
) -> gspread.Worksheet:
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols)


def push_dataframe(spreadsheet: gspread.Spreadsheet, tab_name: str, df: pd.DataFrame) -> None:
    """Overwrite ``tab_name`` with ``df`` (header + values)."""
    ws = get_or_create_worksheet(spreadsheet, tab_name, rows=max(len(df) + 10, 50))
    ws.clear()
    if df.empty:
        return
    values = [df.columns.tolist()] + df.astype(object).where(df.notna(), "").values.tolist()
    ws.update(values=values, range_name="A1")


def append_history(spreadsheet: gspread.Spreadsheet, rows: pd.DataFrame) -> None:
    """Append rows to the running ``History`` tab, writing a header only the
    first time the tab is created (or if the schema has since changed —
    resets the log rather than silently misaligning columns under a stale
    header). Otherwise never clears — this is the trend log.
    """
    ws = get_or_create_worksheet(spreadsheet, HISTORY_TAB, rows=2000)
    existing = ws.get_all_values()
    values = rows.astype(object).where(rows.notna(), "").values.tolist()
    header = rows.columns.tolist()
    if not existing or existing[0] != header:
        ws.clear()
        ws.update(values=[header] + values, range_name="A1")
    else:
        ws.append_rows(values, value_input_option="USER_ENTERED")


def export_xlsx(
    spreadsheet: gspread.Spreadsheet, out_dir: Path, date: dt.date | None = None
) -> Path:
    """Download the whole spreadsheet (every tab, with formulas and charts)
    as ``<out_dir>/<YYYY-MM-DD>_fantasy_dashboard.xlsx`` via the Drive export
    endpoint, overwriting any same-day file. Returns the written path.
    """
    date = date or dt.date.today()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.isoformat()}_fantasy_dashboard.xlsx"
    out_path.write_bytes(spreadsheet.export(format=ExportFormat.EXCEL))
    return out_path
