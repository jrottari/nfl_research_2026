"""Native Sheets charts, built via the raw Sheets API (through gspread's
``Spreadsheet.batch_update`` passthrough) rather than by hand.

Every chart here is **deleted and recreated fresh on each run** rather than
created once and left alone — the tabs they read from get fully
overwritten (``push_dataframe`` clears + rewrites) every time
``update_dashboard.py`` runs, and row counts change week to week (more
history rows, roster changes). A chart pinned to a stale range would
silently stop growing; delete-then-recreate keeps it matched to whatever
was just written.
"""

from __future__ import annotations

import gspread

ChartType = str  # "LINE" or "COLUMN" (Sheets API basicChart.chartType values used here)


def _existing_chart_ids(spreadsheet: gspread.Spreadsheet, sheet_id: int) -> list[int]:
    meta = spreadsheet.fetch_sheet_metadata(
        {"fields": "sheets(properties.sheetId,charts.chartId)"}
    )
    for s in meta["sheets"]:
        if s["properties"]["sheetId"] == sheet_id:
            return [c["chartId"] for c in s.get("charts", [])]
    return []


def _grid_range(sheet_id: int, start_row: int, end_row: int, start_col: int, end_col: int) -> dict:
    return {
        "sheetId": sheet_id,
        "startRowIndex": start_row,
        "endRowIndex": end_row,
        "startColumnIndex": start_col,
        "endColumnIndex": end_col,
    }


def replace_chart(
    spreadsheet: gspread.Spreadsheet,
    sheet_id: int,
    chart_type: ChartType,
    title: str,
    n_data_rows: int,
    domain_col: int,
    series_cols: list[int],
    anchor_col_offset: int = 2,
    anchor_row: int = 0,
    reverse_y: bool = False,
    width: int = 700,
    height: int = 400,
) -> None:
    """Delete any existing chart(s) on ``sheet_id`` and add one fresh
    ``chart_type`` ("LINE" or "COLUMN") chart reading ``domain_col`` (0-indexed,
    e.g. the week/league column) as the X-axis and ``series_cols`` (0-indexed)
    as one series each, over rows ``[0, n_data_rows)`` (row 0 = header, used
    as series names). Anchored ``anchor_col_offset`` columns right of the
    data so it doesn't overlap the table.
    """
    requests = [
        {"deleteEmbeddedObject": {"objectId": cid}}
        for cid in _existing_chart_ids(spreadsheet, sheet_id)
    ]

    domain_range = _grid_range(sheet_id, 0, n_data_rows, domain_col, domain_col + 1)
    series = [
        {"series": {"sourceRange": {"sources": [
            _grid_range(sheet_id, 0, n_data_rows, c, c + 1)
        ]}}}
        for c in series_cols
    ]
    # Note: BasicChartAxis has no axis-reversal field in the current Sheets
    # API (verified — the API rejects "reverseDirection" outright). Where
    # "lower = better" matters, that's conveyed in the chart title instead;
    # ``reverse_y`` is accepted but currently a no-op.
    left_axis = {"position": "LEFT_AXIS"}

    chart_spec = {
        "title": title,
        "basicChart": {
            "chartType": chart_type,
            "legendPosition": "RIGHT_LEGEND",
            "axis": [{"position": "BOTTOM_AXIS"}, left_axis],
            "domains": [{"domain": {"sourceRange": {"sources": [domain_range]}}}],
            "series": series,
            "headerCount": 1,
        },
    }
    requests.append({
        "addChart": {
            "chart": {
                "spec": chart_spec,
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": sheet_id,
                            "rowIndex": anchor_row,
                            "columnIndex": domain_col + len(series_cols) + 1 + anchor_col_offset,
                        },
                        "widthPixels": width,
                        "heightPixels": height,
                    }
                },
            }
        }
    })
    spreadsheet.batch_update({"requests": requests})
