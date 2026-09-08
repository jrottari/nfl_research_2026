"""OAuth (user-account) auth for pushing data into a personal Google Sheet.

One-time setup (see repo README section / project memory for the exact
Google Cloud Console steps): download an OAuth client ID's JSON as
``CLIENT_SECRET_PATH``. The first call to ``get_client()`` opens a browser
consent screen; after that, the refresh token cached at ``TOKEN_PATH`` is
reused silently. Both files live under the Google-Drive-backed cache dir,
not in git.
"""

from __future__ import annotations

import gspread

from ..config import CACHE_DIR

GOOGLE_DIR = CACHE_DIR / "google"
CLIENT_SECRET_PATH = GOOGLE_DIR / "client_secret.json"
TOKEN_PATH = GOOGLE_DIR / "authorized_user.json"


def get_client() -> gspread.Client:
    GOOGLE_DIR.mkdir(parents=True, exist_ok=True)
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"Missing Google OAuth client secret at {CLIENT_SECRET_PATH}.\n"
            "Create one at https://console.cloud.google.com/apis/credentials "
            "(OAuth client ID, type 'Desktop app'), download the JSON, and "
            f"save it to that exact path."
        )
    return gspread.oauth(
        credentials_filename=str(CLIENT_SECRET_PATH),
        authorized_user_filename=str(TOKEN_PATH),
        scopes=(
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ),
    )
