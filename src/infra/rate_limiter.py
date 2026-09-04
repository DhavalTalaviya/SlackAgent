import os
import sqlite3
import time
from contextlib import closing

from src.config import config

_DB_PATH = os.path.join(config.workspace_state_dir, "usage.sqlite")
_DAY_SECONDS = 86400


def _connect() -> sqlite3.Connection:
    os.makedirs(config.workspace_state_dir, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS requests (user_id TEXT NOT NULL, ts REAL NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_user_ts ON requests(user_id, ts)")
    return conn


def check_and_record(user_id: str) -> tuple[bool, str, str]:
    """Enforces two per-user caps: a short burst window (stops a runaway loop
    or bug firing rapid requests) and a daily total (bounds a single chatty
    user's sustained spend). Returns (allowed, message_if_blocked,
    reason_code) -- reason_code is "burst"/"daily"/"" so callers can label
    metrics without parsing the human-readable message. Records this request
    if allowed, so it counts toward both windows."""
    now = time.time()
    burst_start = now - config.rate_limit_burst_window_seconds
    day_start = now - _DAY_SECONDS

    with closing(_connect()) as conn:
        conn.execute("DELETE FROM requests WHERE ts < ?", (day_start,))

        burst_count = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE user_id = ? AND ts >= ?", (user_id, burst_start)
        ).fetchone()[0]
        if burst_count >= config.rate_limit_burst_max:
            conn.commit()
            return False, (
                f"You're asking questions too quickly (max {config.rate_limit_burst_max} per "
                f"{config.rate_limit_burst_window_seconds}s). Please wait a moment and try again."
            ), "burst"

        day_count = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE user_id = ? AND ts >= ?", (user_id, day_start)
        ).fetchone()[0]
        if day_count >= config.rate_limit_daily_max:
            conn.commit()
            return False, (
                f"You've hit today's limit of {config.rate_limit_daily_max} questions. "
                f"Please try again tomorrow."
            ), "daily"

        conn.execute("INSERT INTO requests (user_id, ts) VALUES (?, ?)", (user_id, now))
        conn.commit()

    return True, "", ""
