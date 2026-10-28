"""
Tiny sqlite logger: every question + answer + feedback, used by:
  - the admin panel (question log, thumbs-down list, "last synced" time)
  - the per-user rate limiter
  - the semantic cache
No external DB needed -- sqlite file is fine at this scale, and the schema
is simple enough to migrate to Postgres later if the app grows.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT,              -- JSON list of {url, title}
    was_refusal INTEGER DEFAULT 0,
    feedback TEXT,             -- 'up' | 'down' | NULL
    latency_ms INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_user ON questions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_questions_feedback ON questions(feedback);

CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at REAL NOT NULL,
    pages_indexed INTEGER,
    chunks_indexed INTEGER,
    duration_s REAL,
    note TEXT
);
"""


def _connect() -> sqlite3.Connection:
    Path(config.LOG_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_question(user_id: Optional[str], question: str, answer: str, sources: str,
                  was_refusal: bool, latency_ms: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO questions (user_id, question, answer, sources, was_refusal, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, question, answer, sources, int(was_refusal), latency_ms, time.time()),
        )
        return cur.lastrowid


def set_feedback(question_id: int, feedback: str):
    with get_conn() as conn:
        conn.execute("UPDATE questions SET feedback = ? WHERE id = ?", (feedback, question_id))


def count_questions_last_hour(user_id: str) -> int:
    cutoff = time.time() - 3600
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM questions WHERE user_id = ? AND created_at > ?",
            (user_id, cutoff),
        ).fetchone()
        return row["n"] if row else 0


def recent_questions(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM questions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def thumbs_down_questions(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM questions WHERE feedback = 'down' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def log_sync(pages_indexed: int, chunks_indexed: int, duration_s: float, note: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sync_log (ran_at, pages_indexed, chunks_indexed, duration_s, note) VALUES (?, ?, ?, ?, ?)",
            (time.time(), pages_indexed, chunks_indexed, duration_s, note),
        )


def last_sync():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sync_log ORDER BY ran_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None
