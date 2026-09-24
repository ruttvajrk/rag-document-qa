"""
query_log.py
------------
Stores every question asked through the API in a SQLite database, so the
app's usage and performance can be analysed later with plain SQL
(see sql/analysis.sql).

Two tables:
  queries  - one row per question (answer, response time, was it declined?)
  sources  - one row per retrieved chunk for each question (file, page, score)
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

REFUSAL_TEXT = "I don't have enough information to answer that."

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at         TEXT    NOT NULL,
    question         TEXT    NOT NULL,
    answer           TEXT    NOT NULL,
    top_k            INTEGER NOT NULL,
    retrieval_ms     REAL    NOT NULL,
    generation_ms    REAL    NOT NULL,
    total_ms         REAL    NOT NULL,
    top_score        REAL,
    declined         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER NOT NULL REFERENCES queries(id),
    rank       INTEGER NOT NULL,
    source     TEXT,
    page       INTEGER,
    score      REAL
);

CREATE INDEX IF NOT EXISTS idx_sources_query ON sources(query_id);
"""


class QueryLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        folder = os.path.dirname(db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def log(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        top_k: int,
        retrieval_ms: float,
        generation_ms: float,
    ) -> int:
        top_score = max((s.get("score", 0.0) for s in sources), default=None)
        declined = int(REFUSAL_TEXT.lower() in answer.lower())

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO queries
                   (asked_at, question, answer, top_k, retrieval_ms,
                    generation_ms, total_ms, top_score, declined)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    question,
                    answer,
                    top_k,
                    round(retrieval_ms, 1),
                    round(generation_ms, 1),
                    round(retrieval_ms + generation_ms, 1),
                    top_score,
                    declined,
                ),
            )
            query_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO sources (query_id, rank, source, page, score) VALUES (?, ?, ?, ?, ?)",
                [
                    (query_id, rank, s.get("source"), _to_int(s.get("page")), s.get("score"))
                    for rank, s in enumerate(sources, start=1)
                ],
            )
        return query_id

    def stats(self) -> Dict:
        """Summary numbers for the /api/stats endpoint."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*),
                          ROUND(AVG(total_ms), 1),
                          ROUND(AVG(retrieval_ms), 1),
                          ROUND(AVG(generation_ms), 1),
                          SUM(declined)
                   FROM queries"""
            ).fetchone()
            top_docs = conn.execute(
                """SELECT source, COUNT(*) AS times_retrieved
                   FROM sources
                   GROUP BY source
                   ORDER BY times_retrieved DESC
                   LIMIT 5"""
            ).fetchall()
        return {
            "total_questions": row[0],
            "avg_total_ms": row[1],
            "avg_retrieval_ms": row[2],
            "avg_generation_ms": row[3],
            "declined_answers": row[4] or 0,
            "most_retrieved_documents": [
                {"source": s, "times_retrieved": n} for s, n in top_docs
            ],
        }


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
