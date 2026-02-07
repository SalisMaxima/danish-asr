"""Feedback storage abstraction (SQLite for local development)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class SqliteFeedbackStore:
    """Feedback store using SQLite."""

    def __init__(self, db_path: str = "feedback/feedback.db"):
        import sqlite3
        from pathlib import Path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                is_correct BOOLEAN NOT NULL,
                correct_class TEXT,
                user_note TEXT,
                confidence_rating TEXT,
                image_stats TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite feedback database initialized at %s", self.db_path)

    def save_feedback(
        self,
        *,
        image_path: str,
        predicted_class: str,
        predicted_confidence: float,
        is_correct: bool,
        correct_class: str | None = None,
        user_note: str | None = None,
        confidence_rating: str | None = None,
        image_stats: dict | None = None,
    ) -> str:
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO feedback (timestamp, image_path, predicted_class, predicted_confidence,
               is_correct, correct_class, user_note, confidence_rating, image_stats) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                image_path,
                predicted_class,
                predicted_confidence,
                is_correct,
                correct_class,
                user_note,
                confidence_rating,
                json.dumps(image_stats) if image_stats else None,
            ),
        )
        conn.commit()
        row_id = str(cursor.lastrowid)
        conn.close()
        return row_id

    def get_stats(self) -> dict:
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1")
        correct = cursor.fetchone()[0]
        cursor.execute("SELECT predicted_class, COUNT(*) FROM feedback GROUP BY predicted_class")
        class_dist = dict(cursor.fetchall())
        cursor.execute("SELECT timestamp, predicted_class, is_correct FROM feedback ORDER BY timestamp DESC LIMIT 10")
        recent = [{"timestamp": r[0], "predicted_class": r[1], "is_correct": bool(r[2])} for r in cursor.fetchall()]
        conn.close()
        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "class_distribution": class_dist,
            "recent_feedback": recent,
            "timestamp": datetime.now(UTC).isoformat(),
        }


def create_feedback_store() -> SqliteFeedbackStore:
    """Factory: returns SQLite feedback store."""
    db_path = os.environ.get("FEEDBACK_DB", "feedback/feedback.db")
    return SqliteFeedbackStore(db_path=db_path)
