"""User preference store — the persistent memory of human feedback.

Every time the user submits free-text feedback on a round, we append a record
here. The `UserPreferenceReviewer` reads these records and uses them as
few-shot examples to predict the user's verdict on new outputs.

This file is the durable memory that turns the system from "three LLMs talking
to each other" into "three LLMs talking under your supervision".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field

DEFAULT_PATH = Path("data/preferences.json")


class FeedbackRecord(BaseModel):
    session_id: str
    round: int
    output_snippet: str
    comment: str  # free-text from the user
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PreferenceStore:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: list[FeedbackRecord] = self._load()

    def _load(self) -> list[FeedbackRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
            return [FeedbackRecord(**r) for r in raw]
        except (json.JSONDecodeError, ValueError):
            return []

    def _flush(self) -> None:
        self.path.write_text(json.dumps([r.model_dump() for r in self._records], indent=2))

    def append(self, record: FeedbackRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._flush()

    def all(self) -> list[FeedbackRecord]:
        with self._lock:
            return list(self._records)

    def recent(self, n: int = 20) -> list[FeedbackRecord]:
        """Most-recent N records — fresher feedback should weigh more."""
        with self._lock:
            return list(self._records[-n:])
