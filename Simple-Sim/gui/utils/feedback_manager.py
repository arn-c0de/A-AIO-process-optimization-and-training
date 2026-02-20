"""Feedback management for manual prediction corrections in Analysis tab."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOG = logging.getLogger(__name__)

VERDICT_THUMBS_UP = "thumbs_up"
VERDICT_THUMBS_DOWN = "thumbs_down"


class FeedbackManager:
    """Manage append-only user feedback stored in feedback.jsonl."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.feedback_path = self.dataset_dir / "feedback.jsonl"
        self._entries: list[dict[str, Any]] = []
        self._latest_by_sample: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        """Load feedback entries and build latest-per-sample index."""
        self._entries.clear()
        self._latest_by_sample.clear()
        if not self.feedback_path.exists():
            return

        try:
            with self.feedback_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    sample_id = str(entry.get("id") or "").strip()
                    verdict = str(entry.get("verdict") or "").strip()
                    if not sample_id or verdict not in (VERDICT_THUMBS_UP, VERDICT_THUMBS_DOWN):
                        continue
                    self._entries.append(entry)
                    self._latest_by_sample[sample_id] = entry
        except Exception as exc:
            _LOG.warning("Failed to load feedback from %s: %s", self.feedback_path, exc)

    def append_feedback(
        self,
        sample_id: str,
        verdict: str,
        corrected_class: Optional[str] = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Append one feedback entry to feedback.jsonl."""
        sample_id = str(sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id cannot be empty")
        if verdict not in (VERDICT_THUMBS_UP, VERDICT_THUMBS_DOWN):
            raise ValueError(f"Unsupported verdict: {verdict}")

        corrected = str(corrected_class or "").strip()
        if verdict == VERDICT_THUMBS_DOWN and not corrected:
            raise ValueError("corrected_class is required for thumbs_down")
        if verdict == VERDICT_THUMBS_UP:
            corrected = ""

        entry = {
            "schema_version": 1,
            "id": sample_id,
            "verdict": verdict,
            "corrected_class": corrected,
            "note": str(note or "").strip(),
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self._entries.append(entry)
        self._latest_by_sample[sample_id] = entry
        return entry

    def latest_feedback_for(self, sample_id: str) -> Optional[dict[str, Any]]:
        """Return latest feedback entry for a sample if available."""
        return self._latest_by_sample.get(sample_id)

    def latest_feedback_by_sample(self) -> Dict[str, dict[str, Any]]:
        """Return copy of latest-per-sample map."""
        return dict(self._latest_by_sample)

    def effective_label(self, sample_id: str, original_label: str) -> str:
        """Return corrected label when latest feedback is thumbs_down; otherwise original."""
        entry = self._latest_by_sample.get(sample_id)
        if not entry:
            return original_label
        corrected = str(entry.get("corrected_class") or "").strip()
        if str(entry.get("verdict") or "").strip() == VERDICT_THUMBS_DOWN and corrected:
            return corrected
        return original_label

    def summary(self) -> dict[str, Any]:
        """Return simple feedback summary statistics."""
        thumbs_up = 0
        thumbs_down = 0
        corrected = 0
        latest_ts = ""

        for entry in self._entries:
            verdict = str(entry.get("verdict") or "").strip()
            if verdict == VERDICT_THUMBS_UP:
                thumbs_up += 1
            elif verdict == VERDICT_THUMBS_DOWN:
                thumbs_down += 1
            if str(entry.get("corrected_class") or "").strip():
                corrected += 1
            ts = str(entry.get("timestamp") or "")
            if ts and ts > latest_ts:
                latest_ts = ts

        return {
            "entries": len(self._entries),
            "samples_with_feedback": len(self._latest_by_sample),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "corrected_entries": corrected,
            "latest_timestamp": latest_ts,
        }

    def history(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return newest feedback entries first."""
        if limit <= 0:
            return []
        return list(reversed(self._entries[-limit:]))
