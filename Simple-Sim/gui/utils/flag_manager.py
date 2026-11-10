"""Flag management for dataset quality issues."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from simple_sim.io_utils import atomic_write_text, now_iso

_LOG = logging.getLogger(__name__)


class FlagManager:
    """Manage sample flags for dataset quality issues."""

    def __init__(self, dataset_dir: Path) -> None:
        """Initialize flag manager.

        Args:
            dataset_dir: Dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.flags_path = self.dataset_dir / "flags.jsonl"
        self._flags: dict[str, list[dict[str, Any]]] = {}  # id -> list of flag dicts
        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from JSONL file."""
        self._flags.clear()

        if not self.flags_path.exists():
            return

        try:
            lines = self.flags_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            _LOG.warning("Failed to load flags: %s", e)
            return
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            # One bad line must not hide the flags after it (the next save would drop them for good).
            try:
                flag_dict = json.loads(line)
                sample_id = flag_dict['id']
                flag_dict['flag']
            except (ValueError, KeyError, TypeError) as e:
                _LOG.warning("Skipping invalid flag at %s:%d: %s", self.flags_path, line_num, e)
                continue
            self._flags.setdefault(sample_id, []).append(flag_dict)

    def _save_flags(self) -> None:
        """Save flags to JSONL file."""
        try:
            self.flags_path.parent.mkdir(parents=True, exist_ok=True)

            atomic_write_text(
                self.flags_path,
                "".join(json.dumps(flag_dict) + '\n' for flag_list in self._flags.values() for flag_dict in flag_list),
            )
        except Exception as e:
            _LOG.warning("Failed to save flags: %s", e)

    def add_flag(self, sample_id: str, flag_type: str, reason: str) -> None:
        """Add a flag to a sample.

        Args:
            sample_id: Sample ID
            flag_type: Flag type (CORRUPT, OUTLIER, SUSPECT_LABEL, etc.)
            reason: Human-readable reason
        """
        if self._add(sample_id, flag_type, reason):
            self._save_flags()

    def add_flags(self, flags: list[tuple[str, str, str]]) -> int:
        """Add ``(sample_id, flag_type, reason)`` flags with one save; returns how many were new."""
        added = sum(self._add(sample_id, flag_type, reason) for sample_id, flag_type, reason in flags)
        if added:
            self._save_flags()
        return added

    def clear(self) -> None:
        """Remove every flag with one save."""
        if self._flags:
            self._flags.clear()
            self._save_flags()

    def reload(self) -> None:
        """Re-read ``flags.jsonl`` (another manager of the same dataset may have written it)."""
        self._load_flags()

    def _add(self, sample_id: str, flag_type: str, reason: str) -> bool:
        flag_list = self._flags.setdefault(sample_id, [])
        # Don't duplicate flags
        if any(existing_flag['flag'] == flag_type for existing_flag in flag_list):
            return False
        flag_list.append({'id': sample_id, 'flag': flag_type, 'reason': reason, 'timestamp': now_iso()})
        return True

    def remove_flag(self, sample_id: str, flag_type: str | None = None) -> None:
        """Remove flag(s) from a sample.

        Args:
            sample_id: Sample ID
            flag_type: Specific flag type to remove, or None to remove all flags
        """
        if sample_id not in self._flags:
            return

        if flag_type is None:
            # Remove all flags for this sample
            del self._flags[sample_id]
        else:
            # Remove specific flag type
            self._flags[sample_id] = [
                f for f in self._flags[sample_id]
                if f['flag'] != flag_type
            ]

            # Clean up if no flags left
            if not self._flags[sample_id]:
                del self._flags[sample_id]

        self._save_flags()

    def get_flags(self, sample_id: str) -> list[dict[str, Any]]:
        """Get all flags for a sample.

        Args:
            sample_id: Sample ID

        Returns:
            List of flag dictionaries
        """
        return self._flags.get(sample_id, [])

    def get_all_flagged_samples(self) -> list[str]:
        """Get all sample IDs with flags.

        Returns:
            List of sample IDs
        """
        return list(self._flags.keys())
