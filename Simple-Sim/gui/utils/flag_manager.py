"""Flag management for dataset quality issues."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from simple_sim.io_utils import now_iso

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
            with open(self.flags_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    flag_dict = json.loads(line)
                    sample_id = flag_dict['id']

                    if sample_id not in self._flags:
                        self._flags[sample_id] = []

                    self._flags[sample_id].append(flag_dict)
        except Exception as e:
            _LOG.warning("Failed to load flags: %s", e)

    def _save_flags(self) -> None:
        """Save flags to JSONL file."""
        try:
            self.flags_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.flags_path, 'w') as f:
                for _sample_id, flag_list in self._flags.items():
                    for flag_dict in flag_list:
                        f.write(json.dumps(flag_dict) + '\n')
        except Exception as e:
            _LOG.warning("Failed to save flags: %s", e)

    def add_flag(self, sample_id: str, flag_type: str, reason: str) -> None:
        """Add a flag to a sample.

        Args:
            sample_id: Sample ID
            flag_type: Flag type (CORRUPT, OUTLIER, SUSPECT_LABEL, etc.)
            reason: Human-readable reason
        """
        flag_dict = {
            'id': sample_id,
            'flag': flag_type,
            'reason': reason,
            'timestamp': now_iso()
        }

        if sample_id not in self._flags:
            self._flags[sample_id] = []

        # Don't duplicate flags
        for existing_flag in self._flags[sample_id]:
            if existing_flag['flag'] == flag_type:
                return

        self._flags[sample_id].append(flag_dict)
        self._save_flags()

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
