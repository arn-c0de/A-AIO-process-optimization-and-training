"""Flag management for dataset quality issues."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

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
        self._flags: Dict[str, List[Dict[str, Any]]] = {}  # id -> list of flag dicts
        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from JSONL file."""
        self._flags.clear()

        if not self.flags_path.exists():
            return

        try:
            with open(self.flags_path, 'r') as f:
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
                for sample_id, flag_list in self._flags.items():
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
            'timestamp': datetime.now().isoformat()
        }

        if sample_id not in self._flags:
            self._flags[sample_id] = []

        # Don't duplicate flags
        for existing_flag in self._flags[sample_id]:
            if existing_flag['flag'] == flag_type:
                return

        self._flags[sample_id].append(flag_dict)
        self._save_flags()

    def remove_flag(self, sample_id: str, flag_type: Optional[str] = None) -> None:
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

    def get_flags(self, sample_id: str) -> List[Dict[str, Any]]:
        """Get all flags for a sample.

        Args:
            sample_id: Sample ID

        Returns:
            List of flag dictionaries
        """
        return self._flags.get(sample_id, [])

    def get_all_flagged_samples(self) -> List[str]:
        """Get all sample IDs with flags.

        Returns:
            List of sample IDs
        """
        return list(self._flags.keys())

    def get_samples_by_flag_type(self, flag_type: str) -> List[str]:
        """Get sample IDs with specific flag type.

        Args:
            flag_type: Flag type to filter

        Returns:
            List of sample IDs
        """
        result = []
        for sample_id, flag_list in self._flags.items():
            if any(f['flag'] == flag_type for f in flag_list):
                result.append(sample_id)
        return result

    def has_flag(self, sample_id: str, flag_type: Optional[str] = None) -> bool:
        """Check if sample has a flag.

        Args:
            sample_id: Sample ID
            flag_type: Specific flag type to check, or None to check for any flag

        Returns:
            True if sample has the flag
        """
        if sample_id not in self._flags:
            return False

        if flag_type is None:
            return len(self._flags[sample_id]) > 0

        return any(f['flag'] == flag_type for f in self._flags[sample_id])

    def get_flag_summary(self) -> Dict[str, int]:
        """Get summary of flag counts by type.

        Returns:
            Dictionary mapping flag types to counts
        """
        summary: Dict[str, int] = {}

        for flag_list in self._flags.values():
            for flag_dict in flag_list:
                flag_type = flag_dict['flag']
                summary[flag_type] = summary.get(flag_type, 0) + 1

        return summary
