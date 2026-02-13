"""Persist GUI settings to JSON (best-effort, no external deps)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SettingsStore:
    path: Path
    data: Dict[str, Any] = field(default_factory=dict)
    _dirty: bool = False
    _after_id: Optional[str] = None
    _after_widget: Any = None  # tk widget with after/after_cancel

    def load(self) -> None:
        try:
            if not self.path.exists():
                self.data = {}
                return
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = obj if isinstance(obj, dict) else {}
        except Exception:
            self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        # Keep JSON-friendly primitives only.
        if isinstance(value, (str, int, float, bool)) or value is None:
            self.data[key] = value
        else:
            self.data[key] = str(value)
        self._dirty = True

    def schedule_save(self, widget: Any, delay_ms: int = 600) -> None:
        """Debounced save using a Tk widget's after()."""
        try:
            if self._after_id is not None and self._after_widget is not None:
                try:
                    self._after_widget.after_cancel(self._after_id)
                except Exception:
                    pass
            self._after_widget = widget
            self._after_id = widget.after(int(delay_ms), self.save)
        except Exception:
            # If after() not available, fall back to immediate save.
            self.save()

    def save(self) -> None:
        self._after_id = None
        self._after_widget = None
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False
        except Exception:
            # best-effort
            pass

