from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

import yaml

from .yaml_io import dump_yaml, load_yaml, save_yaml

PathKey = tuple[str, ...]
Listener = Callable[[str, str], None]


@dataclass
class DocumentState:
    path: Path | None = None
    data: dict[str, Any] | None = None
    text_error: str | None = None
    dirty: bool = False


class EditorStateStore:
    """Single source of truth for profile and run config documents."""

    def __init__(self):
        self.profile = DocumentState(data={})
        self.run = DocumentState(data={})
        self._listeners: list[Listener] = []

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _notify(self, event: str, source: str) -> None:
        for listener in self._listeners:
            listener(event, source)

    def load_profile(self, path: Path) -> None:
        self.profile.path = Path(path)
        self.profile.data = load_yaml(path)
        self.profile.text_error = None
        self.profile.dirty = False
        self._notify("profile_loaded", "system")

    def load_run(self, path: Path) -> None:
        self.run.path = Path(path)
        self.run.data = load_yaml(path)
        self.run.text_error = None
        self.run.dirty = False
        self._notify("run_loaded", "system")

    def clear_run(self) -> None:
        """Unload the run document (e.g. no run config references the active profile)."""
        self.run = DocumentState(data={})
        self._notify("run_loaded", "system")

    def is_dirty(self, target: str) -> bool:
        return bool(self._get_doc(target).dirty)

    def _get_doc(self, target: str) -> DocumentState:
        if target == "profile":
            return self.profile
        if target == "run":
            return self.run
        raise ValueError(f"Unknown target: {target}")

    def get_text(self, target: str) -> str:
        doc = self._get_doc(target)
        return dump_yaml(doc.data or {})

    def set_text(self, target: str, text: str, *, source: str) -> None:
        doc = self._get_doc(target)
        try:
            parsed = yaml.safe_load(text) if text.strip() else {}
            if parsed is None:
                parsed = {}
            if not isinstance(parsed, dict):
                raise ValueError("Top-level YAML must be a mapping")
        except Exception as exc:
            doc.text_error = str(exc)
            self._notify("yaml_error", source)
            return

        doc.data = parsed
        doc.text_error = None
        doc.dirty = True
        self._notify(f"{target}_updated", source)

    def update_leaf(self, target: str, path: Sequence[str], value: Any, *, source: str) -> None:
        doc = self._get_doc(target)
        if doc.data is None:
            doc.data = {}
        self._set_path_value(doc.data, tuple(path), value)
        doc.dirty = True
        self._notify(f"{target}_updated", source)

    def save(self, target: str) -> None:
        doc = self._get_doc(target)
        if doc.path is None:
            raise ValueError(f"No file selected for {target}")
        save_yaml(doc.path, doc.data or {})
        doc.dirty = False
        self._notify(f"{target}_saved", "system")

    def run_profile_id(self) -> str:
        run = (self.run.data or {}).get("run") or {}
        if not isinstance(run, dict):
            return ""
        return str(run.get("component_profile") or "").strip()

    def run_profile_mismatch(self) -> bool:
        """True if a run is loaded whose component_profile differs from the active profile."""
        if self.run.path is None:
            return False
        pid = self.current_profile_id()
        return bool(pid) and self.run_profile_id() != pid

    def link_run_to_profile(self) -> bool:
        """Explicitly point the loaded run's component_profile at the active profile.

        Returns True if the run document was changed (and marked dirty).
        """
        profile_id = self.current_profile_id()
        if not profile_id or self.run.path is None:
            return False
        if self.run.data is None:
            self.run.data = {}
        run = self.run.data.setdefault("run", {})
        if not isinstance(run, dict) or str(run.get("component_profile") or "") == profile_id:
            return False
        run["component_profile"] = profile_id
        self.run.dirty = True
        self._notify("run_updated", "system")
        return True

    def current_profile_id(self) -> str:
        profile = (self.profile.data or {}).get("profile") or {}
        if not isinstance(profile, dict):
            return ""
        return str(profile.get("profile_id") or "").strip()

    def get_leaf(self, target: str, path: Sequence[str]) -> tuple[bool, Any]:
        """Return (exists, value) for a leaf path in a document."""
        cursor: Any = self._get_doc(target).data or {}
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                return False, None
            cursor = cursor[key]
        return True, cursor

    def commit_form_text(self, target: str, path: Sequence[str], raw_text: str, *, source: str) -> bool:
        """Apply a form field's text to the document.

        No-op (returns False) if the text equals the current value's form text,
        so focus changes never re-parse or dirty a document. If the current value
        is a string it stays a string (no YAML coercion of e.g. "0603" -> 387 or
        "yes" -> True); other values are parsed as YAML scalars/flow collections.
        """
        path = tuple(path)
        exists, current = self.get_leaf(target, path)
        if exists and to_form_text(current) == raw_text:
            return False
        if exists and isinstance(current, str):
            value: Any = raw_text
        else:
            value = self.parse_form_input(raw_text)
        if exists and type(value) is type(current) and value == current:
            return False
        self.update_leaf(target, path, value, source=source)
        return True

    def parse_form_input(self, raw_text: str) -> Any:
        raw_text = raw_text.strip()
        if raw_text == "":
            return ""
        parsed = yaml.safe_load(raw_text)
        if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
            return parsed
        return raw_text

    def _set_path_value(self, data: dict[str, Any], path: PathKey, value: Any) -> None:
        cursor: Any = data
        for key in path[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[path[-1]] = value


def to_form_text(value: Any) -> str:
    """Text shown in a form entry for a YAML leaf value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return str(value)


def flatten_leaf_paths(data: dict[str, Any], prefix: PathKey = ()) -> list[tuple[PathKey, Any]]:
    leaves: list[tuple[PathKey, Any]] = []
    for key, value in data.items():
        p = prefix + (str(key),)
        if isinstance(value, dict):
            leaves.extend(flatten_leaf_paths(value, p))
        else:
            leaves.append((p, value))
    return leaves
