from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from .yaml_io import dump_yaml, load_yaml, save_yaml

PathKey = Tuple[str, ...]
Listener = Callable[[str, str], None]


@dataclass
class DocumentState:
    path: Optional[Path] = None
    data: Dict[str, Any] | None = None
    text_error: Optional[str] = None
    dirty: bool = False


class EditorStateStore:
    """Single source of truth for profile and run config documents."""

    def __init__(self):
        self.profile = DocumentState(data={})
        self.run = DocumentState(data={})
        self._listeners: List[Listener] = []

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
        self._sync_run_profile_id_if_present()
        self._notify("profile_loaded", "system")

    def load_run(self, path: Path) -> None:
        self.run.path = Path(path)
        self.run.data = load_yaml(path)
        self.run.text_error = None
        self.run.dirty = False
        self._sync_run_profile_id_if_present()
        self._notify("run_loaded", "system")

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
        if target == "profile":
            self._sync_run_profile_id_if_present()
        self._notify(f"{target}_updated", source)

    def update_leaf(self, target: str, path: Sequence[str], value: Any, *, source: str) -> None:
        doc = self._get_doc(target)
        if doc.data is None:
            doc.data = {}
        self._set_path_value(doc.data, tuple(path), value)
        doc.dirty = True
        if target == "profile":
            self._sync_run_profile_id_if_present()
        self._notify(f"{target}_updated", source)

    def save(self, target: str) -> None:
        doc = self._get_doc(target)
        if doc.path is None:
            raise ValueError(f"No file selected for {target}")
        save_yaml(doc.path, doc.data or {})
        doc.dirty = False
        self._notify(f"{target}_saved", "system")

    def _sync_run_profile_id_if_present(self) -> None:
        profile_id = self.current_profile_id()
        if not profile_id:
            return
        if self.run.data is None:
            self.run.data = {}
        run = self.run.data.setdefault("run", {})
        if isinstance(run, dict):
            if str(run.get("component_profile") or "") != profile_id:
                run["component_profile"] = profile_id
                self.run.dirty = True

    def current_profile_id(self) -> str:
        profile = (self.profile.data or {}).get("profile") or {}
        if not isinstance(profile, dict):
            return ""
        return str(profile.get("profile_id") or "").strip()

    def parse_form_input(self, raw_text: str) -> Any:
        raw_text = raw_text.strip()
        if raw_text == "":
            return ""
        parsed = yaml.safe_load(raw_text)
        if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
            return parsed
        return raw_text

    def _set_path_value(self, data: Dict[str, Any], path: PathKey, value: Any) -> None:
        cursor: Any = data
        for key in path[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[path[-1]] = value


def flatten_leaf_paths(data: Dict[str, Any], prefix: PathKey = ()) -> List[Tuple[PathKey, Any]]:
    leaves: List[Tuple[PathKey, Any]] = []
    for key, value in data.items():
        p = prefix + (str(key),)
        if isinstance(value, dict):
            leaves.extend(flatten_leaf_paths(value, p))
        else:
            leaves.append((p, value))
    return leaves
