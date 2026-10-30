"""Persist GUI settings to JSON (best-effort, no external deps)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union
from collections.abc import Iterable, Sequence


@dataclass
class SettingsStore:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)
    _dirty: bool = False
    _after_id: str | None = None
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



# A binding is ``(key, var)`` or ``(key, var, default)``; ``var`` is any Tk
# variable-like object with ``get()``, ``set()`` and ``trace_add()``.
VarBinding = Union[tuple[str, Any], tuple[str, Any, Any]]


def restore_persisted_vars(
    store: SettingsStore | None,
    bindings: Iterable[VarBinding],
    *,
    skip_empty_strings: bool = False,
) -> None:
    """Restore Tk variables from ``store``.

    Uses explicit ``is None`` checks so falsy values such as ``False``/``0``
    (an unchecked checkbox) are restored too. With ``skip_empty_strings`` a
    persisted ``""`` is ignored (keeps the widget default). For 3-tuples the
    default is applied when the key is missing.
    """
    if store is None:
        return
    for binding in bindings:
        key, var = binding[0], binding[1]
        value = store.get(key)
        if value is None and len(binding) > 2:
            value = binding[2]
        if value is None:
            continue
        if skip_empty_strings and value == "":
            continue
        try:
            var.set(value)
        except Exception:
            pass


def wire_vars_autosave(store: SettingsStore | None, bindings: Iterable[VarBinding], widget: Any) -> None:
    """Install ``trace_add("write")`` callbacks that persist each variable (debounced via ``widget``)."""
    if store is None:
        return
    for binding in bindings:
        key, var = binding[0], binding[1]

        def _cb(*_args: Any, _var: Any = var, _key: str = key) -> None:
            try:
                store.set(_key, _var.get())
                store.schedule_save(widget)
            except Exception:
                pass

        var.trace_add("write", _cb)


def bind_persisted_vars(
    store: SettingsStore | None,
    bindings: Sequence[VarBinding],
    widget: Any,
    *,
    skip_empty_strings: bool = False,
) -> None:
    """Restore ``bindings`` from ``store`` and then install autosave traces."""
    restore_persisted_vars(store, bindings, skip_empty_strings=skip_empty_strings)
    wire_vars_autosave(store, bindings, widget)
