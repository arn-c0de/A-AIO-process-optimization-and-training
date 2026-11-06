"""AppContext: the services every tab receives (tk-free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gui.core.events import EventBus, Topics
from gui.core.tasks import TaskRunner, UiDispatcher


@dataclass
class AppContext:
    sim_root: Path
    settings: Any = None  # gui.utils.settings_store.SettingsStore
    dispatcher: UiDispatcher = field(default_factory=UiDispatcher)
    bus: EventBus = field(init=False)
    tasks: TaskRunner = field(init=False)

    def __post_init__(self) -> None:
        self.bus = EventBus(post=self.dispatcher.post)
        self.tasks = TaskRunner(self.dispatcher.post, bus=self.bus)

    # Convenience -----------------------------------------------------------------------
    def status(self, text: str, level: str = "info") -> None:
        """Show ``text`` in the status bar (level: info|success|warning|error)."""
        self.bus.publish(Topics.STATUS, text=str(text), level=level)

    def setting(self, key: str, default: Any = None) -> Any:
        if self.settings is None:
            return default
        return self.settings.get(key, default)
