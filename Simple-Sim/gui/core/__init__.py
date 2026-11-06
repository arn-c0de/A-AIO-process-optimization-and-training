"""App-level services shared by all tabs (tk-free): EventBus, TaskRunner, AppContext."""

from gui.core.events import EventBus, Topics
from gui.core.tasks import TaskHandle, TaskRunner, UiDispatcher

__all__ = ["EventBus", "Topics", "TaskHandle", "TaskRunner", "UiDispatcher"]
