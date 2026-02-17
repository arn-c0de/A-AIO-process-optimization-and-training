from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .profile_registry import ProfileRegistry, RegistrySnapshot
from .state_store import EditorStateStore


@dataclass
class ChoiceLists:
    profile_ids: List[str]
    run_ids: List[str]


class SyncController:
    """Coordinates registry discovery and editor state transitions."""

    def __init__(self, sim_root: Path, store: EditorStateStore):
        self.sim_root = Path(sim_root)
        self.store = store
        self.registry = ProfileRegistry(self.sim_root)
        self.snapshot: Optional[RegistrySnapshot] = None

    def refresh_registry(self) -> ChoiceLists:
        self.snapshot = self.registry.snapshot()
        profile_ids = sorted(self.snapshot.profile_files.keys())
        run_ids = sorted(self.snapshot.run_files.keys())
        return ChoiceLists(profile_ids=profile_ids, run_ids=run_ids)

    def maybe_refresh_registry(self) -> Optional[ChoiceLists]:
        if not self.registry.changed():
            return None
        return self.refresh_registry()

    def load_profile(self, profile_id: str) -> None:
        if not self.snapshot:
            self.refresh_registry()
        if not self.snapshot:
            return
        p = self.snapshot.profile_files.get(profile_id)
        if p is None:
            raise ValueError(f"Unknown profile: {profile_id}")
        self.store.load_profile(p)

    def load_run(self, run_id: str) -> None:
        if not self.snapshot:
            self.refresh_registry()
        if not self.snapshot:
            return
        p = self.snapshot.run_files.get(run_id)
        if p is None:
            raise ValueError(f"Unknown run: {run_id}")
        self.store.load_run(p)

    def matching_runs_for_current_profile(self) -> List[str]:
        if not self.snapshot:
            self.refresh_registry()
        if not self.snapshot:
            return []
        pid = self.store.current_profile_id()
        if not pid:
            return sorted(self.snapshot.run_files.keys())
        return self.registry.matching_run_ids(pid, self.snapshot)

    def save_profile(self) -> None:
        self.store.save("profile")

    def save_run(self) -> None:
        self.store.save("run")
