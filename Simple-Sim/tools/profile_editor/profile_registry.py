from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class RegistrySnapshot:
    profile_files: Dict[str, Path]
    run_files: Dict[str, Path]


class ProfileRegistry:
    """Discovers profile/run YAML files and detects changes."""

    def __init__(self, sim_root: Path):
        self.sim_root = Path(sim_root)
        self.profiles_dir = self.sim_root / "configs" / "profiles"
        self.configs_dir = self.sim_root / "configs"
        self._last_signature: Optional[tuple] = None

    def _profile_id_from_file(self, path: Path) -> Optional[str]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        profile = data.get("profile") or {}
        if not isinstance(profile, dict):
            return None
        profile_id = str(profile.get("profile_id") or "").strip()
        if not profile_id:
            profile_id = path.stem
        return profile_id

    def _run_id_from_file(self, path: Path) -> Optional[str]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        run = data.get("run") or {}
        if not isinstance(run, dict):
            return None
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            run_id = path.stem
        return run_id

    def snapshot(self) -> RegistrySnapshot:
        profiles: Dict[str, Path] = {}
        runs: Dict[str, Path] = {}

        if self.profiles_dir.exists():
            for p in sorted(self.profiles_dir.glob("*.yaml")):
                pid = self._profile_id_from_file(p)
                if pid:
                    profiles[pid] = p

        if self.configs_dir.exists():
            for p in sorted(self.configs_dir.glob("run_*.yaml")):
                rid = self._run_id_from_file(p)
                if rid:
                    runs[rid] = p

        return RegistrySnapshot(profile_files=profiles, run_files=runs)

    def _signature(self, snap: RegistrySnapshot) -> tuple:
        entries: List[tuple] = []
        for pid, p in snap.profile_files.items():
            st = p.stat()
            entries.append(("p", pid, str(p), int(st.st_mtime_ns), st.st_size))
        for rid, p in snap.run_files.items():
            st = p.stat()
            entries.append(("r", rid, str(p), int(st.st_mtime_ns), st.st_size))
        entries.sort()
        return tuple(entries)

    def changed(self) -> bool:
        snap = self.snapshot()
        sig = self._signature(snap)
        if self._last_signature is None:
            self._last_signature = sig
            return True
        if sig != self._last_signature:
            self._last_signature = sig
            return True
        return False

    def matching_run_ids(self, profile_id: str, snap: Optional[RegistrySnapshot] = None) -> List[str]:
        snap = snap or self.snapshot()
        out: List[str] = []
        for run_id, run_path in snap.run_files.items():
            try:
                data = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            run = data.get("run") or {}
            if not isinstance(run, dict):
                continue
            if str(run.get("component_profile") or "").strip() == profile_id:
                out.append(run_id)
        return sorted(out)
