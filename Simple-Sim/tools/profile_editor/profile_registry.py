from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simple_sim.file_cache import FileCache, file_signature
from simple_sim.io_utils import load_yaml

_PARSE_FAILED = object()


@dataclass(frozen=True)
class RegistrySnapshot:
    profile_files: dict[str, Path]
    run_files: dict[str, Path]


class ProfileRegistry:
    """Discovers profile/run YAML files and detects changes.

    Parsed YAML is cached per file on its ``stat`` signature (``FileCache``): the 1.4 s poll of
    the profile editor only lists and stats the files and parses nothing while they are unchanged.
    """

    def __init__(self, sim_root: Path):
        self.sim_root = Path(sim_root)
        self.profiles_dir = self.sim_root / "configs" / "profiles"
        self.configs_dir = self.sim_root / "configs"
        self._last_signature: tuple | None = None
        self._last_stat_signature: tuple | None = None
        self._yaml_cache = FileCache(max_entries=1024)

    def _load(self, path: Path) -> object:
        """Parsed YAML of ``path`` (cached until the file changes); ``_PARSE_FAILED`` on errors."""

        def parse(p: Path) -> object:
            try:
                return load_yaml(p) or {}
            except Exception:
                return _PARSE_FAILED

        return self._yaml_cache.get(path, parse, missing=_PARSE_FAILED)

    def _profile_id_from_file(self, path: Path) -> str | None:
        data = self._load(path)
        if data is _PARSE_FAILED:
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

    def _run_id_from_file(self, path: Path) -> str | None:
        data = self._load(path)
        if data is _PARSE_FAILED:
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

    def _profile_files(self) -> list[Path]:
        return sorted(self.profiles_dir.glob("*.yaml")) if self.profiles_dir.exists() else []

    def _run_files(self) -> list[Path]:
        return sorted(self.configs_dir.glob("run_*.yaml")) if self.configs_dir.exists() else []

    def _stat_signature(self) -> tuple:
        """Listing + ``stat`` of every candidate file; no file is opened."""
        return tuple((str(p), file_signature(p)) for p in (*self._profile_files(), *self._run_files()))

    def snapshot(self) -> RegistrySnapshot:
        profiles: dict[str, Path] = {}
        runs: dict[str, Path] = {}

        if self.profiles_dir.exists():
            for p in self._profile_files():
                pid = self._profile_id_from_file(p)
                if pid:
                    profiles[pid] = p

        if self.configs_dir.exists():
            for p in self._run_files():
                rid = self._run_id_from_file(p)
                if rid:
                    runs[rid] = p

        return RegistrySnapshot(profile_files=profiles, run_files=runs)

    def _signature(self, snap: RegistrySnapshot) -> tuple:
        entries: list[tuple] = []
        for kind, files in (("p", snap.profile_files), ("r", snap.run_files)):
            for fid, p in files.items():
                try:
                    st = p.stat()
                except OSError:
                    continue  # file removed between snapshot and stat
                entries.append((kind, fid, str(p), int(st.st_mtime_ns), st.st_size))
        entries.sort()
        return tuple(entries)

    def changed(self) -> bool:
        # Stat-first: unchanged listing + signatures -> nothing to parse (the common poll case).
        stat_sig = self._stat_signature()
        if self._last_signature is not None and stat_sig == self._last_stat_signature:
            return False
        self._last_stat_signature = stat_sig
        snap = self.snapshot()
        sig = self._signature(snap)
        if self._last_signature is None:
            self._last_signature = sig
            return True
        if sig != self._last_signature:
            self._last_signature = sig
            return True
        return False

    def matching_run_ids(self, profile_id: str, snap: RegistrySnapshot | None = None) -> list[str]:
        snap = snap or self.snapshot()
        out: list[str] = []
        for run_id, run_path in snap.run_files.items():
            data = self._load(run_path)
            if data is _PARSE_FAILED or not isinstance(data, dict):
                continue
            run = data.get("run") or {}
            if not isinstance(run, dict):
                continue
            if str(run.get("component_profile") or "").strip() == profile_id:
                out.append(run_id)
        return sorted(out)
