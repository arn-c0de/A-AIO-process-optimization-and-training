"""Part catalog: body geometry (length, width, height above the board, ...) of real parts, shared by every tool.

One catalog is a JSON file ``{"schema": 1, "parts": [entry, ...]}``. An entry::

    {"id": "<maker>-<series>",                 # unique, stable
     "mpn": ["<mpn>"],                         # manufacturer part numbers (matched without spaces / punctuation)
     "packages": ["<library package name>"],   # exact package names of a CAD library
     "match": ["<regex>"],                     # regexes on the part's value / part / package text (upper case)
     "manufacturer": "<maker>", "series": "<series>", "kind": "wire_to_board", "mount": "tht", "orientation": "vertical",
     "body": {"l": 7.5, "w": 3.8, "h": 6.0},   # mm; h = housing height above the board surface
     "totalHeightMm": 8.1,                     # optional: incl. posts / mated plug
     "material": "plastic",                    # plastic | metal | ceramic | mold (hint for renderers)
     "source": {"url": "...", "kind": "datasheet|distributor|estimate", "note": "..."}}

Catalogs are layered: the **local** ones, which hold the actual part data and stay off GitHub (git-ignored):
``$SIMPLE_SIM_PARTS_DB`` (one or more files, ``os.pathsep``-separated) or ``outputs/parts/catalog.json``, then the
**built-in** ``builtin.json`` next to this module (published: generic parts from public datasheets only, e.g. JST series
and 2.54 mm pin headers). Within a layer an MPN hit wins over a package hit, which wins over a regex hit.

Used by the 3D-Board-Renderer (:func:`simple_sim.board3d.parts.apply_part_catalog`) and meant for the AOI generators.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_sim.io_utils import atomic_write_json, load_json
from simple_sim.paths import OUTPUTS_DIR

SCHEMA = 1
#: environment variable naming the local catalog file(s) (``os.pathsep``-separated, first = the one written to)
CATALOG_ENV = "SIMPLE_SIM_PARTS_DB"
BUILTIN_CATALOG = Path(__file__).with_name("builtin.json")
DEFAULT_LOCAL_CATALOG = OUTPUTS_DIR / "parts" / "catalog.json"
#: how a part was matched, best first
MATCH_RANK = {"mpn": 0, "package": 1, "regex": 2}
SOURCE_KINDS = ("datasheet", "distributor", "estimate")


def norm_mpn(text: str) -> str:
    """Upper case, letters and digits only (``AB12-CD-3(X)`` -> ``AB12CD3X``)."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def local_catalog_paths() -> list[Path]:
    env = os.environ.get(CATALOG_ENV, "").strip()
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    return [DEFAULT_LOCAL_CATALOG]


@dataclass(slots=True)
class PartEntry:
    data: dict[str, Any]
    origin: str                          # "builtin" | path of the local catalog
    _mpn: list[str] = field(default_factory=list)
    _rx: list[re.Pattern[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._mpn = [m for m in (norm_mpn(x) for x in self.data.get("mpn", [])) if len(m) >= 4]
        self._rx = [re.compile(r, re.IGNORECASE) for r in self.data.get("match", [])]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def height_mm(self) -> float | None:
        h = (self.data.get("body") or {}).get("h")
        return None if h is None else float(h)

    @property
    def source_kind(self) -> str:
        return str((self.data.get("source") or {}).get("kind", "estimate"))

    def match(self, texts: list[str], package: str) -> str | None:
        """``"mpn"`` / ``"package"`` / ``"regex"`` or None."""
        flat = [norm_mpn(t) for t in texts if t]
        if any(m in t for m in self._mpn for t in flat):
            return "mpn"
        if package and package in self.data.get("packages", []):
            return "package"
        if any(r.search(t) for r in self._rx for t in [*texts, package] if t):
            return "regex"
        return None


@dataclass(frozen=True, slots=True)
class PartMatch:
    entry: PartEntry
    how: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.entry.id, "how": self.how, "origin": self.entry.origin, "source": self.entry.source_kind,
                **{k: self.entry.data[k] for k in ("body", "kind", "orientation", "material", "totalHeightMm") if k in self.entry.data}}


def validate_entry(e: dict[str, Any]) -> list[str]:
    """Problems of one entry (empty = fine)."""
    errs = []
    if not e.get("id"):
        errs.append("no id")
    if not (e.get("mpn") or e.get("packages") or e.get("match")):
        errs.append("nothing to match (mpn / packages / match)")
    body = e.get("body") or {}
    for k in ("l", "w", "h"):
        v = body.get(k)
        if v is not None and (not isinstance(v, int | float) or v < 0):
            errs.append(f"body.{k} must be a number >= 0")
    if (e.get("source") or {}).get("kind", "estimate") not in SOURCE_KINDS:
        errs.append(f"source.kind must be one of {', '.join(SOURCE_KINDS)}")
    for r in e.get("match", []):
        try:
            re.compile(r)
        except re.error as ex:
            errs.append(f"bad regex {r!r}: {ex}")
    return errs


class PartCatalog:
    """Layered catalog: local files first, then the built-in one."""

    def __init__(self, layers: list[tuple[str, list[dict[str, Any]]]]):
        self.layers = [[PartEntry(e, origin) for e in entries] for origin, entries in layers]

    @classmethod
    def load(cls, local: list[Path] | None = None, *, builtin: bool = True) -> PartCatalog:
        layers = []
        for p in local_catalog_paths() if local is None else local:
            if p.is_file():
                layers.append((str(p), read_catalog_file(p)))
        if builtin and BUILTIN_CATALOG.is_file():
            layers.append(("builtin", read_catalog_file(BUILTIN_CATALOG)))
        return cls(layers)

    @property
    def entries(self) -> list[PartEntry]:
        return [e for layer in self.layers for e in layer]

    def get(self, part_id: str) -> PartEntry | None:
        return next((e for e in self.entries if e.id == part_id), None)

    def lookup(self, *, texts: list[str] | tuple[str, ...] = (), package: str = "") -> PartMatch | None:
        """Best entry for a part described by free *texts* (value, part name, MPN) and its library *package*."""
        for layer in self.layers:
            hits = [(MATCH_RANK[h], i, PartMatch(e, h)) for i, e in enumerate(layer) if (h := e.match(list(texts), package))]
            if hits:
                return min(hits, key=lambda t: t[:2])[2]
        return None


def read_catalog_file(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    parts = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(parts, list):
        raise ValueError(f"{path}: not a part catalog (expected {{'schema': {SCHEMA}, 'parts': [...]}})")
    for e in parts:
        if errs := validate_entry(e):
            raise ValueError(f"{path}: entry {e.get('id')!r}: {'; '.join(errs)}")
    return parts


def upsert_entry(entry: dict[str, Any], path: Path | None = None) -> Path:
    """Add or replace (same ``id``) *entry* in the local catalog *path* (default: the first local one). Returns the path."""
    if errs := validate_entry(entry):
        raise ValueError("; ".join(errs))
    path = path or local_catalog_paths()[0]
    parts = read_catalog_file(path) if path.is_file() else []
    parts = [p for p in parts if p.get("id") != entry["id"]] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"schema": SCHEMA, "parts": sorted(parts, key=lambda p: p["id"])}, indent=1)
    return path


__all__ = [
    "BUILTIN_CATALOG",
    "CATALOG_ENV",
    "DEFAULT_LOCAL_CATALOG",
    "SCHEMA",
    "PartCatalog",
    "PartEntry",
    "PartMatch",
    "local_catalog_paths",
    "norm_mpn",
    "read_catalog_file",
    "upsert_entry",
    "validate_entry",
]
