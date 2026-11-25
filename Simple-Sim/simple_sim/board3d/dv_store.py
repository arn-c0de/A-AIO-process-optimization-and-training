"""Read-only access to one product in the DeepVision-Web store (``<dv>/store/products/<product>/``).

Nothing here writes into the store. Files used (all optional except ``pnp.csv``):

=====================================  ====================================================================
file                                   content
=====================================  ====================================================================
``pnp.csv``                            refdes, part, value, -, package, x mm, y mm, rotation deg, bottom YES/NO
``simple-aoi/<side>/parts.json``       every placement of the side with the body polygon in board mm as the
                                       app draws it (package-library resolution), ``mount`` smd/tht, ``dnp``,
                                       pseudonym ``id``; plus ``render.jpg`` (median of good photos of the
                                       mounted board, ``frame`` = x0/y0/pxPerMm/mirrored)
``compare/<side>.json`` + ``.png``     flatbed scan of the SMD-soldered board (600 dpi) with ``boardToImage``
                                       (3x3 homography board mm -> scan pixels) and the fit residual
``package-map.json``                   per-product rotation corrections / shape overrides (recorded, applied
                                       by the app already)
``dnp.json``                           not-placed sites per side (pseudonyms)
=====================================  ====================================================================
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_sim.io_utils import file_sha256, load_json

SIDES = ("top", "bottom")
#: ``pnp.csv`` columns (no header line); the ninth column flags bottom-side placements.
PNP_COLUMNS = ("refdes", "part", "value", "unused", "package", "x_mm", "y_mm", "rotation_deg", "bottom")
#: Values of the ``bottom`` column (upper-cased, stripped) that mean the bottom side.
PNP_BOTTOM_FLAGS = frozenset({"YES", "Y", "TRUE", "1", "BOTTOM"})


@dataclass(frozen=True, slots=True)
class Placement:
    refdes: str
    part: str
    value: str
    package: str
    x_mm: float
    y_mm: float
    rotation_deg: float
    side: str


sha256 = file_sha256  # hex digest of a file (kept for ``board3d.importer``)


def _num(text: str) -> float:
    return float(str(text).strip().replace(",", "."))


def read_pnp(path: Path) -> list[Placement]:
    """Placements of ``pnp.csv`` (:data:`PNP_COLUMNS`, no header; quoted fields may contain commas, numbers may
    use a decimal comma, a UTF-8 BOM is ignored, undecodable bytes are replaced). The bottom flag accepts
    :data:`PNP_BOTTOM_FLAGS`.

    Rows that do not parse (short rows, bad numbers) are skipped; a ValueError listing the first bad rows is
    raised only when no row parses at all (a wrong file, not a few odd lines).
    """
    out: list[Placement] = []
    bad: list[int] = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for i, row in enumerate(csv.reader(f), 1):
            if not row or not any(c.strip() for c in row):
                continue
            try:
                if len(row) < len(PNP_COLUMNS):
                    raise ValueError("short row")
                side = "bottom" if row[8].strip().upper() in PNP_BOTTOM_FLAGS else "top"
                out.append(Placement(row[0].strip(), row[1].strip(), row[2].strip(), row[4].strip(),
                                     _num(row[5]), _num(row[6]), _num(row[7]), side))
            except ValueError:
                bad.append(i)
    if not out:
        raise ValueError(f"{path}: no placement parsed (bad rows {bad[:5]})")
    return out


@dataclass
class ProductStore:
    """Paths of one product; every loader returns None when its file is missing."""
    dv_root: Path
    product: str

    @property
    def dir(self) -> Path:
        return self.dv_root / "store" / "products" / self.product

    def check(self) -> None:
        if not (self.dir / "pnp.csv").is_file():
            raise FileNotFoundError(f"no pnp.csv in {self.dir}")

    def pnp(self) -> list[Placement]:
        return read_pnp(self.dir / "pnp.csv")

    def parts(self, side: str) -> dict[str, Any] | None:
        p = self.dir / "simple-aoi" / side / "parts.json"
        return load_json(p) if p.is_file() else None

    def mounted_render(self, side: str) -> Path | None:
        p = self.dir / "simple-aoi" / side / "render.jpg"
        return p if p.is_file() else None

    def scan(self, side: str) -> tuple[dict[str, Any], Path] | None:
        meta, image = self.dir / "compare" / f"{side}.json", self.dir / "compare" / f"{side}.png"
        if not (meta.is_file() and image.is_file()):
            return None
        return load_json(meta), image

    def package_map(self) -> dict[str, Any] | None:
        p = self.dir / "package-map.json"
        return load_json(p) if p.is_file() else None

    def dnp_ids(self, side: str) -> set[str]:
        p = self.dir / "dnp.json"
        if not p.is_file():
            return set()
        return set(load_json(p).get("sides", {}).get(side, []))

    def mechanics(self, side: str) -> list[str]:
        """SimpleAOI check models of mechanical parts not in the PnP (bolts, bracket, cover, holes)."""
        d = self.dir / "simple-aoi" / side
        names = ("bracket.json", "bolts-model.json", "cover-model.json", "holes-model.json")
        return [n.split(".")[0] for n in names if (d / n).is_file()]


PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


def count_media(folder: Path) -> dict[str, int]:
    """Number of photos / videos below *folder* (counts only, no content)."""
    photos = videos = 0
    for p in folder.rglob("*"):
        s = p.suffix.lower()
        photos += s in PHOTO_SUFFIXES
        videos += s in VIDEO_SUFFIXES
    return {"photos": photos, "videos": videos}
