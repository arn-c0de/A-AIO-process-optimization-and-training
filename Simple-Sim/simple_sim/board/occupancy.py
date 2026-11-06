"""Raster helpers for the board model: a :class:`Grid` maps mm to cells and draws model geometry
into ``uint8`` masks; an :class:`Occupancy` is a grid that also remembers which net owns each cell.

The occupancy is shared by the router (collision checks), the zones (what to keep clear) and the
DRC. Its resolution follows the design rules (half the clearance), not the render resolution, so
checks cost the same at every ``mm_per_px`` and supersampling factor. The pixel rasteriser uses a
plain :class:`Grid` at render resolution with anti-aliased drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from simple_sim.board.model import Vec

BLOCKED = -1
FREE = 0


@dataclass(slots=True)
class Grid:
    origin: Vec                    # mm of cell (0, 0)
    res: float                     # mm per cell
    shape: tuple[int, int]         # (rows, cols)
    aa: bool = False               # anti-aliased drawing (render raster) or hard cells (occupancy)

    @classmethod
    def for_window(cls, bbox: tuple[float, float, float, float], res: float, aa: bool = False) -> Grid:
        x0, y0, x1, y1 = bbox
        cols, rows = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
        return cls(origin=(x0, y0), res=float(res), shape=(rows, cols), aa=aa)

    @property
    def _line(self) -> int:
        return cv2.LINE_AA if self.aa else cv2.LINE_8

    # ---- geometry -> cells ------------------------------------------------------------------
    def cells(self, mm: float) -> int:
        return int(np.ceil(mm / self.res))

    def px(self, mm: float) -> int:
        """Rounded cell count (line widths and radii in the render raster)."""
        return int(round(mm / self.res))

    def to_cell(self, p: Vec) -> tuple[int, int]:
        return int(round((p[0] - self.origin[0]) / self.res)), int(round((p[1] - self.origin[1]) / self.res))

    def scratch(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)

    def rect(self, bbox: tuple[float, float, float, float], inflate: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        (x0, y0), (x1, y1) = self.to_cell((bbox[0] - inflate, bbox[1] - inflate)), self.to_cell((bbox[2] + inflate, bbox[3] + inflate))
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        return mask

    def polyline(self, points: list[Vec], width: float, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        pts = np.array([self.to_cell(p) for p in points], dtype=np.int32)
        w = max(1, self.px(width) if self.aa else self.cells(width))
        cv2.polylines(mask, [pts], False, 255, w, lineType=self._line)
        return mask

    def polygon(self, points: list[Vec], mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        pts = np.array([self.to_cell(p) for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=self._line)
        return mask

    def circle(self, center: Vec, diameter: float, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        r = max(1, self.px(diameter / 2.0) if self.aa else self.cells(diameter / 2.0))
        cv2.circle(mask, self.to_cell(center), r, 255, -1, lineType=self._line)
        return mask

    def inflate(self, mask: np.ndarray, mm: float) -> np.ndarray:
        """*mask* grown by *mm* on every side (clearance)."""
        n = self.cells(mm)
        if n <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
        return cv2.dilate(mask, k)

    def erode(self, mask: np.ndarray, mm: float) -> np.ndarray:
        n = self.cells(mm)
        if n <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
        return cv2.erode(mask, k)


@dataclass(slots=True)
class Occupancy(Grid):
    grid: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int32))   # int32 net ids
    net_ids: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_window(cls, bbox: tuple[float, float, float, float], res: float, aa: bool = False) -> Occupancy:
        g = Grid.for_window(bbox, res)
        return cls(origin=g.origin, res=g.res, shape=g.shape, grid=np.zeros(g.shape, dtype=np.int32))

    def net_id(self, net: str | None) -> int:
        if net is None:
            return BLOCKED
        return self.net_ids.setdefault(net, len(self.net_ids) + 1)

    def conflicts(self, mask: np.ndarray, net: str | None, clearance: float = 0.0, *, ignore_blocked: bool = False) -> bool:
        """True if *mask* (grown by *clearance*) touches copper of another net or (unless
        *ignore_blocked*) a blocked keep-out."""
        area = self.inflate(mask, clearance) > 0
        own = self.net_id(net)
        hit = self.grid[area]
        foreign = (hit != FREE) & (hit != own)
        if ignore_blocked:
            foreign &= hit != BLOCKED
        return bool(np.any(foreign))

    def stamp(self, mask: np.ndarray, net: str | None) -> None:
        """Claim the cells of *mask* for *net* (``None`` = blocked keep-out)."""
        self.grid[mask > 0] = self.net_id(net)

    def clear(self, mask: np.ndarray) -> None:
        """Release the cells of *mask* (undo a stamp)."""
        self.grid[mask > 0] = FREE

    def owner_mask(self, net: str) -> np.ndarray:
        return (self.grid == self.net_id(net)).astype(np.uint8) * 255

    def foreign_mask(self, net: str) -> np.ndarray:
        """Cells owned by any other net (blocked cells excluded)."""
        own = self.net_id(net)
        return ((self.grid != FREE) & (self.grid != own) & (self.grid != BLOCKED)).astype(np.uint8) * 255

    def free_mask(self) -> np.ndarray:
        return (self.grid == FREE).astype(np.uint8) * 255
