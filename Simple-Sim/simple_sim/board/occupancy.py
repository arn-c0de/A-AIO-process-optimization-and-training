"""Collision state of the board model and its raster helpers.

* :class:`CopperMap` holds every piece of copper (and every blocked keep-out) placed so far as an
  exact :mod:`~simple_sim.board.geometry` shape. The router asks it whether a candidate keeps the
  clearance, the zones ask it where stitching vias fit. Items can be removed again (a fan-out that
  found no route is taken back exactly, nothing else is touched).
* :class:`Grid` maps mm to cells and draws model geometry into ``uint8`` masks. Pours are filled on
  a grid (render resolution, anti-aliased, for the image; ``clearance / 4`` for the DRC); a cell
  is set when its centre lies inside the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from simple_sim.board.geometry import TOUCH, BBox, Shape, gap
from simple_sim.board.model import Vec

# Fixed-point bits for sub-cell accurate cv2 drawing.
_SHIFT = 4
_ONE = 1 << _SHIFT


@dataclass(slots=True)
class Grid:
    origin: Vec                    # mm of the centre of cell (0, 0)
    res: float                     # mm per cell
    shape: tuple[int, int]         # (rows, cols)
    aa: bool = False               # anti-aliased drawing (render raster) or hard cells

    @classmethod
    def for_window(cls, bbox: BBox, res: float, aa: bool = False) -> Grid:
        x0, y0, x1, y1 = bbox
        cols, rows = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
        return cls(origin=(x0, y0), res=float(res), shape=(rows, cols), aa=aa)

    @property
    def _line(self) -> int:
        return cv2.LINE_AA if self.aa else cv2.LINE_8

    # ---- geometry -> cells ------------------------------------------------------------------
    def cells(self, mm: float) -> int:
        return int(np.ceil(mm / self.res - 1e-9))

    def px(self, mm: float) -> int:
        """Rounded cell count (line widths and radii)."""
        return int(round(mm / self.res))

    def to_cell(self, p: Vec) -> tuple[int, int]:
        return int(round((p[0] - self.origin[0]) / self.res)), int(round((p[1] - self.origin[1]) / self.res))

    def _fixed(self, p: Vec) -> tuple[int, int]:
        return (int(round((p[0] - self.origin[0]) / self.res * _ONE)), int(round((p[1] - self.origin[1]) / self.res * _ONE)))

    def scratch(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)

    def rect(self, bbox: BBox, inflate: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
        """Cells whose centre lies in the half-open box ``[x0, x1) x [y0, y1)`` (grown by *inflate*):
        a box of ``w`` cells covers exactly ``w`` cells."""
        mask = self.scratch() if mask is None else mask
        u0, v0 = (bbox[0] - inflate - self.origin[0]) / self.res, (bbox[1] - inflate - self.origin[1]) / self.res
        u1, v1 = (bbox[2] + inflate - self.origin[0]) / self.res, (bbox[3] + inflate - self.origin[1]) / self.res
        i0, j0 = int(np.ceil(u0 - 1e-9)), int(np.ceil(v0 - 1e-9))
        i1, j1 = int(np.ceil(u1 - 1e-9)) - 1, int(np.ceil(v1 - 1e-9)) - 1
        if i1 >= i0 and j1 >= j0:
            cv2.rectangle(mask, (i0, j0), (i1, j1), 255, -1)
        return mask

    def polyline(self, points: list[Vec], width: float, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        pts = np.array([self._fixed(p) for p in points], dtype=np.int32)
        cv2.polylines(mask, [pts], False, 255, max(1, self.px(width)), lineType=self._line, shift=_SHIFT)
        return mask

    def polygon(self, points: list[Vec], mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        pts = np.array([self._fixed(p) for p in points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=self._line, shift=_SHIFT)
        return mask

    def circle(self, center: Vec, diameter: float, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.scratch() if mask is None else mask
        r = max(1, int(round(diameter / 2.0 / self.res * _ONE)))
        cv2.circle(mask, self._fixed(center), r, 255, -1, lineType=self._line, shift=_SHIFT)
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
class Item:
    shape: Shape
    net: str | None                # None: NC pad (copper of no net) or a blocked keep-out
    group: int | None = None       # part index: a part's pads, fan-out stubs and thermal vias
    blocked: bool = False          # keep-out (not copper): nothing foreign may enter it


@dataclass(slots=True)
class CopperMap:
    """Exact collision map (see module docstring)."""
    bbox: BBox
    items: list[Item | None] = field(default_factory=list)
    groups: dict[str, int] = field(default_factory=dict)
    _boxes: list[BBox] = field(default_factory=list)

    def group(self, ref: str) -> int:
        return self.groups.setdefault(ref, len(self.groups) + 1)

    def add(self, shape: Shape, net: str | None, *, group: int | None = None, blocked: bool = False) -> int:
        self.items.append(Item(shape, net, group, blocked))
        self._boxes.append(shape.bbox)
        return len(self.items) - 1

    def remove(self, token: int) -> None:
        self.items[token] = None

    def conflicts(self, shape: Shape, net: str | None, clearance: float, *, group: int | None = None,
                  group_clearance: float = 0.0) -> bool:
        """True if *shape* of *net* comes closer than *clearance* to copper of another net or to a
        blocked keep-out. Copper of the same *group* (the own part's pin row) only needs
        *group_clearance*, and the own part's keep-out does not count."""
        reach = max(clearance, group_clearance)
        sb = shape.bbox
        if self._boxes:
            b = np.asarray(self._boxes)
            near = np.flatnonzero((b[:, 0] <= sb[2] + reach) & (sb[0] <= b[:, 2] + reach)
                                  & (b[:, 1] <= sb[3] + reach) & (sb[1] <= b[:, 3] + reach))
        else:
            near = []
        for i in near:
            it = self.items[int(i)]
            if it is None:
                continue
            same_group = group is not None and it.group == group
            if it.blocked:
                if same_group:
                    continue
                need = clearance
            else:
                if net is not None and it.net == net:
                    continue
                need = group_clearance if same_group else clearance
            g = gap(shape, it.shape)
            if g < need or g <= TOUCH:
                return True
        return False

    def copper(self, net: str | None = None) -> list[Item]:
        """Live copper items (of *net*, or of every net)."""
        return [it for it in self.items if it is not None and not it.blocked and (net is None or it.net == net)]
