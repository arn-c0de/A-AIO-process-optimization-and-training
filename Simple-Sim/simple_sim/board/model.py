"""Board model: parts, pads, tracks, vias and zones as plain data in millimetres.

Coordinates are board coordinates in mm, x to the right and y down (like the image). In window
mode (one inspected part per rendered image) the board is the render window with its origin in
the top-left corner. Nothing here depends on OpenCV, supersampling or ``mm_per_px``; the
rasteriser (:mod:`simple_sim.board.raster`) turns the model into pixel masks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from simple_sim.board.rules import DesignRules

Vec = tuple[float, float]
PadRole = Literal["signal", "power", "gnd", "thermal", "nc"]
PadShape = Literal["rect", "roundrect", "circle"]
NetClass = Literal["signal", "power"]

GND = "GND"
VCC = "VCC"
TOP = "top"
BOTTOM = "bottom"


@dataclass(frozen=True, slots=True)
class Pad:
    pin: str                       # "1", "2", "EP" (exposed/thermal pad)
    center: Vec                    # mm, board coordinates
    size: Vec                      # mm, (width along x, height along y)
    rotation: float = 0.0          # degrees
    shape: PadShape = "rect"
    escape_dir: Vec = (0.0, 0.0)   # unit normal of the pad edge away from the body; (0, 0) = no fan-out
    role: PadRole = "signal"
    net: str | None = None

    @property
    def half(self) -> Vec:
        return self.size[0] / 2.0, self.size[1] / 2.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned (x0, y0, x1, y1) in mm (rotation is 0 or 90 degrees for every footprint here)."""
        hx, hy = self.half
        return self.center[0] - hx, self.center[1] - hy, self.center[0] + hx, self.center[1] + hy


@dataclass(slots=True)
class Part:
    ref: str                       # "R12", "U3"
    footprint: str                 # "chip_2pad", "sot23", "soic_16", "qfn_32"
    family: str                    # resistor | capacitor | sot | soic | qfn
    pose: tuple[float, float, float]   # x, y (mm), angle (degrees)
    pads: list[Pad]
    body: Vec                      # length x width in mm (keep-out)
    is_target: bool = False        # the inspected part

    @property
    def center(self) -> Vec:
        return self.pose[0], self.pose[1]

    @property
    def keepout(self) -> tuple[float, float, float, float]:
        """Bounding box of body and pads in mm: nothing of another net may be routed through it."""
        bl, bw = self.body
        if round(self.pose[2]) % 180 == 90:
            bl, bw = bw, bl
        xs = [self.center[0] - bl / 2.0, self.center[0] + bl / 2.0]
        ys = [self.center[1] - bw / 2.0, self.center[1] + bw / 2.0]
        for pad in self.pads:
            x0, y0, x1, y1 = pad.bbox
            xs += [x0, x1]
            ys += [y0, y1]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(slots=True)
class Track:
    net: str
    layer: str
    points: list[Vec]              # polyline in mm
    width: float                   # mm


@dataclass(slots=True)
class Via:
    net: str
    center: Vec
    drill: float
    diameter: float
    layers: tuple[str, str] = (TOP, BOTTOM)
    tented: bool = True


@dataclass(frozen=True, slots=True)
class ThermalRelief:
    gap: float                     # mm between pad and pour
    spoke_width: float             # mm
    spokes: int = 4


@dataclass(slots=True)
class Zone:
    net: str
    layer: str
    outline: list[Vec]             # polygon in mm
    clearance: float               # mm to copper of other nets
    thermal: ThermalRelief | None = None


@dataclass(frozen=True, slots=True)
class Port:
    """Point where a net leaves the window (window mode only)."""
    net: str
    point: Vec
    layer: str = TOP


@dataclass(slots=True)
class Board:
    outline: list[Vec]             # render window or real board contour, mm
    layers: tuple[str, ...]
    rules: DesignRules
    parts: list[Part] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.outline]
        ys = [p[1] for p in self.outline]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def size(self) -> Vec:
        x0, y0, x1, y1 = self.bbox
        return x1 - x0, y1 - y0

    @property
    def target(self) -> Part | None:
        return next((p for p in self.parts if p.is_target), None)

    def nets(self) -> set[str]:
        nets = {t.net for t in self.tracks} | {v.net for v in self.vias} | {z.net for z in self.zones}
        nets |= {pad.net for part in self.parts for pad in part.pads if pad.net}
        return nets

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (``render_meta.board`` or a debug file)."""
        return asdict(self)


def point_in_polygon(p: Vec, poly: list[Vec]) -> bool:
    """Even-odd test of *p* against the closed polygon *poly*."""
    inside = False
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < (x1 - x0) * (p[1] - y0) / (y1 - y0 + 1e-12) + x0:
            inside = not inside
    return inside


def window_board(size_mm: Vec, rules: DesignRules, layers: tuple[str, ...] = (TOP,)) -> Board:
    """Empty board covering a render window of *size_mm* with its origin at the top-left corner."""
    w, h = size_mm
    return Board(outline=[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)], layers=layers, rules=rules)
