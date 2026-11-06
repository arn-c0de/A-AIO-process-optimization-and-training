"""Footprint library: pad geometry per package with pin numbers, roles and fan-out directions.

Pad geometry is computed in **pixels** from the profile's ``nominal`` values, with exactly the
integer arithmetic the renderers always used (``draw._compute_pad_positions`` is a thin wrapper
around :func:`footprint_pads`), so defect geometry, labels and the Blender scene are unchanged.
:func:`part_from_pads` converts a pad list to a :class:`~simple_sim.board.model.Part` in mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simple_sim.board.model import Pad, PadRole, Part, Vec

LEFT: Vec = (-1.0, 0.0)
RIGHT: Vec = (1.0, 0.0)
UP: Vec = (0.0, -1.0)
DOWN: Vec = (0.0, 1.0)
NONE: Vec = (0.0, 0.0)

FAMILIES = ("resistor", "capacitor", "sot", "soic", "qfn", "sot6", "array_r", "array_c", "inductor", "power_inductor",
            "led")
# Two-terminal chip parts (chip_2pad); component_type picks the family.
CHIP_FAMILIES = ("resistor", "capacitor", "inductor", "power_inductor", "led")

# Footprints with two pad columns (left/right of the body) and this many pads per column:
# SOIC-16, SOT-363 / SOT-23-6 and 4-element resistor/capacitor arrays (0804 / 0805 size).
DUAL_ROW_PADS_PER_SIDE = {"soic_16": 8, "sot_6": 3, "chip_array_8": 4, "tssop_14": 7, "tssop_20": 10}


@dataclass(frozen=True, slots=True)
class PadSpec:
    """One pad in pixel space (top-left corner, size) plus its electrical meaning."""
    pin: str
    x: int
    y: int
    w: int
    h: int
    escape: Vec = NONE
    role: PadRole = "signal"

    @property
    def center(self) -> Vec:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def as_dict(self) -> dict[str, int]:
        """The legacy ``{"x", "y", "w", "h"}`` form the drawing code consumes."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def family_of(footprint: str, component_type: str = "") -> str:
    """Part family of a footprint (``component_type`` tells chip resistors and capacitors apart)."""
    if footprint == "sot23":
        return "sot"
    if footprint.startswith("qfn"):
        return "qfn"
    if footprint == "soic_16" or footprint.startswith("tssop"):
        return "soic"
    if footprint == "sot_6":
        return "sot6"
    if footprint == "chip_array_8":
        return "array_c" if "cap" in component_type.lower() else "array_r"
    ct = component_type.lower()
    if "power_inductor" in ct or "molded_inductor" in ct:
        return "power_inductor"
    if "inductor" in ct or "ferrite" in ct:
        return "inductor"
    if "led" in ct:
        return "led"
    return "capacitor" if "cap" in ct else "resistor"


def qfn_pins(footprint: str) -> int:
    """Pin count of ``qfn_<N>`` (at least 4, 32 when unparseable)."""
    try:
        return max(4, int(footprint.split("_", 1)[1]))
    except (IndexError, ValueError):
        return 32


def footprint_pads(footprint: str, nominal: dict[str, float], img_shape: tuple[int, int]) -> list[PadSpec]:
    """Pads of *footprint* centred in an image of *img_shape* (pixels, same rounding as always)."""
    h, w = img_shape
    center_x, center_y = w // 2, h // 2
    pad_width = int(nominal["pad_width"])
    pad_height = int(nominal["pad_height"])
    pad_spacing = int(nominal["pad_spacing"])

    if footprint == "sot23":
        pad_spacing_y = int(nominal["pad_spacing_y"])
        small_h = int(pad_height * 0.8)
        min_gap_px = 10
        effective_spacing_y = max(pad_spacing_y, small_h + min_gap_px)
        left_x = center_x - pad_spacing // 2 - pad_width // 2
        return [
            PadSpec("1", left_x, center_y - effective_spacing_y // 2 - small_h // 2, pad_width, small_h, LEFT),
            PadSpec("2", left_x, center_y + effective_spacing_y // 2 - small_h // 2, pad_width, small_h, LEFT),
            PadSpec("3", center_x + pad_spacing // 2 - pad_width // 2, center_y - pad_height // 2, pad_width, pad_height, RIGHT),
        ]

    if footprint.startswith("qfn"):
        pad_spacing_y = int(nominal["pad_spacing_y"])
        # qfn_<N>: N/4 pads per side on a span of ~70% of the body edge, plus the exposed thermal pad.
        per_side = qfn_pins(footprint) // 4
        body = int(nominal.get("component_length", pad_spacing))
        span = 0.7 * body
        pitch = span / max(1, per_side - 1)
        across = max(2, int(round(pitch * 0.5)))
        left_x = center_x - pad_spacing // 2 - pad_width // 2
        right_x = center_x + pad_spacing // 2 - pad_width // 2
        top_y = center_y - pad_spacing_y // 2 - pad_width // 2
        bot_y = center_y + pad_spacing_y // 2 - pad_width // 2
        pads: list[PadSpec] = []
        for i in range(per_side):
            c = int(round(-span / 2.0 + i * pitch))
            n = 4 * i
            pads.append(PadSpec(str(n + 1), left_x, center_y + c - across // 2, pad_width, across, LEFT))
            pads.append(PadSpec(str(n + 2), right_x, center_y + c - across // 2, pad_width, across, RIGHT))
            pads.append(PadSpec(str(n + 3), center_x + c - across // 2, top_y, across, pad_width, UP))
            pads.append(PadSpec(str(n + 4), center_x + c - across // 2, bot_y, across, pad_width, DOWN))
        thermal = int(round(body * 0.6))
        pads.append(PadSpec("EP", center_x - thermal // 2, center_y - thermal // 2, thermal, thermal, NONE, "thermal"))
        return pads

    if footprint in DUAL_ROW_PADS_PER_SIDE:
        # Pads along y over pad_spacing_y (first to last pad centre); pad_height across, pad_width along.
        n = DUAL_ROW_PADS_PER_SIDE[footprint]
        comp_w = int(nominal.get("component_width", 0))
        pad_spacing_y = int(nominal.get("pad_spacing_y", 0))
        y_span = max(pad_width * (n - 1), pad_spacing_y if pad_spacing_y > 0 else int(comp_w * 0.72))
        pitch = y_span / float(n - 1)
        y0 = center_y - (y_span / 2.0)
        left_x = center_x - pad_spacing // 2 - pad_height // 2
        right_x = center_x + pad_spacing // 2 - pad_height // 2
        pads = []
        for i in range(n):
            y = int(round(y0 + (i * pitch))) - pad_width // 2
            pads.append(PadSpec(str(i + 1), left_x, y, pad_height, pad_width, LEFT))
            pads.append(PadSpec(str(2 * n - i), right_x, y, pad_height, pad_width, RIGHT))
        return pads

    y = center_y - pad_height // 2
    return [
        PadSpec("1", center_x - pad_spacing // 2 - pad_width // 2, y, pad_width, pad_height, LEFT),
        PadSpec("2", center_x + pad_spacing // 2 - pad_width // 2, y, pad_width, pad_height, RIGHT),
    ]


def chip_pads(center: Vec, length: float, width: float, vertical: bool) -> list[PadSpec]:
    """Two-pad chip (neighbour part) at *center* in pixels, pads 45 % of the length at both ends."""
    cx, cy = center
    pad_l, pad_w, off = 0.45 * length, 1.15 * width, 0.45 * length
    pads = []
    for pin, s in (("1", -1.0), ("2", 1.0)):
        if vertical:
            px, py, pw_, ph_ = cx, cy + s * off, pad_w, pad_l
            escape = UP if s < 0 else DOWN
        else:
            px, py, pw_, ph_ = cx + s * off, cy, pad_l, pad_w
            escape = LEFT if s < 0 else RIGHT
        pads.append(PadSpec(pin, int(round(px - pw_ / 2)), int(round(py - ph_ / 2)), int(round(pw_)), int(round(ph_)), escape))
    return pads


def part_from_pads(ref: str, footprint: str, family: str, pads: list[PadSpec], center_px: Vec, body_px: Vec,
                   angle: float, px_to_mm: float, *, is_target: bool = False) -> Part:
    """A :class:`Part` in mm from pixel-space pads (*px_to_mm* = mm per supersampled pixel)."""
    k = float(px_to_mm)
    return Part(
        ref=ref, footprint=footprint, family=family,
        pose=(center_px[0] * k, center_px[1] * k, float(angle)),
        pads=[Pad(pin=p.pin, center=(p.center[0] * k, p.center[1] * k), size=(p.w * k, p.h * k),
                  escape_dir=p.escape, role=p.role) for p in pads],
        body=(body_px[0] * k, body_px[1] * k),
        is_target=is_target,
    )


def pads_px(part: Part, px_to_mm: float) -> list[dict[str, int]]:
    """Legacy pixel pad dicts of a part (inverse of :func:`part_from_pads`, same integers)."""
    k = 1.0 / float(px_to_mm)
    out = []
    for pad in part.pads:
        x0, y0, x1, y1 = pad.bbox
        out.append({"x": int(round(x0 * k)), "y": int(round(y0 * k)), "w": int(round((x1 - x0) * k)), "h": int(round((y1 - y0) * k))})
    return out


def pin_count(part: Part) -> int:
    return sum(1 for p in part.pads if p.role != "thermal")


def thermal_pads(part: Part) -> list[Pad]:
    return [p for p in part.pads if p.role == "thermal"]


def describe(part: Part) -> dict[str, Any]:
    return {"ref": part.ref, "footprint": part.footprint, "family": part.family, "pins": pin_count(part)}
