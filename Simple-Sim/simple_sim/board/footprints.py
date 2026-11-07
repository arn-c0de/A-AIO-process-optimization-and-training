"""Footprint library: pad geometry per package with pin numbers, roles and fan-out directions.

Pad geometry is computed in **pixels** from the profile's ``nominal`` values, with exactly the
integer arithmetic the renderers always used (``opencv2d.geometry.pad_positions`` is a thin wrapper
around :func:`footprint_pads`), so defect geometry, labels and the Blender scene are unchanged.
:func:`part_from_pads` converts a pad list to a :class:`~simple_sim.board.model.Part` in mm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from simple_sim.board.model import Pad, PadRole, Part, Vec

# Footprint of the renderers, jobs and specs when a caller or profile names none: a two-terminal chip.
DEFAULT_FOOTPRINT = "chip_2pad"

LEFT: Vec = (-1.0, 0.0)
RIGHT: Vec = (1.0, 0.0)
UP: Vec = (0.0, -1.0)
DOWN: Vec = (0.0, 1.0)
NONE: Vec = (0.0, 0.0)

# Body drawer kinds of the 2D renderer (``photoreal.parts`` dispatches on them).
KIND_CHIP = "chip"                       # two-terminal chip with end caps (R, C, L, LED)
KIND_POWER_INDUCTOR = "power_inductor"   # moulded block, terminals underneath
KIND_ARRAY = "array"                     # 4-element R/C array (chip_array_8)
KIND_IC = "ic"                           # moulded package: pin-1 mark, gloss, laser marking
BODY_KINDS = (KIND_CHIP, KIND_POWER_INDUCTOR, KIND_ARRAY, KIND_IC)

# Marking rules of the 2D renderer (``photoreal.parts.draw_marking``); ``None``: unmarked.
MARK_RESISTOR_CODE = "resistor_code"     # printed 3-4 character value code
MARK_ARRAY_CODE = "array_code"           # printed 1-3 character code between the terminal rows
MARK_LASER = "laser"                     # laser marking (part number / date / lot, or a short SOT code)
PRINTED_MARKINGS = (MARK_RESISTOR_CODE, MARK_ARRAY_CODE)

# Marking vocabularies (the 2D renderer via ``photoreal.palette`` and the Blender script via the scene).
RESISTOR_CODES = ("103", "472", "1002", "000", "0", "221", "4R7", "100", "330", "10K", "1R0", "5602")
ARRAY_CODES = ("0", "000", "101", "103", "220", "221", "330", "472", "473", "102", "4R7")
MARK_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

# Part builders of the Blender script (``blender/render_batch.py``, D35).
BLENDER_CHIP = "chip"   # body with two end caps
BLENDER_IC = "ic"       # moulded body with pin-1 dimple and laser marking (every other family)


@dataclass(frozen=True, slots=True)
class BlenderPart:
    """How ``blender/render_batch.py`` builds a family's part (baked into the scene as ``part``, D35).

    The Blender model is coarser than the 2D one and differs from it on purpose where the 2D
    numbers model something else; the defaults are the generic moulded IC. Known differences to
    the 2D renderer, kept so the Blender output stays as it was:

    * ``height_ratio`` 0.56 for chips vs ``photoreal.parts.THICKNESS_RATIO`` 0.55 (2D tombstone
      end face: thickness / width).
    * Only resistors and capacitors get end caps; inductors and LEDs (2D: chips with ``cap_len``
      0.25 / 0.16), arrays and power inductors are built as moulded ICs.
    * Gull-wing leads only for ``sot`` / ``soic`` (2D also ``sot6``); lead width
      ``min(a * min(pad w, h), b * max(L, W))`` vs 2D ``0.6 * min(pad extent, 0.3 * L)``.
    * Markings ignore ``render.marking_probability``; arrays, inductors and LEDs carry laser marking
      (2D: array codes or unmarked).
    """

    body: str = BLENDER_IC                  # BLENDER_CHIP | BLENDER_IC
    height_ratio: float = 0.5               # body height / min(length, width), capped by component_height_mm
    standoff: float = 0.01                  # body lift above the board / body height
    body_material: str = "mold"             # "ceramic" (fixed tan) | "mold" (render_3d.materials.body)
    cap_len: float | None = None            # end cap length / body length (BLENDER_CHIP)
    leads: bool = False                     # gull-wing leads from the body edge to the pads
    lead_width: tuple[float, float] = (0.55, 0.15)  # (a, b): lead width = min(a * min(pad w, h), b * max(L, W))
    marking: str | None = MARK_LASER        # MARK_RESISTOR_CODE (one RESISTOR_CODES pick) | MARK_LASER | None
    marking_lines: int = 3                  # laser marking lines of 3-6 MARK_CHARS
    marking_size: tuple[float, float | None] = (0.13, 0.085)  # (a, b): text size = min(a * W, b * L); b None: a * W


BLENDER_GENERIC = BlenderPart()


@dataclass(frozen=True, slots=True)
class PartFamily:
    """Per-family facts the renderers, the board model and the copper rules share (cv2-free, N1).

    The old per-module constants (``CHIP_FAMILIES``, ``FINE_PITCH_FAMILIES``, ``placement.REFDES_PREFIX``,
    ``photoreal.palette`` ``CHIPS`` / ``LEADED`` / ``ARRAY_FAMILIES`` / ``UNMARKED`` / ``CAP_LEN``) are
    derived from :data:`FAMILIES`, in its order.
    """

    name: str
    kind: str                       # body drawer (:data:`BODY_KINDS`)
    refdes: str                     # reference-designator prefix (silkscreen, board model refs)
    marking: str | None             # marking rule (:data:`PRINTED_MARKINGS`, :data:`MARK_LASER`) or None
    chip: bool = False              # two-terminal chip_2pad part (component_type picks the family)
    leaded: bool = False            # gull-wing leads (SOT / SOIC)
    fine_pitch: bool = False        # escape router necks tracks down / fans ground out
    cap_len: float | None = None    # 2D chip terminal (end cap) length / body length
    edge_inset: float = 0.07        # visible body edge slope from above / min(length, width)
    grain_sd: float = 0.06          # top-face surface grain (normal noise sd)
    blender: BlenderPart = BLENDER_GENERIC  # how the Blender script builds the part (scene ``part``)

    @property
    def unmarked(self) -> bool:
        return self.marking is None

    @property
    def printed(self) -> bool:
        """Printed ink marking (resistors) rather than laser marking (ICs)."""
        return self.marking in PRINTED_MARKINGS


# Every part family, in the legacy order of the former ``FAMILIES`` name tuple (iterating the mapping
# yields the names, so ``for fam in FAMILIES`` / ``fam in FAMILIES`` work as before).
FAMILIES: Mapping[str, PartFamily] = MappingProxyType({f.name: f for f in (
    PartFamily("resistor", KIND_CHIP, "R", MARK_RESISTOR_CODE, chip=True, cap_len=0.22, grain_sd=0.025,
               blender=BlenderPart(BLENDER_CHIP, 0.56, cap_len=0.22, marking=MARK_RESISTOR_CODE, marking_lines=1,
                                   marking_size=(0.42, None))),
    PartFamily("capacitor", KIND_CHIP, "C", None, chip=True, cap_len=0.26, grain_sd=0.025,
               blender=BlenderPart(BLENDER_CHIP, 0.56, body_material="ceramic", cap_len=0.26, marking=None)),
    PartFamily("sot", KIND_IC, "Q", MARK_LASER, leaded=True, edge_inset=0.05,
               blender=BlenderPart(height_ratio=0.75, standoff=0.08, leads=True, marking_lines=1, marking_size=(0.32, None))),
    PartFamily("soic", KIND_IC, "U", MARK_LASER, leaded=True, fine_pitch=True, edge_inset=0.035,
               blender=BlenderPart(height_ratio=0.38, standoff=0.08, leads=True)),
    PartFamily("qfn", KIND_IC, "U", MARK_LASER, fine_pitch=True, edge_inset=0.012, blender=BlenderPart(height_ratio=0.18)),
    PartFamily("sot6", KIND_IC, "U", MARK_LASER, leaded=True, fine_pitch=True, edge_inset=0.05),
    PartFamily("array_r", KIND_ARRAY, "RN", MARK_ARRAY_CODE, fine_pitch=True, edge_inset=0.05),
    PartFamily("array_c", KIND_ARRAY, "CN", None, fine_pitch=True, edge_inset=0.05),
    PartFamily("inductor", KIND_CHIP, "L", None, chip=True, cap_len=0.25, grain_sd=0.05),
    PartFamily("power_inductor", KIND_POWER_INDUCTOR, "L", None, chip=True, grain_sd=0.07),
    PartFamily("led", KIND_CHIP, "D", None, chip=True, cap_len=0.16, grain_sd=0.015),
)})
FAMILY_NAMES = tuple(FAMILIES)

# Unknown family names (not produced by family_of; kept drawable): a generic moulded IC, refdes "U".
GENERIC_FAMILY = PartFamily("generic", KIND_IC, "U", MARK_LASER)


def part_family(name: str) -> PartFamily:
    """:class:`PartFamily` of *name* (:data:`GENERIC_FAMILY` for an unknown name)."""
    return FAMILIES.get(name, GENERIC_FAMILY)


def scene_part(name: str) -> dict[str, Any]:
    """The ``part`` entry of the Blender scene (``job_schema.ScenePart``): :attr:`PartFamily.blender` of *name*,
    JSON-ready, with the marking vocabulary, so ``render_batch.py`` builds geometry from the scene alone."""
    b = part_family(name).blender
    marking = None
    if b.marking is not None:
        codes = list(RESISTOR_CODES) if b.marking == MARK_RESISTOR_CODE else []
        marking = {"codes": codes, "chars": MARK_CHARS, "lines": 1 if codes else b.marking_lines, "length": [3, 6],
                   "size": list(b.marking_size)}
    return {"body": b.body, "height_ratio": b.height_ratio, "standoff": b.standoff, "body_material": b.body_material,
            "cap_len": b.cap_len, "leads": b.leads, "lead_width": list(b.lead_width), "marking": marking}


def _names(pred) -> tuple[str, ...]:
    return tuple(n for n, f in FAMILIES.items() if pred(f))


# Two-terminal chip parts (chip_2pad); component_type picks the family.
CHIP_FAMILIES = _names(lambda f: f.chip)
# Families with many small pads at a fine pitch: the escape router necks their tracks down at the
# pad and fans their ground pins out, and the window board keeps their fan-out corridor clear.
FINE_PITCH_FAMILIES = _names(lambda f: f.fine_pitch)


def is_fine_pitch(family: str) -> bool:
    """True for a fine-pitch family (:data:`FINE_PITCH_FAMILIES`)."""
    return family in FINE_PITCH_FAMILIES

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


def profile_footprint(profile: Mapping[str, Any]) -> str:
    """``component.footprint`` of a component profile (:data:`DEFAULT_FOOTPRINT` when unset)."""
    return str((profile.get("component") or {}).get("footprint", DEFAULT_FOOTPRINT))


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
