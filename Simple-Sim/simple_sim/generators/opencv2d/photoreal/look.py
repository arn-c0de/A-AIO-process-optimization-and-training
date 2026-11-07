"""Target looks of the photoreal renderer (``render.look``): which board, parts and camera to imitate.

``legacy`` (default) is the material set of :mod:`.palette` exactly as before; every hook below
returns the legacy constant for it, so ``render.look: legacy`` renders bit-identical images.

``quins`` imitates the QUINS AOI scans of the DeepVision xpa production boards (flatbed line
scanner, ~23.7 px/mm, diffuse ring illumination). Measured on ~20 000 good crops of
``DeepVision-Web/data/store/datasets/xpa-top/quins/crops`` (median BGR per form, centre patch of
the part; see ``docs/guides/PHOTOREAL_RENDERING.md``):

=====================  ====================  =================================================
material               real (BGR median)     legacy render
=====================  ====================  =================================================
solder mask            (67, 87, 49)          palette already matches; mask texture weaker
MLCC ceramic           (86, 108, 121)        (120, 152, 170): too light and yellow
moulded IC / SOT       (64, 68, 65)          (34, 34, 34): real moulding is dark grey
chip resistor          ~(44, 47, 44)         (34, 34, 34)
moulded power ind.     (99, 105, 104)        (34, 34, 34)
chip LED housing       (200, 207, 209)       dark body under the lens
chip terminals         flat, bright tin      ridged "tube" terminals
=====================  ====================  =================================================

The scanner itself (low resolution, very little noise, no glare on the mask, a slight RGB
misregistration along the scan direction) is the camera preset ``quins``
(:data:`.camera.CAMERA_PRESETS`), which ``render.look: quins`` selects unless ``render.camera``
names a preset. The look also switches the joints to the meniscus model with a fixed ring light
(:func:`.meniscus.sample_scanner_look`), since a scanner has no room lamps or flash.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

import numpy as np

from simple_sim.board.placement import LEGACY_PACKING, Packing

from .meniscus import SolderLook, sample_look, sample_scanner_look
from .palette import CAP_BODY, COPPER_UNDER_MASK, END_CAP, LEAD, MASK_PALETTE

DEFAULT_LOOK = "legacy"
MASK_GLOSS_PROBABILITY = 0.5  # legacy share of images with a glossy reflection on the mask (``render.mask_gloss``)
REWORK_PROBABILITY = 0.15     # legacy share of images with rework residue (``render.rework_artifacts``)


def _bgr(*v: float) -> np.ndarray:
    return np.array(v, dtype=np.float32)


def _room_joints(rng: np.random.Generator, roll_deg: float) -> SolderLook:
    """Joint lighting of the default looks: random room lamps / ring light (the roll is irrelevant)."""
    del roll_deg
    return sample_look(rng)


JointSampler = Callable[[np.random.Generator, float], SolderLook]


@dataclass(frozen=True, slots=True)
class Look:
    """Materials of one target domain. Colours are BGR float32."""

    name: str
    mask_palette: tuple[tuple[int, int, int], ...]
    mask_mottle: float            # low-frequency resin-thickness variation of the mask
    mask_weave: float             # glass-fibre weave showing through the mask
    copper_under_mask: np.ndarray  # mask over copper / mask over laminate
    cap_body: np.ndarray          # MLCC ceramic
    end_cap: np.ndarray           # tin-plated chip terminals
    lead: np.ndarray              # gull-wing lead metal
    flat_caps: bool               # flat tin terminal tops with bright edges (else: legacy ridge rule)
    bodies: Mapping[str, np.ndarray] = field(default_factory=dict)  # family -> colour for the dark default body
    # Neighbour layout (row pitch, clearances, families) and density: ``Packing.parts`` is the look's
    # neighbour count (the former ``Look.crowd`` factor, always 1.0, is folded into it).
    packing: Packing = LEGACY_PACKING
    cap_len: Mapping[str, float] = field(default_factory=dict)  # family -> terminal length / body length
    mask_gloss: float = MASK_GLOSS_PROBABILITY  # default of render.mask_gloss
    rework: float = REWORK_PROBABILITY          # default of render.rework_artifacts
    camera_preset: str | None = None  # default of render.camera.preset (None: aoi)
    solder: str | None = None         # joint model this look implies (None: render.solder decides)
    shadow: float = 1.0               # contact-shadow strength factor (diffuse scanner light: < 1)
    silkscreen_skip: float = 0.45     # probability that no silkscreen is printed next to the part
    # Meniscus joint lighting ``(side stream, roll_deg) -> SolderLook`` when the joints use the meniscus
    # model; *roll_deg* is the roll of the board in the final image (fixed scanner lights follow it).
    joint_sampler: JointSampler = _room_joints

    def __post_init__(self) -> None:
        # Shared, immutable registry entries: arrays read-only, dicts behind read-only proxies.
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, np.ndarray):
                        v.setflags(write=False)
                object.__setattr__(self, f.name, MappingProxyType(value))

    def body_colour(self, family: str, base_color: np.ndarray) -> np.ndarray:
        """Body colour of a part whose profile colour is the dark default (legacy: ``max(base, 34)``)."""
        col = self.bodies.get(family, self.bodies.get("*"))
        return np.maximum(base_color, 34.0) if col is None else col


LEGACY_LOOK = Look(name="legacy", mask_palette=MASK_PALETTE, mask_mottle=0.03, mask_weave=0.03,
                   copper_under_mask=COPPER_UNDER_MASK, cap_body=CAP_BODY, end_cap=END_CAP, lead=LEAD, flat_caps=False)

# Neighbour layout of the quins look (DeepVision xpa QUINS scans): rows of passives almost touching,
# mostly MLCCs, 4-10 neighbours in a crop of 3x the part length.
DENSE_PACKING = Packing(array_probability=0.9, pitch=(1.12, 1.3), clearance=0.18, spacing=0.05, parts=10.0,
                        capacitor_share=0.75, tries=24)

_MOULD = _bgr(64, 68, 65)
_RESISTOR = _bgr(44, 47, 44)

QUINS_LOOK = Look(
    name="quins",
    # Per-image mask medians of the scans: p5 (62, 80, 46), p50 (67, 87, 49), p95 (72, 94, 55).
    mask_palette=((62, 80, 46), (64, 84, 47), (66, 86, 49), (68, 88, 50), (70, 91, 52), (72, 94, 55)),
    mask_mottle=0.02, mask_weave=0.012,           # the scans show an almost uniform mask
    copper_under_mask=_bgr(1.12, 1.18, 1.2),     # traces are only faintly visible through the mask
    cap_body=_bgr(86, 108, 121),
    # Metal in the scans is nearly white with a green cast (median BGR of pixels > 200: (209, 228, 214),
    # p90 (253, 253, 253)): the terminal tops saturate, the fillets around them are bright too.
    end_cap=_bgr(200, 216, 206),
    lead=_bgr(192, 208, 198),
    flat_caps=True,
    bodies={"resistor": _RESISTOR, "array_r": _RESISTOR, "inductor": _bgr(58, 61, 59),
            "power_inductor": _bgr(99, 105, 104), "led": _bgr(200, 207, 209), "*": _MOULD},
    cap_len={"capacitor": 0.16, "resistor": 0.16},  # scans: narrow bright terminals, the joint shows around them
    packing=DENSE_PACKING,                        # xpa boards: rows of almost touching passives, mostly MLCCs, 4-10 per 3x crop
    mask_gloss=0.0,                               # diffuse scanner light: no glare on the mask
    rework=0.03,                                  # good production boards: patch wires and flux are rare
    camera_preset="quins",
    solder="meniscus",
    shadow=0.3,                                   # diffuse ring light: hardly any contact shadow
    silkscreen_skip=0.85,                         # the crops show almost no print next to the passives
    joint_sampler=sample_scanner_look,            # two line lights fixed in the final image
)

_LOOKS = {"legacy": LEGACY_LOOK, "quins": QUINS_LOOK}
LOOKS = tuple(_LOOKS)  # registered names, in registration order


def look_name(config: dict[str, Any] | None) -> str:
    """``render.look`` of a render config (default :data:`DEFAULT_LOOK`), validated."""
    name = (config or {}).get("look") or DEFAULT_LOOK
    if name not in _LOOKS:
        raise ValueError(f"render.look must be one of {list(LOOKS)}, got {name!r}")
    return str(name)


def get_look(config: dict[str, Any] | None) -> Look:
    return _LOOKS[look_name(config)]


def camera_config(config: dict[str, Any] | None, look: Look | None = None) -> dict[str, Any] | None:
    """``render.camera`` with the look's preset filled in when the config names none (*look*: the
    already resolved :func:`get_look` of *config*)."""
    cam = (config or {}).get("camera")
    preset = (look or get_look(config)).camera_preset
    if preset is None or (isinstance(cam, dict) and "preset" in cam):
        return cam
    return {"preset": preset, **(cam or {})}


def _probability(config: dict[str, Any] | None, key: str, attr: str, default: float | None,
                 look: Look | None = None) -> float:
    value = (config or {}).get(key)
    if value is None:
        value = getattr(look or get_look(config), attr)
    return float(default if value is None else value)


def mask_gloss(config: dict[str, Any] | None, default: float | None = None, *, look: Look | None = None) -> float:
    """``render.mask_gloss``, else the look's default (:attr:`Look.mask_gloss`). *default* is only used for a
    custom look whose ``mask_gloss`` is ``None`` (kept for callers of the old two-argument form)."""
    return _probability(config, "mask_gloss", "mask_gloss", default, look)


def rework_probability(config: dict[str, Any] | None, default: float | None = None, *, look: Look | None = None) -> float:
    """``render.rework_artifacts``, else the look's default (:attr:`Look.rework`); *default*: see :func:`mask_gloss`."""
    return _probability(config, "rework_artifacts", "rework", default, look)
