"""Design rules: copper widths and clearances in mm, with presets and per-board sampling.

The profiles define part geometry in pixels at ``roi.mm_per_px`` (0.01 mm/px in the default run
configs, which is *not* the physical scale of the packages), so the presets are tuned so that the
copper looks right next to those parts: at 0.01 mm/px a 0603 pad is ~0.3 mm wide and a signal
track 0.04-0.1 mm. Every value is a (min, max) range; :func:`sample_rules` draws one rule set per
board so that layouts stay varied.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

import numpy as np

AngleMode = Literal["45", "90", "any"]
Range = tuple[float, float]


@dataclass(frozen=True, slots=True)
class DesignRules:
    track_width: Mapping[str, float]   # net class -> mm ("signal", "power")
    clearance: float                   # mm copper-copper between different nets
    neck_down: float                   # mm track width right at a fine-pitch pad
    fanout_stub: float                 # mm of straight exit before the first bend
    via_drill: float
    via_diameter: float
    zone_clearance: float              # mm pour to foreign copper
    thermal_gap: float                 # mm pour to a pad of the pour's own net
    thermal_spoke: float               # mm width of the relief spokes
    stitch_pitch: float | None         # mm between stitching vias in a pour, None = none
    angle_mode: AngleMode = "45"

    def width(self, net_class: str) -> float:
        return float(self.track_width.get(net_class, self.track_width["signal"]))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: (dict(getattr(self, f.name)) if f.name == "track_width" else getattr(self, f.name))
                for f in fields(self)}


@dataclass(frozen=True, slots=True)
class RulesPreset:
    """Ranges (mm) from which :func:`sample_rules` draws a :class:`DesignRules` per board."""
    signal_width: Range
    power_width: Range
    clearance: Range
    neck_down: Range
    fanout_stub: Range
    via_drill: Range
    via_ring: Range                    # annular ring width added to the drill on each side
    zone_clearance: Range
    thermal_gap: Range
    thermal_spoke: Range
    stitch_pitch: Range | None
    angle_mode: AngleMode = "45"


# Scale note: mm at the profile scale (see module docstring), not physical package sizes.
PRESETS: dict[str, RulesPreset] = {
    "consumer_dense": RulesPreset(
        signal_width=(0.04, 0.09), power_width=(0.10, 0.18), clearance=(0.03, 0.05), neck_down=(0.03, 0.05),
        fanout_stub=(0.06, 0.16), via_drill=(0.05, 0.07), via_ring=(0.03, 0.05), zone_clearance=(0.04, 0.07),
        thermal_gap=(0.03, 0.05), thermal_spoke=(0.04, 0.07), stitch_pitch=(0.45, 0.9),
    ),
    "industrial": RulesPreset(
        signal_width=(0.07, 0.14), power_width=(0.16, 0.30), clearance=(0.06, 0.10), neck_down=(0.05, 0.08),
        fanout_stub=(0.10, 0.25), via_drill=(0.07, 0.10), via_ring=(0.05, 0.08), zone_clearance=(0.08, 0.14),
        thermal_gap=(0.06, 0.09), thermal_spoke=(0.07, 0.12), stitch_pitch=(0.8, 1.6),
    ),
    "hdi": RulesPreset(
        signal_width=(0.03, 0.06), power_width=(0.08, 0.14), clearance=(0.02, 0.035), neck_down=(0.025, 0.04),
        fanout_stub=(0.04, 0.10), via_drill=(0.03, 0.05), via_ring=(0.02, 0.035), zone_clearance=(0.03, 0.05),
        thermal_gap=(0.025, 0.04), thermal_spoke=(0.03, 0.05), stitch_pitch=(0.3, 0.6),
    ),
}
DEFAULT_PRESET = "consumer_dense"


def _draw(rng: np.random.Generator, r: Range) -> float:
    return float(rng.uniform(r[0], r[1]))


def sample_rules(preset: str | RulesPreset, rng: np.random.Generator, scale: float = 1.0) -> DesignRules:
    """One rule set drawn from *preset*; *scale* multiplies every length (e.g. to follow the part size)."""
    p = PRESETS[preset] if isinstance(preset, str) else preset
    drill = _draw(rng, p.via_drill) * scale
    return DesignRules(
        track_width={"signal": _draw(rng, p.signal_width) * scale, "power": _draw(rng, p.power_width) * scale},
        clearance=_draw(rng, p.clearance) * scale,
        neck_down=_draw(rng, p.neck_down) * scale,
        fanout_stub=_draw(rng, p.fanout_stub) * scale,
        via_drill=drill,
        via_diameter=drill + 2.0 * _draw(rng, p.via_ring) * scale,
        zone_clearance=_draw(rng, p.zone_clearance) * scale,
        thermal_gap=_draw(rng, p.thermal_gap) * scale,
        thermal_spoke=_draw(rng, p.thermal_spoke) * scale,
        stitch_pitch=None if p.stitch_pitch is None else _draw(rng, p.stitch_pitch) * scale,
        angle_mode=p.angle_mode,
    )


def scaled(rules: DesignRules, k: float) -> DesignRules:
    """*rules* with every length multiplied by *k*."""
    return DesignRules(
        track_width={c: w * k for c, w in rules.track_width.items()}, clearance=rules.clearance * k,
        neck_down=rules.neck_down * k, fanout_stub=rules.fanout_stub * k, via_drill=rules.via_drill * k,
        via_diameter=rules.via_diameter * k, zone_clearance=rules.zone_clearance * k, thermal_gap=rules.thermal_gap * k,
        thermal_spoke=rules.thermal_spoke * k, stitch_pitch=None if rules.stitch_pitch is None else rules.stitch_pitch * k,
        angle_mode=rules.angle_mode)


# A pin row must hold a necked-down track plus a clearance per pitch (with this much slack).
PITCH_FILL = 0.85


def fit_to_pitch(rules: DesignRules, pitch: float) -> DesignRules:
    """*rules* shrunk (all lengths alike) until a necked-down track plus a clearance fit into
    *pitch* (mm, the finest pin pitch of the inspected part): a board carrying a fine-pitch part
    is laid out with rules fine enough for it."""
    need = rules.neck_down + rules.clearance
    if not np.isfinite(pitch) or need <= PITCH_FILL * pitch:
        return rules
    return scaled(rules, PITCH_FILL * pitch / need)
