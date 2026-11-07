"""``render.board`` options, the keyed random streams and the DRC summary sink (cv2-free).

Everything here is needed to validate a run config or to read a board's DRC result; the board
pipeline itself (:func:`simple_sim.board.pipeline.build_window_board`) draws with cv2 and is only
imported when a board is built.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from simple_sim.board.rules import DEFAULT_PRESET, PRESETS

if TYPE_CHECKING:
    from simple_sim.board.drc import DrcReport

LEGACY = "legacy"
NONE = "none"
# ``roi.mm_per_px`` of the default run configs: the renderer's fallback when a caller passes no
# ``mm_per_px`` in the render config (profiles are in pixels at that scale, see rules.py).
DEFAULT_MM_PER_PX = 0.01
# Names of :data:`simple_sim.board.routing.ROUTERS` (kept here so validation does not load the
# routers; ``tests/test_board.py`` pins the two to each other).
ROUTER_NAMES = ("escape",)
BOARD_ROUTERS = (LEGACY, NONE, *ROUTER_NAMES)
DRC_MODES = ("off", "warn", "strict")
DRC_STRICT_ATTEMPTS = 4
POUR_PROBABILITY = 0.6  # default share of boards with a copper pour (``render.board.pour_probability``)
# Keyed random streams per stage (``default_rng([board_seed, key])``): a change in one stage never
# shifts the draws of another. Keys 1 and 2 are taken by the 3D backend (camera model, camera pose).
STREAM_RULES, STREAM_PLACE, STREAM_NET, STREAM_ZONE, STREAM_ROUTE = 10, 11, 12, 13, 14


@dataclass(frozen=True, slots=True)
class BoardOptions:
    routing: str = LEGACY
    design_rules: str = DEFAULT_PRESET
    pour_probability: float = POUR_PROBABILITY
    drc: str = "off"

    @property
    def legacy(self) -> bool:
        return self.routing == LEGACY


def validate_board_options(raw: Any) -> None:
    """Raise ``ValueError`` unless *raw* is a valid ``render.board`` mapping."""
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ValueError("render.board must be a mapping")
    unknown = set(raw) - {"routing", "design_rules", "pour_probability", "drc"}
    if unknown:
        raise ValueError(f"render.board: unknown keys {sorted(unknown)}")
    if raw.get("routing", LEGACY) not in BOARD_ROUTERS:
        raise ValueError(f"render.board.routing must be one of {list(BOARD_ROUTERS)}")
    if raw.get("design_rules", DEFAULT_PRESET) not in PRESETS:
        raise ValueError(f"render.board.design_rules must be one of {sorted(PRESETS)}")
    p = raw.get("pour_probability", POUR_PROBABILITY)
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("render.board.pour_probability must be a probability in [0, 1]")
    if raw.get("drc", "off") not in DRC_MODES:
        raise ValueError(f"render.board.drc must be one of {list(DRC_MODES)}")


def board_options(render_config: Mapping[str, Any] | None) -> BoardOptions:
    """``render.board`` of a render config as :class:`BoardOptions` (validated)."""
    raw = (render_config or {}).get("board")
    validate_board_options(raw)
    raw = raw or {}
    return BoardOptions(routing=str(raw.get("routing", LEGACY)), design_rules=str(raw.get("design_rules", DEFAULT_PRESET)),
                        pour_probability=float(raw.get("pour_probability", POUR_PROBABILITY)), drc=str(raw.get("drc", "off")))


#: DRC summaries of the boards built in the current context (see :func:`collect_drc`); ``None`` = nobody listens.
_DRC_SINK: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("board_drc_sink", default=None)


def drc_summary(report: DrcReport, options: BoardOptions) -> dict[str, Any]:
    """Compact, JSON-ready DRC result of one board (``render_meta.board_drc`` in ``meta.jsonl``):
    ``ok``, the number of violations, the counts per violation kind and the ``render.board`` options
    (router, design-rule preset, DRC mode) the board was built with. No messages."""
    return {"ok": report.ok, "violations": len(report.violations), "kinds": report.counts(),
            "routing": options.routing, "design_rules": options.design_rules, "drc": options.drc}


@contextmanager
def collect_drc() -> Iterator[list[dict[str, Any]]]:
    """Collect the :func:`drc_summary` of every board :func:`~simple_sim.board.build_window_board` checks while the
    block runs (``drc: warn | strict``; nothing is collected with ``off``, and a strict board that
    fails every attempt raises instead). The sink is per thread / context, so parallel renders never
    mix, and listening changes neither the boards nor any random stream::

        with collect_drc() as reports:
            img = render_record(spec, record)
        meta["board_drc"] = reports[0]   # one board per 2D render window
    """
    sink: list[dict[str, Any]] = []
    token = _DRC_SINK.set(sink)
    try:
        yield sink
    finally:
        _DRC_SINK.reset(token)


def record_drc(report: DrcReport, options: BoardOptions) -> None:
    """Hand *report* to an enclosing :func:`collect_drc` (no-op when nobody listens)."""
    sink = _DRC_SINK.get()
    if sink is not None:
        sink.append(drc_summary(report, options))
