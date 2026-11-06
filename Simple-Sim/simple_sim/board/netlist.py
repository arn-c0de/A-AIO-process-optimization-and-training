"""Net assignment per part family instead of a coin toss per pad.

Almost every pin of a real part is connected: chips on both ends, a SOT-23 on all three, ICs on
~85-90 % of their pins with a few NC. Ground is usually reached through the copper pour, supply
through a wide track. Neighbours share nets with the inspected part (decoupling capacitor at
VCC/GND, pull-up on a signal pin, series resistor in a signal line).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from simple_sim.board.model import GND, VCC, Board, Pad, Part

# Share of connected pins per family (the rest are NC).
CONNECTED = {"resistor": 1.0, "capacitor": 1.0, "sot": 1.0, "soic": 0.9, "qfn": 0.85}
# Probability that one of a passive's pads sits on ground / supply.
PASSIVE_GND = {"resistor": 0.25, "capacitor": 0.6}
PASSIVE_VCC = {"resistor": 0.1, "capacitor": 0.35}


def _set(part: Part, index: int, **changes) -> None:
    part.pads[index] = replace(part.pads[index], **changes)


def _assign_passive(part: Part, rng: np.random.Generator, signal: list[str], new_signal) -> None:
    a, b = (0, 1) if rng.random() < 0.5 else (1, 0)
    if rng.random() < PASSIVE_GND[part.family]:
        _set(part, a, net=GND, role="gnd")
    elif rng.random() < PASSIVE_VCC[part.family]:
        _set(part, a, net=VCC, role="power")
    else:
        _set(part, a, net=new_signal())
    # Second pad: a signal, preferably one shared with the inspected part (pull-up, series part).
    if signal and rng.random() < 0.6:
        _set(part, b, net=str(rng.choice(signal)))
    elif part.pads[a].net != VCC and rng.random() < PASSIVE_VCC[part.family]:
        _set(part, b, net=VCC, role="power")
    else:
        _set(part, b, net=new_signal())


def _assign_ic(part: Part, rng: np.random.Generator, new_signal) -> None:
    pins = [i for i, p in enumerate(part.pads) if p.role != "thermal"]
    for i, p in enumerate(part.pads):
        if p.role == "thermal":
            _set(part, i, net=GND)
    order = rng.permutation(pins)
    n_connected = max(2, int(round(CONNECTED[part.family] * len(pins))))
    connected, nc = order[:n_connected], order[n_connected:]
    for i in nc:
        _set(part, int(i), role="nc", net=None)
    gnd, vcc = int(connected[0]), int(connected[1])
    _set(part, gnd, net=GND, role="gnd")
    _set(part, vcc, net=VCC, role="power")
    for i in connected[2:]:
        _set(part, int(i), net=new_signal())


def assign_nets(board: Board, rng: np.random.Generator) -> None:
    """Assign nets to every pad of every part in *board* (in place). Target first, neighbours share."""
    counter = 0

    def new_signal() -> str:
        nonlocal counter
        counter += 1
        return f"N{counter}"

    parts = sorted(board.parts, key=lambda p: not p.is_target)
    target_signals: list[str] = []
    for part in parts:
        if part.family in ("resistor", "capacitor"):
            _assign_passive(part, rng, target_signals, new_signal)
        elif part.family == "sot":
            order = rng.permutation(3)
            _set(part, int(order[0]), net=GND, role="gnd")
            _set(part, int(order[1]), net=new_signal())
            third = VCC if rng.random() < 0.5 else new_signal()
            _set(part, int(order[2]), net=third, role="power" if third == VCC else "signal")
        else:
            _assign_ic(part, rng, new_signal)
        if part.is_target:
            target_signals = [p.net for p in part.pads if p.net and p.net not in (GND, VCC)]


def net_class(net: str | None) -> str:
    return "power" if net in (GND, VCC) else "signal"


def connected_pads(part: Part) -> list[tuple[int, Pad]]:
    return [(i, p) for i, p in enumerate(part.pads) if p.net and p.role not in ("thermal", "nc")]
