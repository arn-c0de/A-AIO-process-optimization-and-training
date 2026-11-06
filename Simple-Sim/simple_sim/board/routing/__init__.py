"""Routers: ``escape`` (fan-out, window boards). ``grid`` (A*, full boards) is planned."""

from __future__ import annotations

from simple_sim.board.routing.base import RouteResult, Router, seed_occupancy
from simple_sim.board.routing.escape import EscapeRouter

ROUTERS: dict[str, type] = {"escape": EscapeRouter}

__all__ = ["ROUTERS", "EscapeRouter", "RouteResult", "Router", "seed_occupancy"]
