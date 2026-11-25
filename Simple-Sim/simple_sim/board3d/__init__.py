"""3D-Board-Renderer: a whole populated board as one 3D scene description (``board3d.json``).

Plan: ``BACKLOG-Sim/board3d-plan.md``. T0 (this package so far): read the DeepVision store read-only
(:mod:`simple_sim.board3d.dv_store`), resolve every placement to a body box, a 3D family and a mounting stage
(:mod:`simple_sim.board3d.packages`), write the scene (:mod:`simple_sim.board3d.importer`) and draw the bodies onto
the real images for a visual check (:mod:`simple_sim.board3d.overlay`). CLI: ``python -m simple_sim board3d-import``.
"""

from pathlib import Path
from typing import Any

from simple_sim.io_utils import load_json

SCHEMA = "board3d/1"
#: The scene file in a board3d folder (``outputs/board3d/<product>/<side>/``).
BOARD3D_FILE = "board3d.json"


def load_board3d(folder: Path, *, part_catalog: bool = True) -> dict[str, Any]:
    """``<folder>/board3d.json`` (written by ``tools/board3d_import.py``), with the part catalog's body heights applied
    (:func:`simple_sim.board3d.parts.apply_part_catalog`; the result is in ``board3d["partCatalog"]``)."""
    board3d = load_json(Path(folder) / BOARD3D_FILE)
    if part_catalog:
        from simple_sim.board3d.parts import apply_part_catalog
        board3d["partCatalog"] = apply_part_catalog(board3d)
    return board3d


__all__ = ["BOARD3D_FILE", "SCHEMA", "load_board3d"]
