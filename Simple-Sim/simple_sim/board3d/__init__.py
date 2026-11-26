"""3D-Board-Renderer: a whole populated board as one 3D scene description (``board3d.json``).

Plan: ``BACKLOG-Sim/board3d-plan.md``. T0 (this package so far): read the DeepVision store read-only
(:mod:`simple_sim.board3d.dv_store`), resolve every placement to a body box, a 3D family and a mounting stage
(:mod:`simple_sim.board3d.packages`), write the scene (:mod:`simple_sim.board3d.importer`) and draw the bodies onto
the real images for a visual check (:mod:`simple_sim.board3d.overlay`). CLI: ``python -m simple_sim board3d-import``.
"""

import os
from pathlib import Path
from typing import Any

from simple_sim.io_utils import load_json
from simple_sim.paths import SIM_ROOT

SCHEMA = "board3d/1"
#: DeepVision-Web checkout to read the store from; default: the stored ``sources.dvRoot`` if it exists, else the
#: ``DeepVision-Web`` folder next to this checkout (``<SIM_ROOT>/../../DeepVision-Web``)
DV_ROOT_ENV = "SIMPLE_SIM_DV_ROOT"
#: The scene file in a board3d folder (``outputs/board3d/<product>/<side>/``).
BOARD3D_FILE = "board3d.json"


def load_board3d(folder: Path, *, part_catalog: bool = True) -> dict[str, Any]:
    """``<folder>/board3d.json`` (written by ``tools/board3d_import.py``), with the part catalog's body heights applied
    (:func:`simple_sim.board3d.parts.apply_part_catalog`; the result is in ``board3d["partCatalog"]``), the lands
    measured on bare-board photos (``bare/lands.json``, ``tools/board3d_bare.py``) as ``c["lands"]`` and the photo
    outline fits (``parttex/photofit.json``) as ``c["photoFit"]`` and the mechanics measured next to the data
    (``mechanics.json``: ``brackets``, :func:`simple_sim.board3d.scene.bracket_job`) as ``board3d["brackets"]``."""
    board3d = load_json(Path(folder) / BOARD3D_FILE)
    rebase_dv_root(board3d)
    if part_catalog:
        from simple_sim.board3d.parts import apply_part_catalog
        board3d["partCatalog"] = apply_part_catalog(board3d)
    lf = Path(folder) / "bare" / "lands.json"
    if lf.is_file():
        # SMD lands measured on bare-board photos (tools/board3d_bare.py), board frame, one entry per pad index
        lands = load_json(lf).get("lands", {})
        for c in board3d.get("components", []):
            if c.get("refdes") in lands:
                c["lands"] = lands[c["refdes"]]
    pf = Path(folder) / "parttex" / "photofit.json"
    if pf.is_file():
        # body outlines measured on the photos (simple_sim.board3d.parttex.photo_fit): applied after the scan fit
        fits = load_json(pf).get("parts", {})
        for c in board3d.get("components", []):
            if c.get("refdes") in fits:
                c["photoFit"] = fits[c["refdes"]]
    mf = Path(folder) / "mechanics.json"
    if mf.is_file():
        # product mechanics not in the PnP (front bracket, ...), measured on the photos: stays with the data
        board3d["brackets"] = load_json(mf).get("brackets", [])
    return board3d


def dv_root_for(stored: str | None) -> Path | None:
    """Where the DeepVision store is on this machine (see :data:`DV_ROOT_ENV`)."""
    if env := os.environ.get(DV_ROOT_ENV, "").strip():
        return Path(env).expanduser().resolve()
    if stored and Path(stored).is_dir():
        return Path(stored)
    default = SIM_ROOT.parent.parent / "DeepVision-Web"
    return default.resolve() if default.is_dir() else None


def _rebase(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new + value[len(old):] if value == old or value.startswith(old + "/") else value
    if isinstance(value, list):
        return [_rebase(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rebase(v, old, new) for k, v in value.items()}
    return value


def rebase_dv_root(board3d: dict[str, Any]) -> dict[str, Any]:
    """Point every stored DeepVision path (``sources``) at :func:`dv_root_for` in place, so a board3d folder copied to
    another machine or checkout layout still finds its scans and stills."""
    src = board3d.get("sources") or {}
    old = str(src.get("dvRoot") or "").rstrip("/")
    new = dv_root_for(old or None)
    if old and new is not None and str(new) != old:
        board3d["sources"] = _rebase(src, old, str(new))
    return board3d


__all__ = ["BOARD3D_FILE", "DV_ROOT_ENV", "SCHEMA", "dv_root_for", "load_board3d", "rebase_dv_root"]
