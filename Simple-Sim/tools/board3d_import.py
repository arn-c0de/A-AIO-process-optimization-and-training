#!/usr/bin/env python3
"""3D-Board-Renderer T0: read one product side from the DeepVision-Web store (read-only) and write the scene.

Output in ``--out`` (default ``outputs/board3d/<product>/<side>/``, git-ignored; holds real designators, keep it
on this machine):

* ``board3d.json``  scene (schema ``board3d/1``, see ``BACKLOG-Sim/board3d-plan.md``)
* ``inventory.json``   counts per package / family / stage, parts without body or height, families T2 must build
* ``overlay_scan.png`` / ``overlay_mounted.jpg``  body boxes on the flatbed scan / the mounted median render

Usage:
    python tools/board3d_import.py --dv ../../DeepVision-Web --product <product> --side top \\
        [--photos <folder of phone photos>] [--no-overlay] [--no-hash] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.paths import OUTPUTS_DIR  # noqa: E402
from simple_sim.board3d import BOARD3D_FILE  # noqa: E402
from simple_sim.board3d.dv_store import SIDES, ProductStore  # noqa: E402
from simple_sim.board3d.importer import build_board3d  # noqa: E402

DEFAULT_DV = Path(__file__).resolve().parents[3] / "DeepVision-Web"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer T0: DeepVision product side -> board3d.json + inventory + overlays")
    ap.add_argument("--dv", type=Path, default=DEFAULT_DV, help="DeepVision-Web checkout (read-only)")
    ap.add_argument("--product", required=True, help="product id in the DeepVision store")
    ap.add_argument("--side", choices=SIDES, default="top")
    ap.add_argument("--photos", type=Path, action="append", default=[], help="photo folder to count as texture source (repeatable)")
    ap.add_argument("--out", type=Path, help="output folder (default outputs/board3d/<product>/<side>)")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of the source images")
    add_json_flag(ap)
    return ap.parse_args(argv)


def summary_lines(inv: dict) -> list[str]:
    lines = [f"{inv['product']} {inv['side']}: {inv['components']} placements, stages {inv['byStage']}, DNP {inv['dnp']}",
             f"  families: {inv['byFamilyStatus']}  shape source: {inv['byShapeSource']}"]
    for fam, e in inv["familiesToBuild"].items():
        lines.append(f"  to build: {fam:14s} {e['parts']:4d} parts  {', '.join(e['packages'])}")
    if inv["withoutBody"]:
        lines.append(f"  without body: {', '.join(m['refdes'] + ' ' + m['package'] for m in inv['withoutBody'])}")
    if inv["withoutHeight"]:
        lines.append(f"  without height (from photos, T3): {', '.join(inv['withoutHeight'])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    store = ProductStore(a.dv.resolve(), a.product)
    out = a.out or OUTPUTS_DIR / "board3d" / a.product / a.side
    board3d, inv = build_board3d(store, a.side, [p.resolve() for p in a.photos], hash_images=not a.no_hash)
    out.mkdir(parents=True, exist_ok=True)
    (out / BOARD3D_FILE).write_text(json.dumps(board3d, indent=1, ensure_ascii=False), encoding="utf-8")
    overlays = {}
    if not a.no_overlay:
        from simple_sim.board3d.overlay import write_overlays
        overlays = write_overlays(board3d, out)
    inv["overlays"] = overlays
    (out / "inventory.json").write_text(json.dumps(inv, indent=1, ensure_ascii=False), encoding="utf-8")
    for line in summary_lines(inv):
        print(line)
    print(f"written: {out}")
    if a.json:
        emit_result("board3d-import", {"out": str(out), "components": inv["components"], "byStage": inv["byStage"],
                                    "byFamilyStatus": inv["byFamilyStatus"], "withoutBody": len(inv["withoutBody"]),
                                    "overlays": {k: v["path"] for k, v in overlays.items()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
