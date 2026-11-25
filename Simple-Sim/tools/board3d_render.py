#!/usr/bin/env python3
"""3D-Board-Renderer: render a window (or the whole board) in Blender and compare it with the real flatbed scan.

Reads ``<board3d>/board3d.json`` (from ``board3d-import``), builds the substrate textures once (``<board3d>/substrate/``),
renders with Cycles and writes ``<board3d>/renders/<name>.png``, ``<name>_compare.png`` (real | render | Delta E) and
``<name>.json`` (timing, SSIM, Delta E overall / on parts / on bare board). With ``--defects d.json`` the defect
ops (``simple_sim.board3d.scene``) are applied first; the comparison then shows what changed against the real board.

Usage:
    python tools/board3d_render.py --board3d outputs/board3d/<product>/<side> --window 64 41 20 20 --name probe [--samples 64]
    python tools/board3d_render.py --board3d outputs/board3d/<product>/<side> --full --px-per-mm 12 --name full
    python tools/board3d_render.py --board3d outputs/board3d/<product>/<side> --window 64 41 20 20 --defects d.json --name probe_defects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.io_utils import load_json  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.render import SOURCES, photo_view, prepare_substrate, probe  # noqa: E402
from simple_sim.board3d.scene import apply_defects  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer: render a window in Blender and compare with the real scan")
    ap.add_argument("--board3d", type=Path, required=True, help="folder with board3d.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--window", type=float, nargs=4, metavar=("X0", "Y0", "W", "H"), help="board mm (y up)")
    g.add_argument("--full", action="store_true", help="the whole board")
    g.add_argument("--photo", type=Path, metavar="GEOMETRY_JSON",
                   help="phone view: render from the camera of a real photo (DeepVision run geometry.json next to still.jpg)")
    ap.add_argument("--photo-scale", type=float, default=0.5, help="--photo: render size / photo size")
    ap.add_argument("--photo-source", default="photos", choices=["photos", "mounted"],
                    help="--photo: texture of the mounted board: fused DeepVision stills (default) or SimpleAOI's render")
    ap.add_argument("--raw", action="store_true",
                    help="--photo: plain Cycles image (no colour calibration, no backplate, no lens blur / noise / JPEG)")
    ap.add_argument("--focal-px", type=float, help="--photo: focal length in photo pixels (default: estimate or phone prior)")
    ap.add_argument("--px-per-mm", type=float, help="render resolution (default: the source's)")
    ap.add_argument("--source", default="scan", choices=sorted(SOURCES),
                    help="real image the board3d is built from: scan (SMD flatbed, 24 px/mm) or mounted (median of photos of "
                         "the fully populated board, 16 px/mm)")
    ap.add_argument("--substrate-px-per-mm", type=float, help="default: the source's resolution")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--stages", help="comma list of smd,tht,mount (default: what the source shows)")
    ap.add_argument("--defects", type=Path, help="JSON list of defect ops")
    ap.add_argument("--no-photo", action="store_true", help="synthetic body materials instead of the scan texture")
    ap.add_argument("--light", default="flatbed", choices=["flatbed", "aoi"])
    ap.add_argument("--line-strength", type=float, default=0.0, help="flatbed: scanner line light (W)")
    ap.add_argument("--rebuild-substrate", action="store_true")
    ap.add_argument("--name", default="probe")
    add_json_flag(ap)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    board3d = load_board3d(a.board3d)
    ops = load_json(a.defects) if a.defects else []
    if a.photo:
        scene = apply_defects(board3d, ops) if ops else board3d
        rep = photo_view(scene, a.board3d, a.photo, a.name, scale=a.photo_scale, samples=a.samples, device=a.device, f_px=a.focal_px,
                         calibrate=not a.raw, backplate=not a.raw, camera_fx=not a.raw,
                         source=a.photo_source)
        m, cam = rep["metrics"], rep["camera"]
        print(f"{a.name}: photo view, f {cam['focalPx']:.0f} px ({cam['focalSource']}), tilt {cam['tiltDeg']:.1f} deg, "
              f"{cam['distanceMm']:.0f} mm, {rep['seconds']} s | SSIM board box {m['ssimBoardBox']}, Delta E in board "
              f"{m.get('deltaE_inBoard', {}).get('mean')} (parts {m.get('deltaE_parts', {}).get('mean')}, board {m.get('deltaE_board', {}).get('mean')})")
        mr = rep["metricsRaw"]
        print(f"  raw Cycles: SSIM board box {mr['ssimBoardBox']}, Delta E in board {mr.get('deltaE_inBoard', {}).get('mean')} | "
              f"gain {rep['light'].get('gainRgb')} | camera fx {rep['cameraFx']}")
        print(f"compare: {a.board3d / 'renders' / (a.name + '_compare.png')}")
        if a.json:
            emit_result("board3d-render", {"image": rep["image"], "seconds": rep["seconds"], "metrics": m})
        return 0
    sub = prepare_substrate(board3d, a.board3d, a.substrate_px_per_mm, force=a.rebuild_substrate, source=a.source)
    px = a.px_per_mm or sub["raster"]["pxPerMm"]
    if a.full:
        window = BoardRaster(0, 0, board3d["widthMm"], board3d["heightMm"], px)
    else:
        x0, y0, w, h = a.window
        window = BoardRaster(x0, y0, w, h, px)
    light = {"preset": a.light, "lineStrength": a.line_strength}
    res = probe(board3d, a.board3d, window, a.name, ops=ops, source=a.source, stages=set(a.stages.split(",")) if a.stages else None,
                samples=a.samples, device=a.device, photo=not a.no_photo, light=light)
    metrics = res["metrics"]
    print(f"{a.name}: {res['components']} components, {res['seconds']} s, SSIM {metrics['ssim']}, "
          f"Delta E mean {metrics['deltaE']['mean']} (parts {metrics.get('deltaE_parts', {}).get('mean')}, "
          f"board {metrics.get('deltaE_board', {}).get('mean')})")
    print(f"compare: {a.board3d / 'renders' / (a.name + '_compare.png')}")
    if a.json:
        emit_result("board3d-render", {"image": res["image"], "seconds": res["seconds"], "metrics": metrics})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
