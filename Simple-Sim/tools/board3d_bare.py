#!/usr/bin/env python3
"""3D-Board-Renderer: real SMD lands and a real bare-board texture from photos of the unpopulated board.

Registers every photo to board millimetres (mounting holes -> land ICP -> dense match against the substrate scan),
measures the land under every estimated pad and writes ``<board3d>/bare/``:

* ``lands.json``        lands per refdes and pad index, board frame ``[x, y, w, h, rotDeg]`` (median over photos);
                        ``load_board3d`` puts them on the components, ``footprint_for`` uses them instead of the
                        rule-of-thumb pads
* ``registration.json`` per photo: mapping board mm -> photo px and its residuals
* ``bare_rgb.png``      the best photo resampled to the substrate raster (``substrate/substrate.json``)
* ``lands_mask.png``    bare copper of that photo on the same raster

See ``simple_sim/board3d/barepads.py``.

  python tools/board3d_bare.py --board3d outputs/board3d/<product>/<side> --photo <bare photo or run dir> [--photo ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from simple_sim.board3d import barepads, load_board3d  # noqa: E402
from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.io_utils import load_json  # noqa: E402


def _photo(p: Path) -> Path:
    return p / "photo.jpg" if p.is_dir() else p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board3d", required=True, type=Path, help="board3d folder (board3d.json, substrate/)")
    ap.add_argument("--photo", required=True, action="append", type=Path,
                    help="photo of the bare board side, or a folder with photo.jpg (repeatable)")
    ap.add_argument("--no-scan", action="store_true", help="skip the dense refinement against substrate/scan_rgb.png")
    ap.add_argument("--max-residual", type=float, default=0.2,
                    help="drop a photo whose final registration residual (median, mm) is above this")
    ap.add_argument("--min-lands", type=float, default=0.3,
                    help="drop a photo that measures less than this share of the pads (wrong side, partial view)")
    add_json_flag(ap)
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    board3d = load_board3d(a.board3d)
    board3d["components"] = [{k: v for k, v in c.items() if k != "lands"} for c in board3d["components"]]
    sub_dir = a.board3d / "substrate"
    raster = load_json(sub_dir / "substrate.json")["raster"]
    scan = None if a.no_scan else cv2.imread(str(sub_dir / "scan_rgb.png"), cv2.IMREAD_COLOR)
    out = a.board3d / "bare"
    out.mkdir(parents=True, exist_ok=True)
    regs, per_photo, best = [], [], None
    for p in a.photo:
        path = _photo(p)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path}: not an image", file=sys.stderr)
            continue
        try:
            r = barepads.measure_photo(img, board3d, raster, scan)
        except ValueError as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            regs.append({"photo": str(path), "error": str(e)})
            continue
        ms, used = r.measure, r.usable(a.max_residual, a.min_lands)
        regs.append({"photo": str(path), "model": r.model.to_json(), "stats": r.stats, "lands": ms, "residualMm": r.residual_mm,
                     "used": used})
        print(f"{path.name}: {r.stats['holeHits']} holes, residual {r.residual_mm:.3f} mm, edge bias {ms['edgeBiasMm']:+.3f} mm "
              f"({ms['edgeBiasRings']} rings), lands {ms['measured']}/{ms['pads']}"
              + ("" if used else " - dropped (residual or land share below the limits)"), flush=True)
        if not used:
            continue
        per_photo.append(r.lands)
        if best is None or ms["measured"] > best[0]:
            best = (ms["measured"], r.warped, r.mask)
    if not per_photo:
        print("no photo could be registered", file=sys.stderr)
        if a.json:
            emit_result("board3d-bare", ok=False, error="no photo could be registered")
        return 2
    lands = barepads.fuse_lands(per_photo)
    measured = sum(m is not None for v in lands.values() for m in v)
    total = sum(len(v) for v in lands.values())
    finishes = sorted({r["lands"]["finish"] for r in regs if r.get("used")})
    (out / "lands.json").write_text(json.dumps({"schema": 1, "photos": len(per_photo), "finish": finishes,
                                                "measured": measured, "pads": total,
                                                "lands": lands}, indent=1), encoding="utf-8")
    (out / "registration.json").write_text(json.dumps({"raster": raster, "photos": regs}, indent=1), encoding="utf-8")
    cv2.imwrite(str(out / "bare_rgb.png"), best[1])
    cv2.imwrite(str(out / "lands_mask.png"), best[2])
    secs = round(time.monotonic() - t0, 1)
    print(f"lands {measured}/{total} from {len(per_photo)} photo(s) -> {out}, {secs} s")
    if a.json:
        emit_result("board3d-bare", {"measured": measured, "pads": total, "photos": len(per_photo), "out": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
