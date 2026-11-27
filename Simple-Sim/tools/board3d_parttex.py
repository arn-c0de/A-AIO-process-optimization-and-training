#!/usr/bin/env python3
"""3D-Board-Renderer: per-part face textures (top + walls) from all registered DeepVision stills.

Writes ``<board3d>/parttex/atlas.png`` (all face tiles) and ``parttex.json`` (per part: pose, size, atlas rects, the
candidate chosen per face group and every candidate's k-fold Delta E against held-out photos). The renderer uses the
atlas for the ``photos`` / ``mounted`` sources (``simple_sim.board3d.render``). See ``simple_sim/board3d/parttex.py``.

  python tools/board3d_parttex.py --board3d outputs/board3d/<product>/<side>
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

from simple_sim.board3d import dv_path, dv_root_for, load_board3d  # noqa: E402
from simple_sim.board3d import cleanlook, parttex  # noqa: E402
from simple_sim.board3d.scene import apply_fit  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402
from simple_sim.io_utils import load_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board3d", required=True, type=Path)
    ap.add_argument("--stages", default="smd,tht,mount")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fit", action="store_true",
                    help="trim body outlines to the photos (parttex/photofit.json), then rebuild the textures on them")
    ap.add_argument("--clean-only", action="store_true",
                    help="only (re)build the clean-look atlas (atlas_clean.png) from the existing atlas.png")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    out = a.board3d / "parttex"
    if a.clean_only:
        rep = load_json(out / "parttex.json")
        clean = cleanlook.clean_atlas(cv2.imread(str(out / "atlas.png"), cv2.IMREAD_COLOR), rep)
        cv2.imwrite(str(out / "atlas_clean.png"), clean)
        print(f"clean atlas {clean.shape[1]}x{clean.shape[0]}, {round(time.monotonic() - t0, 1)} s")
        return 0
    board3d = apply_fit(load_board3d(a.board3d))
    mos_json = a.board3d / "photos" / "mosaic.json"
    if not mos_json.is_file():
        print("no photos/mosaic.json: build the photo mosaic first (3d-board-viewer.sh)", file=sys.stderr)
        return 2
    mos = load_json(mos_json)
    root = dv_root_for(None)
    geos = [g for g in (dv_path(x, root) for x in mos["photos"]) if g.is_file()]
    mosaic = cv2.imread(str(a.board3d / "photos" / "mosaic.png"), cv2.IMREAD_COLOR)
    rm = mos["raster"]
    raster = BoardRaster.from_meta(rm)
    print(f"part textures from {len(geos)} photos ...", flush=True)
    atlas, report = parttex.reconstruct(board3d, geos, set(a.stages.split(",")), mosaic=mosaic, mosaic_raster=raster,
                                        n_folds=a.folds, log=lambda m: print(m, flush=True))
    out = a.board3d / "parttex"
    out.mkdir(parents=True, exist_ok=True)
    if a.fit:
        fits = {r: p["photoFit"] for r, p in report["parts"].items() if p.get("photoFit") and not p["photoFit"].get("absent")}
        absent = sorted(r for r, p in report["parts"].items() if (p.get("photoFit") or {}).get("absent"))
        prev = load_json(out / "photofit.json") if (out / "photofit.json").is_file() else {}
        merged = {**prev.get("parts", {}), **fits}
        (out / "photofit.json").write_text(json.dumps({"parts": merged, "absentInPhotos": absent}, indent=1), encoding="utf-8")
        print(f"photo fit: {len(fits)} outlines trimmed, {len(absent)} parts absent in the photos: {', '.join(absent)}")
        if fits:
            print("rebuilding the textures on the trimmed outlines ...", flush=True)
            board3d = apply_fit(load_board3d(a.board3d))
            atlas, report = parttex.reconstruct(board3d, geos, set(a.stages.split(",")), mosaic=mosaic, mosaic_raster=raster,
                                                n_folds=a.folds, log=lambda m: print(m, flush=True))
            report["absentInPhotos"] = absent
    cv2.imwrite(str(out / "atlas.png"), atlas)
    cv2.imwrite(str(out / "atlas_clean.png"), cleanlook.clean_atlas(atlas, report))
    report["seconds"] = round(time.monotonic() - t0, 1)
    report["summary"] = parttex.score_summary(report)
    (out / "parttex.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{len(report['parts'])} parts, atlas {atlas.shape[1]}x{atlas.shape[0]}, {report['seconds']} s")
    for g, d in report["summary"].items():
        print(f"  {g}: " + ", ".join(f"{v} {e}" for v, e in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
