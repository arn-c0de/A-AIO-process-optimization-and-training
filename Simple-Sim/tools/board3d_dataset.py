#!/usr/bin/env python3
"""3D-Board-Renderer T7: labelled defect crops (OK / MISALIGNED / MISSING / TOMBSTONE / EXCESS_SOLDER) from the 3D board scene.

Each sheet renders one random window with several far-apart parts changed at once (``simple_sim.board3d.dataset``);
every chosen part gives one crop of the render and one of the real image at the same place.

Output (``--out``, default ``<board3d>/dataset/<name>``):
    images/<id>.png   render crop        real/<id>.png   real crop of the same part (OK in reality)
    labels.jsonl      one line per crop  summary.json    counts per class / family, timing

Usage:
    python tools/board3d_dataset.py --board3d outputs/board3d/<product>/<side> --sheets 10 --per-sheet 8 [--source scan] [--samples 64]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.io_utils import load_json  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.camera import board_to_source  # noqa: E402
from simple_sim.board3d.dataset import crop_part, crop_photo, plan_sheet  # noqa: E402
from simple_sim.board3d.render import SOURCES, crop_real, photo_view, prepare_substrate, render_window, still_source  # noqa: E402
from simple_sim.board3d.scene import apply_defects, sample_variants  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402


def photo_geometries(spec: Path, board3d: dict) -> list[Path]:
    """geometry.json files of *board3d*'s product and side: listed in a text file, or found under a directory."""
    if spec.is_file():
        paths = [Path(x.strip()) for x in spec.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        paths = sorted(spec.rglob("geometry.json"))
    product = (board3d.get("sources") or {}).get("product") or board3d.get("product")
    side = board3d.get("side")
    out = []
    for p in paths:
        try:
            g = load_json(p)
        except (OSError, ValueError):
            continue
        if (product is None or g.get("product") in (None, product)) and (side is None or g.get("side") in (None, side)) \
                and g.get("boardToSource") and (p.parent / "still.jpg").is_file():
            out.append(p)
    return out


def phone_fx(rng: random.Random, size: int) -> dict[str, float]:
    """Random phone-camera parameters for one crop of *size* px: the app's crops are ~49 px for the same field (a third
    of ours), soft and JPEG'd."""
    return {"lowPx": rng.randint(max(16, size // 4), max(17, size // 2)), "blurSigma": round(rng.uniform(0.0, 1.2), 2),
            "noiseStd": round(rng.uniform(0.5, 4.0), 2), "jpegQuality": rng.randint(55, 92)}


def apply_phone_fx(img, fx: dict[str, float], seed: int = 0):
    """*fx* (:func:`phone_fx`) on a crop: down to ``lowPx`` and back (resolution), then blur / noise / JPEG."""
    from simple_sim.board3d.photofx import camera_fx
    n = img.shape[0]
    low = cv2.resize(img, (int(fx["lowPx"]), int(fx["lowPx"])), interpolation=cv2.INTER_AREA)
    up = cv2.resize(low, (n, n), interpolation=cv2.INTER_CUBIC)
    return camera_fx(up, blur_sigma=fx["blurSigma"] * n / fx["lowPx"], noise_std=fx["noiseStd"], jpeg_quality=int(fx["jpegQuality"]), seed=seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer: labelled defect crops, several per render")
    ap.add_argument("--board3d", type=Path, required=True)
    ap.add_argument("--source", default="scan", choices=sorted(SOURCES))
    ap.add_argument("--sheets", type=int, default=10)
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--window-mm", type=float, default=25.0)
    ap.add_argument("--spacing-mm", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=128, help="crop size in px")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--photos", type=Path,
                    help="render each sheet through a real phone camera: a directory searched for geometry.json of this "
                         "product / side (e.g. the DeepVision runs) or a text file with one geometry.json per line; crops "
                         "are cut like the app's (perspective kept), the real crop is the phone still at the same place")
    ap.add_argument("--photo-scale", type=float, default=0.5, help="--photos: render size / photo size")
    ap.add_argument("--photo-texture", action="store_true",
                    help="--photos: texture the board and the low parts with the sheet's own still (its light, colour "
                         "and residue; tall parts from the mosaic) instead of the photo mosaic: an OK part then shows "
                         "the photo itself and only the changed parts are rendered")
    ap.add_argument("--fitted", type=Path,
                    help="shapes.json of tools/board3d_fitdefect.py: half the defect ops take the shape of a reproduced "
                         "real defect (tombstone tilt, bridge size, tinned lands, ...)")
    ap.add_argument("--camera-fx", action="store_true",
                    help="phone-camera look per crop (random resolution loss, blur, sensor noise, JPEG; the same on the "
                         "render and its real crop): the reviewed app crops are phone stills at ~16 px/mm")
    ap.add_argument("--real-ok", action="store_true",
                    help="also label every real crop (the place as it is, the part fine) as OK: without real pixels in "
                         "OK a classifier learns 'rendered = defect' and rejects every real part")
    ap.add_argument("--variants", action="store_true",
                    help="each render a random unit: part catalog variants and mechanics with a presence probability p")
    ap.add_argument("--all-rendered", action="store_true",
                    help="render every part (bare board under the parts, every part lit): by default a part at its "
                         "nominal place shows the scan, so OK crops would be real pixels and defect crops rendered, "
                         "a shortcut a classifier can learn instead of the defect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", type=Path)
    add_json_flag(ap)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.photos and a.source == "scan":
        a.source = "photos"                       # a phone view shows the mounted board: its texture and stages
    board3d = load_board3d(a.board3d)
    sub = prepare_substrate(board3d, a.board3d, source=a.source)
    r = sub["raster"]
    out = a.out or a.board3d / "dataset" / a.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "real").mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    shapes = load_json(a.fitted) if a.fitted else None
    rows, t0 = [], time.monotonic()
    w = min(a.window_mm, r["widthMm"], r["heightMm"])
    geos = photo_geometries(a.photos, board3d) if a.photos else []
    if a.photos and not geos:
        raise SystemExit(f"no photo geometry of this product / side under {a.photos}")
    for s in range(a.sheets):
        geo_path = rng.choice(geos) if geos else None
        if geos:                                  # a phone view sees the whole board: parts from all of it
            window = BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], r["pxPerMm"])
            # the still's own reviewed defects (tools/board3d_register.py --labels): never in a chosen part's field,
            # its real crop would go in as OK
            faults = [tuple(f["centerMm"]) for f in load_json(geo_path).get("faults", [])]
            plan = plan_sheet(board3d, window, a.per_sheet, a.spacing_mm, rng, stages=set(sub["stages"]), avoid=faults, shapes=shapes)
        else:
            for _ in range(20):                   # a window with enough eligible parts
                x0 = rng.uniform(r["x0"], r["x0"] + r["widthMm"] - w)
                y0 = rng.uniform(r["y0"], r["y0"] + r["heightMm"] - w)
                window = BoardRaster(round(x0, 2), round(y0, 2), w, w, r["pxPerMm"])
                plan = plan_sheet(board3d, window, a.per_sheet, a.spacing_mm, rng, stages=set(sub["stages"]), shapes=shapes)
                if len(plan) >= max(1, a.per_sheet // 2):
                    break
        if not plan:
            continue
        ops = [p["op"] for p in plan if p["op"]]
        unit = sample_variants(board3d, rng) if a.variants else board3d
        scene = apply_defects(unit, ops) if ops else unit
        name = f"sheet_{a.name}_{s:04d}"
        if geos:
            src = still_source(geo_path) if a.photo_texture else a.source if a.source in ("photos", "mounted") else "photos"
            res = photo_view(scene, a.board3d, geo_path, name, scale=a.photo_scale, samples=a.samples, device=a.device,
                             source=src, refine=False,
                             # the still is its own texture: no colour matrix / tone curve (fitted on the green board,
                             # the curve turned every white pink)
                             calibrate=not a.photo_texture)
            render = cv2.imread(res["image"], cv2.IMREAD_COLOR)
            still = cv2.imread(res["still"], cv2.IMREAD_COLOR)
            real = cv2.resize(still, render.shape[1::-1], interpolation=cv2.INTER_AREA)
            h_img = np.diag([a.photo_scale, a.photo_scale, 1.0]) @ board_to_source(load_json(geo_path))
            crop = lambda im, c: crop_photo(im, h_img, c, a.size)  # noqa: E731
        else:
            look = {"light": {"preset": "flatbed", "scanNominal": False}, "under_parts": "bare"} if a.all_rendered else {}
            res = render_window(scene, a.board3d, window, name, samples=a.samples, device=a.device, **look)
            render, real = cv2.imread(res["image"], cv2.IMREAD_COLOR), crop_real(board3d, window)
            crop = lambda im, c: crop_part(im, window, c, a.size)  # noqa: E731
        kept = 0
        for k, p in enumerate(plan):
            sid = f"{a.name}_{s:04d}_{k:02d}"
            got, ref = crop(render, p["component"]), crop(real, p["component"])
            if got is None or ref is None:
                continue                          # outside this photo's frame
            (img, geo), (ref, _) = got, ref
            fx = None
            if a.camera_fx and not geos:          # a phone view has its own camera look (photo_view)
                fx = phone_fx(rng, a.size)
                img, ref = (apply_phone_fx(x, fx, seed=rng.randrange(2**31)) for x in (img, ref))
            cv2.imwrite(str(out / "images" / f"{sid}.png"), img)
            cv2.imwrite(str(out / "real" / f"{sid}.png"), ref)
            c = p["component"]
            rows.append({"id": sid, "image": f"images/{sid}.png", "real": f"real/{sid}.png", "label": str(p["class"]),
                         "op": p["op"], "refdes": c["refdes"], "package": c["package"], "family": c["family"],
                         "sheet": s, "source": a.source, **({"variants": unit.get("variants", {})} if a.variants else {}),
                         **({"cameraFx": fx} if fx else {}), **({"photo": str(geo_path)} if geo_path else {}), **geo})
            kept += 1
            if a.real_ok:
                rows.append({**rows[-1], "id": f"{sid}_real", "image": f"real/{sid}.png", "label": "OK", "op": None, "realCrop": True})
        plan = plan[:kept] if kept < len(plan) else plan
        print(f"sheet {s + 1}/{a.sheets}: {len(plan)} parts, {res['seconds']} s", flush=True)
    (out / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = {"crops": len(rows), "byClass": dict(Counter(x["label"] for x in rows)), "byFamily": dict(Counter(x["family"] for x in rows)),
               "seconds": round(time.monotonic() - t0, 1), "source": a.source, "samples": a.samples, "size": a.size, "seed": a.seed,
               "allRendered": a.all_rendered, "cameraFx": a.camera_fx, "realOk": a.real_ok, "variants": a.variants,
               "photos": str(a.photos) if a.photos else None, "photoTexture": a.photo_texture,
               "fitted": str(a.fitted) if a.fitted else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if a.json:
        emit_result("board3d-dataset", {"out": str(out), **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
