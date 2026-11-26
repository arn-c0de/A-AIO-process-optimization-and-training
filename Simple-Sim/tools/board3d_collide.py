#!/usr/bin/env python3
"""3D-Board-Renderer: find bodies that run into each other and fix the outlines that are not measured.

Builds the bodies as the renderer does (measured fits and photo fits applied, :func:`simple_sim.board3d.scene.apply_fit`),
tests every pair (:func:`simple_sim.board3d.collide.find_collisions`), fixes the connector / box outlines from the app
with the part catalog's datasheet size, the plated holes of the scan and the photo mosaic
(:func:`simple_sim.board3d.collide.resolve`) and writes ``<board3d>/parttex/collide.json``, which every later render and
the viewer's GLB apply. Collisions of two measured bodies are only reported.

``--check`` draws each collision region onto the photo mosaic (``<board3d>/collide/<n>_<a>_<b>.jpg``: red = before,
green = after); ``--render`` also renders those regions in Blender (top view against the photos,
``renders/collide_<n>_compare.png``).

Usage:
    python tools/board3d_collide.py --board3d outputs/board3d/<product>/<side> [--dry-run] [--check] [--render]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.collide import BOARD_DE, CATALOG_SLACK, corners, find_collisions, is_soft, resolve  # noqa: E402
from simple_sim.board3d.packages import ipc_dimensions  # noqa: E402
from simple_sim.board3d.parttex import _lab  # noqa: E402
from simple_sim.board3d.scene import apply_fit, job_component  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402
from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.io_utils import atomic_write_json, load_json  # noqa: E402
from simple_sim.parts import PartCatalog  # noqa: E402

COLLIDE_FILE = "parttex/collide.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer: find and fix bodies that run into each other")
    ap.add_argument("--board3d", type=Path, required=True, help="folder with board3d.json")
    ap.add_argument("--stages", default="smd,tht,mount", help="comma list of the stages whose bodies are tested")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write collide.json")
    ap.add_argument("--check", action="store_true", help="draw every collision region onto the photo mosaic (before / after)")
    ap.add_argument("--render", action="store_true", help="also render the regions in Blender (top view, compared with the photos)")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--device", default="GPU")
    add_json_flag(ap)
    return ap.parse_args(argv)


def job_bodies(board3d: dict, stages: set[str]) -> list[dict]:
    """The renderer's bodies (job components) of *stages*, with the PnP point (pin reference of THT parts)."""
    out = []
    for c in apply_fit(board3d)["components"]:
        if c.get("stage") not in stages or not c.get("pose") or not c.get("body") or c.get("dnp"):
            continue
        jc = copy.deepcopy(job_component(c))
        p = c.get("pnp") or c["pose"]
        jc["pnp"] = [float(p["x"]), float(p["y"])]
        out.append(jc)
    return out


def name_sizes(bodies: list[dict], board3d: dict) -> dict[str, dict[str, float]]:
    """Bodies a fit grew past the size of their IPC package name by CATALOG_SLACK: flagged ``soft`` in place; returns
    refdes -> the name's ``{"l", "w"}`` (long / short matched to the outline's)."""
    pkg = {c["refdes"]: str(c.get("package", "")) for c in board3d["components"]}
    out = {}
    for jc in bodies:
        if is_soft(jc) or not (d := ipc_dimensions(pkg.get(jc["refdes"], ""))):
            continue
        a, b = sorted((float(d["bodyW"]), float(d["bodyH"])))
        name = {"l": a, "w": b} if jc["l"] <= jc["w"] else {"l": b, "w": a}
        if any(jc[k] > CATALOG_SLACK * name[k] for k in ("l", "w")):
            jc["soft"] = True
            out[jc["refdes"]] = name
    return out


def catalog_sizes(board3d: dict, refs: set[str]) -> dict[str, dict[str, float]]:
    """refdes -> the part catalog's datasheet ``{"l", "w"}`` (only those it has)."""
    cat = PartCatalog.load()
    out = {}
    for c in board3d["components"]:
        if c["refdes"] not in refs:
            continue
        m = cat.lookup(texts=[str(c[k]) for k in ("value", "part", "mpn") if c.get(k)], package=str(c.get("package", "")))
        body = (m.entry.data.get("body") or {}) if m else {}
        if sizes := {k: float(body[k]) for k in ("l", "w") if body.get(k)}:
            out[c["refdes"]] = sizes
    return out


def mosaic(folder: Path) -> tuple[np.ndarray, BoardRaster] | None:
    mj, mp = folder / "photos" / "mosaic.json", folder / "photos" / "mosaic.png"
    if not (mj.is_file() and mp.is_file()):
        return None
    r = load_json(mj)["raster"]
    return cv2.imread(str(mp), cv2.IMREAD_COLOR), BoardRaster(r["x0"], r["y0"], r["widthMm"], r["heightMm"], r["pxPerMm"])


def part_mask_fn(img: np.ndarray, raster: BoardRaster, board3d: dict):
    """board mm N x 2 -> True where the photo mosaic is not board colour (median of the mosaic away from all bodies)."""
    free = np.full(img.shape[:2], 255, np.uint8)
    for c in apply_fit(board3d)["components"]:
        if c.get("pose") and c.get("body"):
            cv2.fillPoly(free, [np.int32(raster.to_px(corners(c)))], 0)
    free = cv2.erode(free, np.ones((9, 9), np.uint8))
    board = np.median(img[free > 0], 0) if (free > 0).sum() > 1000 else np.median(img.reshape(-1, 3), 0)
    not_board = np.linalg.norm(_lab(img.reshape(-1, 3)) - _lab(board.reshape(1, 3)), axis=1).reshape(img.shape[:2]) >= BOARD_DE

    def fn(pts: np.ndarray) -> np.ndarray:
        px = np.rint(raster.to_px(pts)).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < img.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < img.shape[0])
        out = np.zeros(len(pts), bool)
        out[ok] = not_board[px[ok, 1], px[ok, 0]]
        return out
    return fn


def regions(cols: list[dict], pad: float = 3.0) -> list[dict]:
    """Collisions grouped into board regions (overlapping boxes merged): ``[{"bbox", "refs"}]``."""
    groups = []
    for c in cols:
        b = [c["bbox"][0] - pad, c["bbox"][1] - pad, c["bbox"][2] + pad, c["bbox"][3] + pad]
        refs = {c["a"], c["b"]}
        for g in [g for g in groups if not (b[2] < g["bbox"][0] or g["bbox"][2] < b[0] or b[3] < g["bbox"][1] or g["bbox"][3] < b[1])]:
            groups.remove(g)
            b = [min(b[0], g["bbox"][0]), min(b[1], g["bbox"][1]), max(b[2], g["bbox"][2]), max(b[3], g["bbox"][3])]
            refs |= g["refs"]
        groups.append({"bbox": b, "refs": refs})
    return groups


def draw_check(img: np.ndarray, raster: BoardRaster, region: dict, before: dict, after: dict, path: Path) -> None:
    out = img.copy()
    for ref in sorted(region["refs"]):
        for comp, col, th in ((before[ref], (0, 0, 255), 2), (after.get(ref, before[ref]), (0, 220, 0), 2)):
            pts = np.int32(raster.to_px(corners(comp)))
            cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
        p = raster.to_px(corners(after.get(ref, before[ref]))).mean(0)
        cv2.putText(out, ref, (int(p[0]) - 20, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    x0, y0, x1, y1 = region["bbox"]
    (c0, r1), (c1, r0) = np.int32(raster.to_px(np.array([[x0, y0], [x1, y1]])))
    crop = out[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
    if crop.size:
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    board3d = load_board3d(a.board3d)
    board3d.pop("collide", None)                      # fix from the measured outlines, not from an earlier fix
    stages = set(a.stages.split(","))
    bodies = job_bodies(board3d, stages)
    pf = a.board3d / "parttex" / "photofit.json"
    absent = set(load_json(pf).get("absentInPhotos", [])) if pf.is_file() else set()
    for jc in bodies:
        if jc["refdes"] in absent:
            jc["noPhoto"] = True                      # the photos show board there: placement from the pins alone
    grown = name_sizes(bodies, board3d)
    before = {c["refdes"]: copy.deepcopy(c) for c in bodies}
    found = find_collisions(bodies)
    soft = {r for col in found for r in col["soft"]}
    holes_file = a.board3d / "substrate" / "holes.json"
    holes = load_json(holes_file)["holes"] if holes_file.is_file() else []
    mos = mosaic(a.board3d)
    mask = part_mask_fn(*mos, board3d) if mos is not None else None
    sizes = {**catalog_sizes(board3d, soft), **{r: v for r, v in grown.items() if r in soft}}
    res = resolve(bodies, catalog=sizes, holes=holes, part_mask=mask)
    after = {c["refdes"]: c for c in bodies}

    print(f"{len(found)} collisions in {len(bodies)} bodies ({', '.join(sorted(stages))})")
    for col in found:
        f = " ".join(f"{r}: {', '.join(res['fixed'][r]['how'])}" for r in col["soft"] if r in res["fixed"])
        print(f"  {col['a']:>6} x {col['b']:<6} {col['depthMm']:5.2f} mm deep, {col['areaMm2']:6.2f} mm2 ({100 * col['share']:3.0f} %)"
              f"  {'-> ' + f if f else '(both measured: report only)' if not col['soft'] else '(not fixed)'}")
    print(f"fixed {len(res['fixed'])} outlines, {len(res['left'])} collisions left")
    for col in res["left"]:
        print(f"  left: {col['a']:>6} x {col['b']:<6} {col['depthMm']:5.2f} mm deep ({100 * col['share']:3.0f} %)")
    if not a.dry_run:
        path = a.board3d / COLLIDE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": sorted(stages),
                                 "collisions": found, "fixed": res["fixed"], "left": res["left"]}, indent=1)
        print(f"wrote {path}")

    if a.check or a.render:
        out_dir = a.board3d / "collide"
        out_dir.mkdir(exist_ok=True)
        for old in out_dir.glob("*.jpg"):
            old.unlink()
        for i, reg in enumerate(regions(found)):
            tag = "_".join(sorted(reg["refs"])[:3])
            if mos is not None:
                draw_check(*mos, reg, before, after, out_dir / f"{i:02d}_{tag}.jpg")
            if a.render:
                from simple_sim.board3d.render import probe
                x0, y0, x1, y1 = reg["bbox"]
                scene = load_board3d(a.board3d) if not a.dry_run else board3d
                r = probe(scene, a.board3d, BoardRaster(x0, y0, x1 - x0, y1 - y0, 20.0), f"collide_{i:02d}", source="photos",
                          stages=stages, samples=a.samples, device=a.device)
                print(f"  region {i} ({tag}): {r['seconds']} s, compare {a.board3d / 'renders' / f'collide_{i:02d}_compare.png'}")
        print(f"check images: {out_dir}")
    if a.json:
        emit_result("board3d-collide", {"collisions": len(found), "fixed": res["fixed"], "left": res["left"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
