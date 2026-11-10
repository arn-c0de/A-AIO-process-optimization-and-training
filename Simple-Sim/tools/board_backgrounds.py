#!/usr/bin/env python3
"""Cut empty board areas out of the DeepVision autopilot stills for ``render.texture_dir``.

The autopilot keeps one folder per inspection run (``test-results/auto/**/<run>/``) with the camera
still (``still.jpg``), ``geometry.json`` (``boardToSource``: 3x3 homography board mm -> still pixels,
``side``, ``sourceSize``) and ``components.json`` (every placed part with ``centerMm``, ``fieldMm`` (the
app's crop field = 3.04 body lengths) and, on newer runs, ``shapes`` (body polygon in board mm)). This
tool slides a window over the board, keeps the windows that

1. lie inside the populated board area (bounding box of the part centres shrunk by ``--margin-mm``,
   so neither the board edge nor the table shows),
2. contain no part: every part occupies a square around its centre of half side
   ``max(0.6 mm, 0.5 x fieldMm / 3.04 x 1.6)`` (body plus pads plus a little clearance; the body polygon
   when ``shapes`` exist), grown by ``--keepout-mm``,
3. look like bare solder mask (colour heuristic, needed because the pose of a run can be off by a few
   mm and components.json knows nothing about test points, holes, labels or fiducials): at least
   ``--mask-share`` of the pixels green (OpenCV hue 35-100, saturation > 40, value 25-215), at most
   0.5 % bright (grey > 200), 2 % grey/white print (saturation < 45, value > 110) and 1 % very dark
   (grey < 20) pixels,

and writes each as a rectified PNG at ``--mm-per-px`` (``roi.mm_per_px`` of the run configs the textures
are for, default 0.01613 = 62 px/mm of the 0402 profiles): ``render.texture_dir`` crops a ROI-sized patch
at 1:1, so the texture must be at the profile's pixel scale, not at the still's ~20 px/mm (the renderer's
camera model resamples it to the app's 16 px/mm anyway). The window is ``--window-mm`` wide (default
5.2 mm: a 256 px ROI at 62 px/mm is 4.1 mm, the camera roll canvas needs ~1.25x). The same still is
used by several runs (293 source photos behind 1047 runs); each photo is processed once, with the run
that has the most parts.

Output: ``<out>/<side>/<photo-stem>_<x>_<y>.png`` plus ``<out>/<side>/summary.json`` (counts per
reason, px/mm statistics, the settings; no image content). ``--dry-run`` only prints the summary.

Known gap: ``geometry.json`` holds the board pose, not the board outline or the drill/test-point
layout, and 70 % of the runs have no body polygons. The bounding-box margin and the colour heuristic
stand in for both; windows with a visible part or hole are rejected by colour, not by geometry.

Usage:
    python tools/board_backgrounds.py --runs .../DeepVision-Web/test-results/auto --out data/board_textures [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.io_utils import read_json  # noqa: E402

SIDES = ("top", "bottom")
CROP_PART_LENGTHS = 3.04       # the app's crop field in body lengths (measured on the real crops)
PART_HALF_FACTOR = 1.6         # keep-out half side / body half length (body + pads + a little clearance)
PART_HALF_MIN_MM = 0.6
DEFAULT_MM_PER_PX = 0.01613    # 62 px/mm: roi.mm_per_px of the 0402 run configs
DEFAULT_WINDOW_MM = 5.2
DEFAULT_MARGIN_MM = 4.0
DEFAULT_KEEPOUT_MM = 0.4
DEFAULT_MASK_SHARE = 0.9
MASK_HUE = (35.0, 100.0)       # OpenCV hue of green solder masks (olive to teal)
BRIGHT_SHARE_MAX = 0.005       # metal, silkscreen, labels
PRINT_SHARE_MAX = 0.02         # grey/white print (silkscreen text, labels): low saturation, mid-bright
DARK_SHARE_MAX = 0.01          # holes, deep shadows, board edge


@dataclass
class Run:
    folder: Path
    side: str
    photo: str
    homography: np.ndarray          # board mm -> still px
    source_size: tuple[int, int]    # (w, h) of the still
    parts: list[dict[str, Any]]
    quality: dict[str, Any] = field(default_factory=dict)


def load_run(folder: Path) -> Run | None:
    """The run in *folder* (still + geometry + components), or ``None`` when anything is unusable."""
    geometry = read_json(folder / "geometry.json")
    components = read_json(folder / "components.json")
    if not geometry or not components or not (folder / "still.jpg").is_file():
        return None
    h = geometry.get("boardToSource")
    side = str(geometry.get("side") or components.get("side") or "")
    if not isinstance(h, list) or len(h) != 9 or side not in SIDES:
        return None
    size = geometry.get("sourceSize") or []
    parts = [p for p in components.get("items", []) if isinstance(p.get("centerMm"), list) and len(p["centerMm"]) == 2]
    if len(parts) < 10:
        return None
    return Run(folder=folder, side=side, photo=str(geometry.get("file") or folder.name),
               homography=np.array(h, dtype=np.float64).reshape(3, 3),
               source_size=(int(size[0]), int(size[1])) if len(size) == 2 else (0, 0), parts=parts,
               quality=dict(geometry.get("quality") or {}))


def find_runs(root: Path) -> list[Run]:
    """One run per source photo and side (the one with the most parts), sorted by photo name."""
    best: dict[tuple[str, str], Run] = {}
    for still in sorted(root.rglob("still.jpg")):
        run = load_run(still.parent)
        if run is None:
            continue
        key = (run.side, run.photo)
        if key not in best or len(run.parts) > len(best[key].parts):
            best[key] = run
    return [best[k] for k in sorted(best)]


def part_boxes(parts: list[dict[str, Any]], keepout_mm: float) -> np.ndarray:
    """``[x0, y0, x1, y1]`` (board mm) per part: body polygon when known, else a square from ``fieldMm``."""
    boxes = []
    for p in parts:
        pts = [q for s in (p.get("shapes") or []) if s.get("kind") == "body" for q in s.get("points", [])]
        if pts:
            arr = np.array(pts, dtype=np.float64)
            x0, y0 = arr.min(axis=0)
            x1, y1 = arr.max(axis=0)
            half = 0.0
            # the polygon is the body: add the pads (about half a body length on each side)
            grow = 0.35 * max(x1 - x0, y1 - y0)
            x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
        else:
            cx, cy = float(p["centerMm"][0]), float(p["centerMm"][1])
            body = float(p.get("fieldMm") or 0.0) / CROP_PART_LENGTHS
            half = max(PART_HALF_MIN_MM, 0.5 * body * PART_HALF_FACTOR)
            x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        boxes.append([x0 - keepout_mm, y0 - keepout_mm, x1 + keepout_mm, y1 + keepout_mm])
    return np.array(boxes, dtype=np.float64)


def free_windows(parts: list[dict[str, Any]], window_mm: float, margin_mm: float, keepout_mm: float,
                 step_mm: float | None = None) -> list[tuple[float, float]]:
    """Top-left corners (board mm) of the ``window_mm`` squares inside the populated area that touch no part box."""
    centres = np.array([p["centerMm"] for p in parts], dtype=np.float64)
    x_lo, y_lo = centres.min(axis=0) + margin_mm
    x_hi, y_hi = centres.max(axis=0) - margin_mm
    boxes = part_boxes(parts, keepout_mm)
    step = step_mm if step_mm is not None else window_mm / 2.0
    out = []
    for y in np.arange(y_lo, y_hi - window_mm + 1e-9, step):
        for x in np.arange(x_lo, x_hi - window_mm + 1e-9, step):
            hit = (boxes[:, 0] < x + window_mm) & (boxes[:, 2] > x) & (boxes[:, 1] < y + window_mm) & (boxes[:, 3] > y)
            if not hit.any():
                out.append((float(x), float(y)))
    return out


def window_transform(homography: np.ndarray, x_mm: float, y_mm: float, mm_per_px: float) -> np.ndarray:
    """Still px -> texture px for the window whose top-left corner is ``(x_mm, y_mm)``."""
    mm_to_tex = np.array([[1.0 / mm_per_px, 0.0, -x_mm / mm_per_px],
                          [0.0, 1.0 / mm_per_px, -y_mm / mm_per_px],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
    return mm_to_tex @ np.linalg.inv(homography)


def cut_window(still: np.ndarray, homography: np.ndarray, x_mm: float, y_mm: float, window_mm: float,
               mm_per_px: float) -> np.ndarray:
    """Rectified window of the still at *mm_per_px* (square, ``window_mm / mm_per_px`` px per side)."""
    size = int(round(window_mm / mm_per_px))
    return cv2.warpPerspective(still, window_transform(homography, x_mm, y_mm, mm_per_px), (size, size),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def inside_still(homography: np.ndarray, x_mm: float, y_mm: float, window_mm: float, size: tuple[int, int]) -> bool:
    """All four window corners project into the still (no black border in the cut)."""
    if size == (0, 0):
        return True
    corners = np.array([[x_mm, y_mm, 1.0], [x_mm + window_mm, y_mm, 1.0], [x_mm, y_mm + window_mm, 1.0],
                        [x_mm + window_mm, y_mm + window_mm, 1.0]])
    proj = corners @ homography.T
    proj = proj[:, :2] / proj[:, 2:3]
    return bool((proj[:, 0] >= 0).all() and (proj[:, 1] >= 0).all() and (proj[:, 0] <= size[0] - 1).all()
                and (proj[:, 1] <= size[1] - 1).all())


def mask_check(patch: np.ndarray, mask_share: float) -> str | None:
    """``None`` when *patch* looks like bare solder mask, else the rejection reason."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).reshape(-1)
    green = (hsv[:, 0] >= MASK_HUE[0]) & (hsv[:, 0] <= MASK_HUE[1]) & (hsv[:, 1] > 40) & (hsv[:, 2] > 25) & (hsv[:, 2] < 215)
    if float(np.mean(grey > 200)) > BRIGHT_SHARE_MAX:
        return "bright pixels (metal, print, label)"
    if float(np.mean((hsv[:, 1] < 45) & (hsv[:, 2] > 110))) > PRINT_SHARE_MAX:
        return "print (grey/white text, label)"
    if float(np.mean(grey < 20)) > DARK_SHARE_MAX:
        return "dark pixels (hole, edge, shadow)"
    if float(green.mean()) < mask_share:
        return "not mask-coloured"
    return None


def still_px_per_mm(homography: np.ndarray, centre_mm: np.ndarray) -> float:
    """Local scale of the still at *centre_mm* (px per mm along x)."""
    pts = np.array([[centre_mm[0], centre_mm[1], 1.0], [centre_mm[0] + 1.0, centre_mm[1], 1.0]]) @ homography.T
    pts = pts[:, :2] / pts[:, 2:3]
    return float(np.hypot(*(pts[1] - pts[0])))


def process_run(run: Run, out: Path, *, mm_per_px: float, window_mm: float, margin_mm: float, keepout_mm: float,
                mask_share: float, max_per_run: int, dry_run: bool, rng: np.random.Generator) -> dict[str, Any]:
    """Cut the accepted windows of one run; returns its counts."""
    counts: Counter = Counter()
    candidates = free_windows(run.parts, window_mm, margin_mm, keepout_mm)
    counts["candidates (free of parts)"] = len(candidates)
    candidates = [c for c in candidates if inside_still(run.homography, c[0], c[1], window_mm, run.source_size)]
    counts["inside the still"] = len(candidates)
    if candidates and max_per_run > 0 and len(candidates) > max_per_run * 3:
        idx = sorted(rng.choice(len(candidates), max_per_run * 3, replace=False))
        candidates = [candidates[i] for i in idx]
    still = None
    accepted = 0
    if candidates:
        still = cv2.imread(str(run.folder / "still.jpg"), cv2.IMREAD_COLOR)
        if still is None:
            counts["still unreadable"] += 1
            candidates = []
    for x, y in candidates:
        if max_per_run > 0 and accepted >= max_per_run:
            break
        patch = cut_window(still, run.homography, x, y, window_mm, mm_per_px)
        reason = mask_check(patch, mask_share)
        if reason is not None:
            counts[f"rejected: {reason}"] += 1
            continue
        accepted += 1
        if not dry_run:
            dest = out / run.side
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / f"{Path(run.photo).stem}_{x:.1f}_{y:.1f}.png"), patch)
    counts["accepted"] = accepted
    centre = np.array([p["centerMm"] for p in run.parts], dtype=np.float64).mean(axis=0)
    return {"side": run.side, "photo": run.photo, "run": str(run.folder), "parts": len(run.parts),
            "with_shapes": sum(1 for p in run.parts if p.get("shapes")), "still_px_per_mm": still_px_per_mm(run.homography, centre),
            "quality": run.quality.get("mode"), "counts": dict(counts)}


def summarize(results: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"settings": settings, "sides": {}}
    for side in SIDES:
        rows = [r for r in results if r["side"] == side]
        if not rows:
            continue
        counts: Counter = Counter()
        for r in rows:
            counts.update(r["counts"])
        ppm = [r["still_px_per_mm"] for r in rows]
        out["sides"][side] = {"photos": len(rows), "textures": counts["accepted"], "counts": dict(counts),
                              "still_px_per_mm": [round(float(v), 2) for v in np.percentile(ppm, (10, 50, 90))],
                              "runs_with_shapes": sum(1 for r in rows if r["with_shapes"]),
                              "photos_without_texture": sum(1 for r in rows if r["counts"].get("accepted", 0) == 0)}
    return out


def format_summary(summary: dict[str, Any]) -> str:
    s = summary["settings"]
    lines = [f"window {s['window_mm']} mm -> {int(round(s['window_mm'] / s['mm_per_px']))} px at {1 / s['mm_per_px']:.1f} px/mm "
             f"(mm_per_px {s['mm_per_px']}), margin {s['margin_mm']} mm, keep-out {s['keepout_mm']} mm, mask share >= {s['mask_share']}"]
    for side, st in summary["sides"].items():
        lines.append(f"{side}: {st['photos']} photos ({st['runs_with_shapes']} with body polygons), still {st['still_px_per_mm']} px/mm "
                     f"(p10/p50/p90), {st['textures']} textures, {st['photos_without_texture']} photos without")
        for k, v in sorted(st["counts"].items()):
            lines.append(f"    {k:40s} {v:7d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True, help="autopilot results root (searched for still.jpg)")
    ap.add_argument("--out", type=Path, required=True, help="texture root: <out>/<side>/*.png")
    ap.add_argument("--sides", nargs="+", default=list(SIDES), choices=SIDES)
    ap.add_argument("--mm-per-px", type=float, default=DEFAULT_MM_PER_PX, help="texture scale = roi.mm_per_px of the target configs")
    ap.add_argument("--window-mm", type=float, default=DEFAULT_WINDOW_MM)
    ap.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM, help="stay this far inside the populated area")
    ap.add_argument("--keepout-mm", type=float, default=DEFAULT_KEEPOUT_MM, help="extra clearance around every part")
    ap.add_argument("--mask-share", type=float, default=DEFAULT_MASK_SHARE)
    ap.add_argument("--max-per-photo", type=int, default=12, help="textures per source photo (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    runs = [r for r in find_runs(args.runs) if r.side in set(args.sides)]
    if not runs:
        raise SystemExit(f"no usable runs (still.jpg + geometry.json + components.json) below {args.runs}")
    rng = np.random.default_rng(args.seed)
    results = [process_run(r, args.out, mm_per_px=args.mm_per_px, window_mm=args.window_mm, margin_mm=args.margin_mm,
                           keepout_mm=args.keepout_mm, mask_share=args.mask_share, max_per_run=args.max_per_photo,
                           dry_run=args.dry_run, rng=rng) for r in runs]
    settings = {"mm_per_px": args.mm_per_px, "window_mm": args.window_mm, "margin_mm": args.margin_mm,
                "keepout_mm": args.keepout_mm, "mask_share": args.mask_share, "max_per_photo": args.max_per_photo,
                "runs_root": str(args.runs), "dry_run": bool(args.dry_run)}
    summary = summarize(results, settings)
    print(format_summary(summary))
    if not args.dry_run:
        for side, st in summary["sides"].items():
            dest = args.out / side
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "summary.json").write_text(json.dumps({**st, "settings": settings}, indent=1) + "\n", encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
