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

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.dataset import crop_part, plan_sheet  # noqa: E402
from simple_sim.board3d.render import SOURCES, crop_real, prepare_substrate, render_window  # noqa: E402
from simple_sim.board3d.scene import apply_defects, sample_variants  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402


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
    board3d = load_board3d(a.board3d)
    sub = prepare_substrate(board3d, a.board3d, source=a.source)
    r = sub["raster"]
    out = a.out or a.board3d / "dataset" / a.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "real").mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    rows, t0 = [], time.monotonic()
    w = min(a.window_mm, r["widthMm"], r["heightMm"])
    for s in range(a.sheets):
        for _ in range(20):                       # a window with enough eligible parts
            x0 = rng.uniform(r["x0"], r["x0"] + r["widthMm"] - w)
            y0 = rng.uniform(r["y0"], r["y0"] + r["heightMm"] - w)
            window = BoardRaster(round(x0, 2), round(y0, 2), w, w, r["pxPerMm"])
            plan = plan_sheet(board3d, window, a.per_sheet, a.spacing_mm, rng, stages=set(sub["stages"]))
            if len(plan) >= max(1, a.per_sheet // 2):
                break
        if not plan:
            continue
        ops = [p["op"] for p in plan if p["op"]]
        unit = sample_variants(board3d, rng) if a.variants else board3d
        scene = apply_defects(unit, ops) if ops else unit
        look = {"light": {"preset": "flatbed", "scanNominal": False}, "under_parts": "bare"} if a.all_rendered else {}
        res = render_window(scene, a.board3d, window, f"sheet_{a.name}_{s:04d}", samples=a.samples, device=a.device, **look)
        render, real = cv2.imread(res["image"], cv2.IMREAD_COLOR), crop_real(board3d, window)
        for k, p in enumerate(plan):
            sid = f"{a.name}_{s:04d}_{k:02d}"
            img, geo = crop_part(render, window, p["component"], a.size)
            ref, _ = crop_part(real, window, p["component"], a.size)
            cv2.imwrite(str(out / "images" / f"{sid}.png"), img)
            cv2.imwrite(str(out / "real" / f"{sid}.png"), ref)
            c = p["component"]
            rows.append({"id": sid, "image": f"images/{sid}.png", "real": f"real/{sid}.png", "label": str(p["class"]),
                         "op": p["op"], "refdes": c["refdes"], "package": c["package"], "family": c["family"],
                         "sheet": s, "source": a.source, **({"variants": unit.get("variants", {})} if a.variants else {}), **geo})
        print(f"sheet {s + 1}/{a.sheets}: {len(plan)} parts, {res['seconds']} s", flush=True)
    (out / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = {"crops": len(rows), "byClass": dict(Counter(x["label"] for x in rows)), "byFamily": dict(Counter(x["family"] for x in rows)),
               "seconds": round(time.monotonic() - t0, 1), "source": a.source, "samples": a.samples, "size": a.size, "seed": a.seed,
               "allRendered": a.all_rendered}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if a.json:
        emit_result("board3d-dataset", {"out": str(out), **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
