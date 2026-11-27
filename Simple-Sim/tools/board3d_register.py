#!/usr/bin/env python3
"""3D-Board-Renderer: give phone photos without a DeepVision run a camera, by registering them on the photo mosaic.

Each photo becomes ``<out>/<stem>/geometry.json`` + ``still.jpg`` (the layout of a run's ``auto/<view>/`` folder), so
``board3d_dataset.py --photos <out>`` renders through these cameras. Meant for the prepared defect boards (prep1 /
prep2): most of the reviewed real defects are on them, and their photos (warm light, soft, flux and glue residue)
look unlike the production runs.

With ``--labels`` (the app's ``.../aoi/labels/<side>.ndjson``) and ``--board`` the board's reviewed faults go into
the geometry (``faults``: centre in board mm and fault): ``board3d_dataset.py`` then never uses the real crop of such
a place as an OK sample.

Usage:
    python tools/board3d_register.py --board3d outputs/board3d/<product>/top --out outputs/board3d/<product>/top/photos_prep \\
        --labels .../products/<product>/aoi/labels/top.ndjson --board <board> <photos>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2  # noqa: E402

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.photoreg import geometry, register_photo  # noqa: E402
from simple_sim.board3d.render import prepare_substrate  # noqa: E402
from simple_sim.board3d.substrate import BoardRaster  # noqa: E402


def board_faults(labels: Path, board: str) -> list[dict]:
    """Newest review per part of *board* (by ``at``, else file order) that says ``fault``: ``[{centerMm, fault, id}]``."""
    newest: dict[str, dict] = {}
    for k, line in enumerate(labels.read_text(encoding="utf-8").splitlines()):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("board") != board or d.get("retracted") or not d.get("centerMm"):
            continue
        key = str(d.get("id"))
        stamp = (d.get("at") or "", k)
        if key not in newest or stamp >= newest[key]["_stamp"]:
            newest[key] = {**d, "_stamp": stamp}
    return [{"id": d["id"], "centerMm": [float(v) for v in d["centerMm"]], "fault": d.get("fault") or "unspecified"}
            for d in newest.values() if d.get("label") == "fault"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="register phone photos on the board3d photo mosaic (camera for --photos)")
    ap.add_argument("--board3d", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="app review labels (<side>.ndjson): the board's faults go into the geometry")
    ap.add_argument("--board", help="board name in --labels (e.g. prep1-<product>-top)")
    ap.add_argument("--min-cc", type=float, default=0.3, help="reject a registration whose final ECC correlation is lower")
    ap.add_argument("photos", nargs="+", type=Path)
    add_json_flag(ap)
    a = ap.parse_args(argv)
    board3d = load_board3d(a.board3d)
    sub = prepare_substrate(board3d, a.board3d, source="photos")
    raster = BoardRaster.from_meta(sub["raster"])
    mosaic = cv2.imread(sub["maps"]["scan_rgb"], cv2.IMREAD_COLOR)
    faults = board_faults(a.labels, a.board) if a.labels and a.board else []
    done, failed = [], {}
    for p in a.photos:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            failed[str(p)] = "not readable"
            continue
        try:
            b, stats = register_photo(img, mosaic, raster)
        except ValueError as e:
            failed[str(p)] = str(e)
            continue
        if stats["cc"] < a.min_cc:
            failed[str(p)] = f"ECC correlation {stats['cc']} < {a.min_cc}"
            continue
        d = a.out / p.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, d / "still.jpg")
        g = geometry(b, img.shape, product=board3d.get("product"), side=board3d.get("side"), stats=stats, source=str(p))
        if a.board:
            g["board"] = a.board
        if faults:
            g["faults"] = faults
        (d / "geometry.json").write_text(json.dumps(g, indent=1))
        done.append(str(d / "geometry.json"))
        print(f"{p.name}: cc {stats['cc']}, coarse {stats['coarse']}", flush=True)
    for k, v in failed.items():
        print(f"{Path(k).name}: FAILED {v}", flush=True)
    res = {"registered": len(done), "failed": failed, "faults": len(faults), "out": str(a.out)}
    print(json.dumps(res))
    if a.json:
        emit_result("board3d-register", res, ok=bool(done))
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
