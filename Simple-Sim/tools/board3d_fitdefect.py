#!/usr/bin/env python3
"""3D-Board-Renderer T6: reproduce reviewed real defects as defect ops (parameter search against the real crop).

For every reviewed fault of the registered stills (``tools/board3d_register.py --labels``) with a reproducible class
(fehlt, loetbruecke, grabstein, versetzt, verdreht) the candidates of that class are rendered in the part's crop
through the still's camera and scored against the real crop (:mod:`simple_sim.board3d.defectfit`).

Output (``--out``): ``fitted.jsonl`` (one line per fault: part, nominal score, best op and score, ``beatsNominal``),
``shapes.json`` (per op kind the shape parameters of the fits that beat the nominal part: ``board3d_dataset.py
--fitted``) and with ``--keep`` per fault an image real | nominal | best.

Usage:
    python tools/board3d_fitdefect.py --board3d outputs/board3d/<product>/top --photos outputs/board3d/<product>/top/photos_prep \\
        --out outputs/board3d/<product>/top/fitted --candidates 10 [--faults loetbruecke grabstein] [--limit 20] [--keep]
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

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.io_utils import load_json  # noqa: E402
from simple_sim.board3d import load_board3d  # noqa: E402
from simple_sim.board3d.defectfit import FAULT_CLASS, fit_fault, shape_params  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fit defect ops to reviewed real defects of registered stills")
    ap.add_argument("--board3d", type=Path, required=True)
    ap.add_argument("--photos", type=Path, required=True, help="folder of registered stills (geometry.json with faults)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--faults", nargs="*", default=sorted(FAULT_CLASS), help="app fault names to fit")
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="at most this many faults (0: all)")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--keep", action="store_true", help="write real | nominal | best per fault")
    ap.add_argument("--seed", type=int, default=0)
    add_json_flag(ap)
    a = ap.parse_args(argv)
    board3d = load_board3d(a.board3d)
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    # a fault is reviewed once per board, every still of it shows it: fit each fault on one still (the first that
    # has it in frame)
    seen: set[str] = set()
    jobs = []
    for g in sorted(a.photos.rglob("geometry.json")):
        for f in load_json(g).get("faults", []):
            key = f"{load_json(g).get('board')}:{f.get('id')}"
            if f.get("fault") in a.faults and key not in seen:
                jobs.append((g, f, key))
    rows, t0 = [], time.monotonic()
    for g, f, key in jobs:
        if key in seen:
            continue
        if a.limit and len(rows) >= a.limit:
            break
        res = fit_fault(board3d, a.board3d, g, f, rng, candidates=a.candidates, samples=a.samples, device=a.device,
                        keep=a.out / "keep" if a.keep else None)
        if res is None:
            continue                          # no part / out of this still's frame: maybe in the next still
        seen.add(key)
        rows.append(res)
        b = res["best"] or {}
        print(f"{len(rows)}: {res['fault']} {res['refdes']}: nominal {res['nominal']['score']} -> best {b.get('score')} "
              f"({(b.get('op') or {}).get('op')}) {round(time.monotonic() - t0)} s", flush=True)
    (a.out / "fitted.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    shapes = shape_params(rows)
    (a.out / "shapes.json").write_text(json.dumps(shapes, indent=1))
    summary = {"fitted": len(rows), "beatsNominal": sum(r["beatsNominal"] for r in rows),
               "byFault": dict(Counter(r["fault"] for r in rows)), "shapes": {k: len(v) for k, v in shapes.items()},
               "seconds": round(time.monotonic() - t0, 1), "out": str(a.out)}
    print(json.dumps(summary))
    if a.json:
        emit_result("board3d-fitdefect", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
