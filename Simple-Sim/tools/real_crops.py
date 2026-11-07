#!/usr/bin/env python3
"""Export real AOI crops from a DeepVision QUINS manifest into the class folders of the sim-to-real tools.

``tools/eval_real.py`` and ``tools/mix_real.py`` take a folder with one sub-folder per Simple-Sim
class (``OK``, ``MISSING``, ``MISSING@empty_pad``, ...). This tool fills such a folder from the
DeepVision-Web crop manifest (``.../datasets/<product>/quins/manifest.jsonl``, one JSON object per
crop with ``crop``, ``id``, ``form``, ``scan``, ``split``, ``class``), filtered by package form(s)
and split(s), with a random per-folder limit. Crops are copied unchanged (``<scan>_<id>.png``: ids
repeat across scans) and only counts are printed, so the real data stays local.

DeepVision review classes map to Simple-Sim classes as follows (:data:`CLASS_FOLDERS`); crops of
unmapped classes (``ungeprüft``, ``variante``) are skipped:

=======================  =========================  ====================================================
DeepVision class         folder                     meaning
=======================  =========================  ====================================================
gut                      OK                         reviewed good part
fehlalarm                OK@fehlalarm               AOI alarm, reviewed as good (hard positives)
leer                     MISSING@empty_pad          pad without a part (unpopulated by design)
fehler:fehlt             MISSING                    part missing
fehler:versetzt          MISALIGNED                 part shifted
fehler:verdreht          MISALIGNED@verdreht        part rotated
fehler:loetfehler        DEFECT@loetfehler          solder defect (no Simple-Sim class, counts as defect)
fehler:fremdkoerper      DEFECT@fremdkoerper        foreign object
fehler:falsches_bauteil  DEFECT@falsches_bauteil    wrong part
=======================  =========================  ====================================================

Keep evaluation crops (``--splits test``) apart from the ones mixed into training (``--splits train``).

Usage:
    python tools/real_crops.py --manifest .../quins/manifest.jsonl --out real_eval --forms form-02 --splits test --limit 500
    python tools/real_crops.py --manifest .../quins/manifest.jsonl --out real_eval --forms form-02 --splits train val test \\
        --classes fehler:fehlt fehler:versetzt fehler:verdreht fehler:loetfehler    # rare defects: every split
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

#: DeepVision review class -> class folder of ``tools/eval_real.py`` / ``tools/mix_real.py``.
CLASS_FOLDERS: dict[str, str] = {
    "gut": "OK",
    "fehlalarm": "OK@fehlalarm",
    "leer": "MISSING@empty_pad",
    "fehler:fehlt": "MISSING",
    "fehler:versetzt": "MISALIGNED",
    "fehler:verdreht": "MISALIGNED@verdreht",
    "fehler:loetfehler": "DEFECT@loetfehler",
    "fehler:fremdkoerper": "DEFECT@fremdkoerper",
    "fehler:falsches_bauteil": "DEFECT@falsches_bauteil",
}
SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select(rows: list[dict[str, Any]], forms: set[str], splits: set[str], classes: set[str] | None) -> dict[str, list[dict[str, Any]]]:
    """Rows per class folder, in manifest order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cls = str(row.get("class", ""))
        folder = CLASS_FOLDERS.get(cls)
        if folder is None or row.get("form") not in forms or row.get("split") not in splits:
            continue
        if classes is not None and cls not in classes:
            continue
        out.setdefault(folder, []).append(row)
    return out


def sample(rows: list[dict[str, Any]], limit: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    idx = sorted(rng.choice(len(rows), limit, replace=False))
    return [rows[i] for i in idx]


def export(manifest: Path, out: Path, forms: set[str], splits: set[str], classes: set[str] | None,
           limit: int, seed: int) -> dict[str, int]:
    """Copy the selected crops into ``out/<folder>/<scan>_<id><ext>``; returns the count per folder."""
    rng = np.random.default_rng(seed)
    root = manifest.parent
    counts: dict[str, int] = {}
    for folder, rows in sorted(select(read_manifest(manifest), forms, splits, classes).items()):
        chosen = sample(rows, limit, rng)
        dest = out / folder
        dest.mkdir(parents=True, exist_ok=True)
        for row in chosen:
            src = root / str(row["crop"])
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copyfile(src, dest / f"{row.get('scan', 'scan')}_{row['id']}{src.suffix.lower()}")
        counts[folder] = len(chosen)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True, help="DeepVision quins manifest.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="target folder (one sub-folder per class)")
    ap.add_argument("--forms", nargs="+", required=True, help="package forms, e.g. form-02")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=SPLITS)
    ap.add_argument("--classes", nargs="*", help=f"DeepVision classes to export (default: all of {sorted(CLASS_FOLDERS)})")
    ap.add_argument("--limit", type=int, default=0, help="max crops per folder (random subset; 0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    classes = set(args.classes) if args.classes else None
    if classes and classes - set(CLASS_FOLDERS):
        raise SystemExit(f"unknown classes {sorted(classes - set(CLASS_FOLDERS))}; known: {sorted(CLASS_FOLDERS)}")
    counts = export(args.manifest, args.out, set(args.forms), set(args.splits), classes, args.limit, args.seed)
    if not counts:
        raise SystemExit("no crops matched")
    total = Counter(counts)
    for folder, n in counts.items():
        print(f"{folder:28s} {n:6d}")
    print(f"{'total':28s} {sum(total.values()):6d}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
