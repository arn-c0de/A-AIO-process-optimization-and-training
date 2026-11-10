#!/usr/bin/env python3
"""Export the DeepVision *app* crops (phone / flatbed camera, judge labels) into the class folders of the
sim-to-real tools.

``tools/real_crops.py`` exports the QUINS scanner crops from the dataset manifest; this tool does the same
for the per-part crops the DeepVision inspection app cut from camera stills and that reviewers judged in
the app (``.../products/<product>/aoi/labels/<side>.ndjson``, one JSON line per review with ``board``,
``id``, ``label`` (``ok`` | ``fault`` | ``dnp`` | ...), ``fault``, ``size``, ``fieldMm``, ``centerMm``,
``crop`` (grey PNG) and ``cropColor`` (RGB PNG, both relative to the labels folder)). A part is judged
again on every run, so only the **newest review per (board, id)** counts (by ``at``, else file order);
retracted reviews and reviews without a colour crop are skipped.

Review ``label`` / ``fault`` -> class folder of ``tools/eval_real.py`` / ``tools/mix_real.py``
(:data:`FAULT_FOLDERS`; ``dnp``, ``unclear`` and ``unjudgeable`` are skipped):

===================  ======================  ============================================================
label / fault        folder                  meaning
===================  ======================  ============================================================
ok                   OK                      reviewed good part
fault / fehlt        MISSING                 part missing
fault / versetzt     MISALIGNED              part shifted
fault / verdreht     MISALIGNED@verdreht     part rotated (as in the DeepVision export contract, ``tools/export_deepvision.py``,
                                             and ``tools/real_crops.py``)
fault / grabstein    TOMBSTONE               tombstone
fault / loetbruecke  SOLDER_BRIDGE           solder bridge
fault / fremdkoerper DEFECT@fremdkoerper     foreign object (no Simple-Sim class, counts as defect)
fault / other        DEFECT@<fault>          beschaedigt, loetfehler, falsches-teil, ... (or ``DEFECT@unspecified``)
===================  ======================  ============================================================

With ``--pnp`` (the product's ``pnp.csv``: refdes, part, value, -, package, x mm, y mm, rotation, bottom
YES/NO) every crop gets the package of the nearest placement on its side (``--match-mm`` tolerance), so
``--packages CAPC1005 RESC1005`` exports one package form only and the summary counts crops per package.

Crops are copied unchanged as ``<board>_<id>.png``; only counts and aggregate statistics are printed /
written (``summary.json``), so the production images stay on the machine.

Usage:
    python tools/dv_crops.py --labels .../products/xpa/aoi/labels --side top --out outputs/dv_real/top \\
        --pnp .../products/xpa/pnp.csv --limit-ok 500 [--packages CAPC1005] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.io_utils import read_jsonl_dicts  # noqa: E402

#: review fault -> class folder (``label: fault``). Unlisted faults go to ``DEFECT@<fault>``.
FAULT_FOLDERS: dict[str, str] = {
    "fehlt": "MISSING",
    "versetzt": "MISALIGNED",
    "verdreht": "MISALIGNED@verdreht",
    "grabstein": "TOMBSTONE",
    "loetbruecke": "SOLDER_BRIDGE",
    "fremdkoerper": "DEFECT@fremdkoerper",
}
OK_FOLDER = "OK"
SIDES = ("top", "bottom")
#: ``pnp.csv`` columns (no header line).
PNP_COLUMNS = ("refdes", "part", "value", "unused", "package", "x_mm", "y_mm", "rotation_deg", "bottom")


def latest_reviews(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Newest review per ``(board, id)`` (by ``at``, else file order), retracted ones dropped."""
    latest: dict[tuple[str, str], tuple[str, int, dict[str, Any]]] = {}
    for i, row in enumerate(rows):
        key = (str(row.get("board") or row.get("run") or ""), str(row.get("id") or ""))
        stamp = (str(row.get("at") or ""), i)
        if key not in latest or stamp >= latest[key][:2]:
            latest[key] = (*stamp, row)
    return [row for _, _, row in latest.values() if not row.get("retracted")]


def folder_of(row: dict[str, Any]) -> str | None:
    """Class folder of a review, ``None`` for labels that are neither good nor a fault."""
    label = str(row.get("label") or "")
    if label == "ok":
        return OK_FOLDER
    if label != "fault":
        return None
    fault = str(row.get("fault") or "").strip()
    if not fault:
        return "DEFECT@unspecified"
    return FAULT_FOLDERS.get(fault, f"DEFECT@{fault}")


def read_pnp(path: Path) -> dict[str, list[tuple[float, float, str, str]]]:
    """Placements per side: ``(x_mm, y_mm, package, refdes)``."""
    out: dict[str, list[tuple[float, float, str, str]]] = {"top": [], "bottom": []}
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for rec in csv.reader(fh):
            if len(rec) < len(PNP_COLUMNS):
                continue
            row = dict(zip(PNP_COLUMNS, rec))
            try:
                x, y = float(row["x_mm"]), float(row["y_mm"])
            except ValueError:
                continue
            side = "bottom" if row["bottom"].strip().upper() in ("YES", "Y", "TRUE", "1", "BOTTOM") else "top"
            out[side].append((x, y, row["package"].strip(), row["refdes"].strip()))
    return out


def nearest_package(center: list[float] | None, placements: list[tuple[float, float, str, str]],
                    tolerance_mm: float) -> tuple[str, float]:
    """``(package, distance_mm)`` of the placement nearest to *center*; ``("unknown", inf)`` when none is
    within *tolerance_mm*."""
    if not center or len(center) < 2 or not placements:
        return "unknown", math.inf
    cx, cy = float(center[0]), float(center[1])
    best, pkg = math.inf, "unknown"
    for x, y, package, _ in placements:
        d = math.hypot(x - cx, y - cy)
        if d < best:
            best, pkg = d, package
    return (pkg, best) if best <= tolerance_mm else ("unknown", best)


def select(rows: list[dict[str, Any]], labels_dir: Path, *, packages: set[str] | None,
           placements: list[tuple[float, float, str, str]] | None, match_mm: float) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Rows per class folder (newest review per part, with a colour crop) and the skip reasons."""
    out: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter = Counter()
    for row in latest_reviews(rows):
        folder = folder_of(row)
        if folder is None:
            skipped[f"label:{row.get('label')}"] += 1
            continue
        crop = row.get("cropColor")
        if not crop or not (labels_dir / str(crop)).is_file():
            skipped["no colour crop"] += 1
            continue
        package, dist = ("unknown", math.inf) if placements is None else nearest_package(row.get("centerMm"), placements, match_mm)
        if packages is not None and package not in packages:
            skipped["other package"] += 1
            continue
        out.setdefault(folder, []).append({**row, "package": package, "match_mm": dist})
    return out, skipped


def sample(rows: list[dict[str, Any]], limit: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    idx = sorted(rng.choice(len(rows), limit, replace=False))
    return [rows[i] for i in idx]


def _percentiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [round(float(v), 3) for v in np.percentile(values, (10, 50, 90))]


def summarize(selected: dict[str, list[dict[str, Any]]], skipped: Counter, side: str) -> dict[str, Any]:
    """Counts per folder and package plus crop size / field of view statistics (no image content)."""
    sizes = [float(r["size"]) for rows in selected.values() for r in rows if r.get("size")]
    fields = [float(r["fieldMm"]) for rows in selected.values() for r in rows if r.get("fieldMm")]
    px_per_mm = [float(r["size"]) / float(r["fieldMm"]) for rows in selected.values() for r in rows
                 if r.get("size") and r.get("fieldMm")]
    packages: Counter = Counter(r["package"] for rows in selected.values() for r in rows)
    per_package: dict[str, Any] = {}
    for pkg in packages:
        rows = [r for folder_rows in selected.values() for r in folder_rows if r["package"] == pkg]
        per_package[pkg] = {"crops": len(rows), "size": _percentiles([float(r["size"]) for r in rows if r.get("size")]),
                            "fieldMm": _percentiles([float(r["fieldMm"]) for r in rows if r.get("fieldMm")])}
    return {"side": side, "folders": {k: len(v) for k, v in sorted(selected.items())}, "skipped": dict(skipped),
            "size_px": _percentiles(sizes), "fieldMm": _percentiles(fields), "px_per_mm": _percentiles(px_per_mm),
            "packages": dict(packages.most_common()), "per_package": per_package}


def crop_stem(row: dict[str, Any]) -> str:
    """File stem of a review's crop: ``<board>_<id>`` (``run`` when the row has no ``board``, as in :func:`latest_reviews`)."""
    return f"{row.get('board') or row.get('run') or ''}_{row.get('id') or ''}"


def remove_stale(out: Path, stems: set[str], keep: set[Path]) -> int:
    """Delete crops of an earlier export that this one no longer writes; returns their number.

    Only files named after a review of the label file (*stems*) in the class folders of *out* are
    touched. Without this a crop whose review changed (ok -> fault, another fault, retracted) or
    whose folder was renamed stayed in its old folder too, i.e. under two labels for
    ``eval-real`` / ``mix-real``. Folders that end up empty are removed.
    """
    removed = 0
    if not out.is_dir():
        return removed
    for folder in sorted(p for p in out.iterdir() if p.is_dir()):
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            if path.stem in stems and path not in keep:
                path.unlink()
                removed += 1
        if removed and not any(folder.iterdir()):
            folder.rmdir()
    return removed


def export(labels_dir: Path, side: str, out: Path, *, pnp: Path | None, packages: set[str] | None, limit_ok: int,
           limit: int, seed: int, match_mm: float, dry_run: bool) -> dict[str, Any]:
    """Copy the selected crops into ``out/<folder>/<board>_<id>.png`` and write ``out/summary.json``."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    rows = read_jsonl_dicts(labels_dir / f"{side}.ndjson")
    placements = read_pnp(pnp)[side] if pnp is not None else None
    selected, skipped = select(rows, labels_dir, packages=packages, placements=placements, match_mm=match_mm)
    rng = np.random.default_rng(seed)
    chosen: dict[str, list[dict[str, Any]]] = {}
    for folder, folder_rows in sorted(selected.items()):
        cap = limit_ok if folder == OK_FOLDER else limit
        chosen[folder] = sample(folder_rows, cap, rng)
    summary = summarize(chosen, skipped, side)
    summary["available"] = {k: len(v) for k, v in sorted(selected.items())}
    summary["dry_run"] = bool(dry_run)
    if not dry_run:
        targets: dict[Path, Path] = {}
        for folder, folder_rows in chosen.items():
            for row in folder_rows:
                src = labels_dir / str(row["cropColor"])
                targets[out / folder / f"{crop_stem(row)}{src.suffix.lower()}"] = src
        summary["removed_stale"] = remove_stale(out, {crop_stem(r) for r in rows}, set(targets))
        for dest, src in targets.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"{'folder':28s} {'chosen':>7s} {'available':>10s}"]
    for folder, n in summary["folders"].items():
        lines.append(f"{folder:28s} {n:7d} {summary['available'].get(folder, n):10d}")
    lines.append(f"{'total':28s} {sum(summary['folders'].values()):7d}")
    lines.append(f"skipped: {summary['skipped']}")
    lines.append(f"size px (p10/p50/p90): {summary['size_px']}  fieldMm: {summary['fieldMm']}  px/mm: {summary['px_per_mm']}")
    lines.append("packages: " + ", ".join(f"{k} {v}" for k, v in summary["packages"].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True, help="DeepVision labels folder (holds <side>.ndjson and the crops)")
    ap.add_argument("--side", required=True, choices=SIDES)
    ap.add_argument("--out", type=Path, required=True, help="target folder (one sub-folder per class)")
    ap.add_argument("--pnp", type=Path, help="product pnp.csv: assigns every crop the package of the nearest placement")
    ap.add_argument("--packages", nargs="*", help="keep only these packages (needs --pnp), e.g. CAPC1005 RESC1005")
    ap.add_argument("--match-mm", type=float, default=0.8, help="max distance crop centre <-> placement (default 0.8 mm)")
    ap.add_argument("--limit-ok", type=int, default=500, help="max OK crops (random subset; 0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="max crops per defect folder (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print the summary, copy nothing")
    args = ap.parse_args(argv)
    if args.packages and args.pnp is None:
        raise SystemExit("--packages needs --pnp")
    if not (args.labels / f"{args.side}.ndjson").is_file():
        raise SystemExit(f"no {args.side}.ndjson in {args.labels}")
    summary = export(args.labels, args.side, args.out, pnp=args.pnp, packages=set(args.packages) if args.packages else None,
                     limit_ok=args.limit_ok, limit=args.limit, seed=args.seed, match_mm=args.match_mm, dry_run=args.dry_run)
    if not summary["folders"]:
        raise SystemExit("no crops matched")
    print(format_summary(summary))
    if not args.dry_run:
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
