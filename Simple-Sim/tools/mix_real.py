#!/usr/bin/env python3
"""Add real camera crops to a generated dataset (real good parts / empty pads next to synthetic defects).

The real crops are sorted into one sub-folder per class (``OK``, ``MISSING``, ...; anything after
``@`` is a sub-group, ``MISSING@empty_pad`` counts as ``MISSING``), like ``tools/eval_real.py``
expects. Each crop is resized to the dataset's ROI size (cubic for small crops, as the inspection
pipeline does) and written to ``images/real/``; ``meta.jsonl``, ``labels.jsonl``, the split file and
``dataset_manifest.json`` (``dataset_stats`` and a ``real_crops`` history entry) are updated.

Real rows use ``domain: real`` (``--domain``), ``render_backend: real``, ``seed: 0``, the dataset's
median nominal geometry (the true one is unknown), zero defect offsets, neutral augment values and
``render_meta: {real: true, source: <original file>, group: <sub-folder>}``.

Why: on the DeepVision xpa QUINS crops, models trained on synthetic images alone reject every real good
part, and real crops in only one class teach "real = that class". Real good parts *and* real empty
pads (both plentiful in production) fixed both (docs/guides/PHOTOREAL_RENDERING.md, "Classifier test
on real crops"). Keep the crops you evaluate on (``tools/eval_real.py``) out of the mix.

Usage:
    python tools/mix_real.py --data data/run_0402_c_quins --real real_train/ [--limit 200] [--split train]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.classes import VALID_CLASSES  # noqa: E402
from simple_sim.io_utils import load_yaml, now_iso  # noqa: E402
from simple_sim.manifest import record_added_samples  # noqa: E402
from simple_sim.schema import LabelRow, MetaRow, append_jsonl_rows, read_jsonl  # noqa: E402
from tools.eval_real import class_folders  # noqa: E402

SPLITS = ("train", "val", "test")
_NO_DEFECT = {"shift_x": 0.0, "shift_y": 0.0, "rotation_deg": 0.0, "tilt_deg": 0.0}
_NEUTRAL_AUGMENT = {"blur_sigma": 0.0, "noise_stddev": 0.0, "brightness_factor": 1.0, "contrast_factor": 1.0,
                    "rotation_deg": 0.0}


def dataset_info(data: Path) -> dict[str, Any]:
    """ROI size, run id, footprint, label schema and known classes/ids of a generated dataset."""
    cfg = load_yaml(data / "config.yaml") or {}
    roi = cfg.get("roi") or {}
    metas = read_jsonl(data / "meta.jsonl", MetaRow)
    labels = read_jsonl(data / "labels.jsonl", LabelRow)
    if not metas or not labels:
        raise SystemExit(f"{data} has no samples (meta.jsonl / labels.jsonl)")
    return {"size": (int(roi.get("width_px", 256)), int(roi.get("height_px", 256))),
            "run_id": metas[0].run_id, "footprint": metas[0].footprint,
            "label_schema": labels[0].schema_version, "profile_id": labels[0].profile_id,
            "classes": set((cfg.get("classes") or {}).keys()) or {r.class_name for r in labels},
            "ids": {m.id for m in metas},
            # next free number in images/real/ (one series for every domain and split, so files never collide)
            "real_next": 1 + max((int(Path(m.image_path).stem) for m in metas
                                  if m.image_path.startswith("images/real/") and Path(m.image_path).stem.isdigit()), default=-1),
            "nominal": {k: float(np.median([m.nominal[k] for m in metas if k in m.nominal]))
                        for k in metas[0].nominal}}


def mix_real(data: Path, real: Path, *, split: str = "train", limit: int = 200, domain: str = "real",
             seed: int = 0) -> dict[str, int]:
    """Add up to *limit* random crops (0 = all) per class sub-folder of *real* to *split* of the dataset *data*.

    Returns the number of crops added per class. Raises ``SystemExit`` for an unknown class or split.
    """
    if split not in SPLITS:
        raise SystemExit(f"--split must be one of {SPLITS}")
    info = dataset_info(data)
    rng = np.random.default_rng(seed)
    # Check every folder before the first image is written: an invalid later folder used to abort
    # with the earlier folders' images already in images/real/ (orphans without rows).
    folders = list(class_folders(real))
    for group, cls, _files in folders:
        if cls not in VALID_CLASSES:
            raise SystemExit(f"unknown class folder {group!r} (classes: {sorted(VALID_CLASSES)})")
        if cls not in info["classes"]:
            raise SystemExit(f"class {cls!r} is not trained in {data} (classes: {sorted(info['classes'])})")
    out_dir = data / "images" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = int(info["real_next"])
    metas, labels, added = [], [], Counter()
    for group, cls, files in folders:
        if 0 < limit < len(files):  # limit <= 0: all, as in real_crops / dv_crops
            files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Failed to read {path}")
            n = start + len(metas)
            sample_id = f"real/{domain}/{split}/{n:06d}"
            rel = f"images/real/{n:06d}.png"
            interp = cv2.INTER_CUBIC if img.shape[1] < info["size"][0] else cv2.INTER_AREA
            if not cv2.imwrite(str(data / rel), cv2.resize(img, info["size"], interpolation=interp)):
                raise OSError(f"Failed to write {data / rel}")
            metas.append(MetaRow(schema_version=2, id=sample_id, run_id=info["run_id"], domain=domain, split=split,
                                 seed=0, image_path=rel, render_backend="real", footprint=info["footprint"],
                                 nominal=dict(info["nominal"]), defect={**_NO_DEFECT, "type": cls},
                                 augment=dict(_NEUTRAL_AUGMENT),
                                 render_meta={"real": True, "source": str(path), "group": group}))
            labels.append(LabelRow(schema_version=info["label_schema"], id=sample_id, class_name=cls,
                                   profile_id=info["profile_id"]))
            added[cls] += 1
    if not metas:
        return {}  # nothing to add: no empty manifest history entry
    append_jsonl_rows(data / "meta.jsonl", metas)
    append_jsonl_rows(data / "labels.jsonl", labels)
    with open(data / "splits" / f"{split}.txt", "a", encoding="utf-8") as fh:
        fh.writelines(m.id + "\n" for m in metas)
    _update_manifest(data, real, split, domain, added)
    return dict(added)


def _update_manifest(data: Path, real: Path, split: str, domain: str, added: Counter) -> None:
    record_added_samples(data / "dataset_manifest.json", split=split, classes=dict(added), log_key="real_crops",
                         log_entry={"added_at": now_iso(), "source": str(real), "split": split, "domain": domain,
                                    "classes": dict(added)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="generated dataset directory")
    ap.add_argument("--real", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--limit", type=int, default=200, help="max crops per sub-folder (random subset; 0 = all)")
    ap.add_argument("--domain", default="real")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    added = mix_real(args.data, args.real, split=args.split, limit=args.limit, domain=args.domain, seed=args.seed)
    print(f"added {sum(added.values())} real crops to {args.data} ({args.split}): {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
