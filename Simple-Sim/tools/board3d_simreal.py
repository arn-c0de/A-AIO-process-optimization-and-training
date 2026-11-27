#!/usr/bin/env python3
"""Quick sim-to-real probe for a board3d defect dataset: train a small classifier on the renders only,
score it on real crops.

Answers "does this dataset teach anything that holds on real images?" in a few minutes, so renderer and
dataset changes can be compared by numbers. A ResNet18 (ImageNet weights) is trained at ``--size`` px on
``labels.jsonl`` of a ``board3d_dataset.py`` output; the validation split goes by sheet (render), so
neighbouring crops of one render never land on both sides. ``--real-ok`` adds the real crop stored with
each row (``real``) as one more OK sample.

The real set uses the ``eval_real.py`` layout: one sub-folder per class (``OK``, ``MISSING``, ...),
anything after ``@`` is a sub-group scored as the class. Reported: OK specificity, defect recall (any
defect class predicted), balanced accuracy and per class the flagged and exact-class rates.

Usage:
    python tools/board3d_simreal.py --data outputs/board3d/<p>/top/dataset/<run> --real <real dir> \
        [--seed 0 --epochs 15 --real-ok --dump pred.jsonl --json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.406, 0.456, 0.485]) * 255        # ImageNet, BGR order
STD = np.array([0.225, 0.224, 0.229]) * 255


def load(path: Path, size: int) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA if im.shape[0] > size else cv2.INTER_CUBIC)


def split_by_sheet(rows: list[dict], frac: float = 1 / 6, seed: int = 0) -> np.ndarray:
    """True for rows of the validation sheets."""
    sheets = sorted({r["sheet"] for r in rows})
    random.Random(seed).shuffle(sheets)
    val = set(sheets[: max(1, int(len(sheets) * frac))])
    return np.array([r["sheet"] in val for r in rows])


def with_real_ok(rows: list[dict]) -> list[dict]:
    """The real image of every crop's place as one more OK sample (the part is fine there in reality)."""
    have = {r["image"] for r in rows}
    return rows + [{**r, "image": r["real"], "label": "OK"} for r in rows if r.get("real") and r["real"] not in have]


def real_folders(real_dir: Path) -> list[tuple[str, list[Path]]]:
    out = []
    for d in sorted(p for p in real_dir.iterdir() if p.is_dir()):
        ims = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if ims:
            out.append((d.name.split("@")[0], ims))
    return out


def score(per_image: list[tuple[str, str]], ok: str = "OK") -> dict:
    """per_image: (true class, predicted class). Binary inspection view plus per-class rates."""
    per: dict[str, dict] = {}
    for true, pred in per_image:
        e = per.setdefault(true, {"n": 0, "flagged": 0, "exact": 0})
        e["n"] += 1
        e["flagged"] += pred != ok
        e["exact"] += pred == true
    res: dict = {}
    spec = rec = None
    if ok in per:
        spec = 1 - per[ok]["flagged"] / per[ok]["n"]
        res["realOkSpecificity"] = round(spec, 3)
    dn = sum(v["n"] for k, v in per.items() if k != ok)
    if dn:
        rec = sum(v["flagged"] for k, v in per.items() if k != ok) / dn
        res["realDefectRecall"] = round(rec, 3)
    if spec is not None and rec is not None:
        res["balancedAcc"] = round((spec + rec) / 2, 3)
    res["perClass"] = {k: {"n": v["n"], "flagged": round(v["flagged"] / v["n"], 3), "exact": round(v["exact"] / v["n"], 3)}
                       for k, v in per.items()}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="board3d dataset dir with labels.jsonl")
    ap.add_argument("--real", type=Path, required=True, help="real crops, one sub-folder per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--real-ok", action="store_true", help="add each row's real crop as an OK sample")
    ap.add_argument("--dump", type=Path, help="append one JSON line per real image (file, true, pred)")
    ap.add_argument("--json", type=Path, help="also write the result here")
    a = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    import torchvision

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in (a.data / "labels.jsonl").read_text().splitlines() if line.strip()]
    if a.real_ok:
        rows = with_real_ok(rows)
    classes = sorted({r["label"] for r in rows})
    cid = {c: i for i, c in enumerate(classes)}
    X = np.stack([load(a.data / r["image"], a.size) for r in rows])
    Y = np.array([cid[r["label"]] for r in rows])
    va = split_by_sheet(rows)

    def tens(x):
        return torch.tensor(((x.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2), dtype=torch.float32)

    def augment(batch):
        out = []
        for im in batch:
            im = np.rot90(im, random.randrange(4))
            if random.random() < 0.5:
                im = im[:, ::-1]
            im = im.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-12, 12)      # exposure
            out.append(np.clip(cv2.GaussianBlur(np.clip(im, 0, 255), (0, 0), random.uniform(0.1, 1.0)), 0, 255))
        return np.stack(out)

    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    counts = Counter(Y[~va].tolist())
    w = torch.tensor([len(Y[~va]) / (len(classes) * max(counts[i], 1)) for i in range(len(classes))],
                     dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    tr = np.nonzero(~va)[0]
    for _ in range(a.epochs):
        m.train()
        np.random.shuffle(tr)
        for i in range(0, len(tr), 64):
            b = tr[i:i + 64]
            opt.zero_grad()
            lossf(m(tens(augment(X[b])).to(dev)), torch.tensor(Y[b], device=dev)).backward()
            opt.step()
        sched.step()

    @torch.no_grad()
    def predict(x):
        m.eval()
        out = []
        for i in range(0, len(x), 256):
            xb = tens(x[i:i + 256]).to(dev)
            out.append((sum(torch.softmax(m(torch.rot90(xb, k, (2, 3))), 1) for k in range(4)) / 4).cpu().numpy())
        return np.concatenate(out)

    res = {"data": str(a.data), "seed": a.seed, "realOk": a.real_ok, "classes": classes,
           "train": int((~va).sum()), "val": int(va.sum()),
           "valAcc": round(float((predict(X[va]).argmax(1) == Y[va]).mean()), 3)}
    per_image = []
    for cls, ims in real_folders(a.real):
        pred = predict(np.stack([load(p, a.size) for p in ims])).argmax(1)
        per_image += [(cls, classes[k]) for k in pred]
        if a.dump:
            with a.dump.open("a") as fh:
                for p, k in zip(ims, pred):
                    fh.write(json.dumps({"file": p.name, "true": cls, "pred": classes[k]}) + "\n")
    res.update(score(per_image))
    print(json.dumps(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
