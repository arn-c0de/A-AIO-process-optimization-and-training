#!/usr/bin/env python3
"""Sample a few images per class from a dataset split and verify predictions vs GT.

Example:
  .venv/bin/python scripts/sample_check.py \\
    --model outputs/models/run_0001.pt \\
    --data outputs/sim_data/runs/run_0001 \\
    --split test \\
    --per-class 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import UsageError, add_json_flag, run_cli
from simple_sim.data_loader import load_image_bgr, preprocess_bgr
from simple_sim.eval_utils import predict_probs
from simple_sim.models import load_checkpoint_model
from simple_sim.dataset_index import DatasetIndex
from simple_sim.schema import MetaRow


def choose_samples(
    data_dir: Path, class_names: list[str], split: str, per_class: int, seed: int
) -> tuple[list[tuple[str, str]], dict[str, MetaRow]]:
    """Pick up to *per_class* random sample IDs per class from *split*.

    Split membership comes from ``splits/<split>.txt`` (the same source as ``ROIDataset``), not from
    the ``/<split>/`` part of the sample id, which is wrong for merged or re-split datasets (plan C1).
    A missing split file selects nothing. Candidates keep the ``labels.jsonl`` order.

    Returns ``([(sample_id, gt_class), ...] sorted by id, id_to_meta)``.
    """
    index = DatasetIndex(data_dir)
    id_to_meta = dict(index.id_to_meta())
    id_to_label = {r.id: r.class_name for r in index.label_rows()}
    try:
        in_split = set(index.split_ids(split))
    except FileNotFoundError:
        in_split = set()

    candidates: dict[str, list[str]] = {c: [] for c in class_names}
    for sid, cls in id_to_label.items():
        if sid in in_split and cls in candidates and sid in id_to_meta:
            candidates[cls].append(sid)

    rng = random.Random(seed)
    chosen: list[tuple[str, str]] = []
    for cls in class_names:
        arr = candidates[cls]
        rng.shuffle(arr)
        chosen.extend((sid, cls) for sid in arr[:per_class])
    # Deterministic order for output readability.
    chosen.sort(key=lambda t: t[0])
    return chosen, id_to_meta


def sample_check(
    model_path: Path,
    data_dir: Path,
    *,
    split: str = "test",
    per_class: int = 5,
    seed: int = 42,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Predict a random per-class sample of *split* and compare with GT.

    Returns ``rows`` (dicts with id/img/gt/pred/ok), ``correct``, ``seen`` and ``accuracy``.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    device = torch.device(device)

    model, class_names, _ = load_checkpoint_model(model_path, device)
    chosen, id_to_meta = choose_samples(data_dir, class_names, split, int(per_class), seed)
    if not chosen:
        raise UsageError("No samples found for the requested split/classes.")

    print("Sample Check")
    print(f"  Model:   {model_path}")
    print(f"  Dataset: {data_dir}")
    print(f"  Split:   {split}")
    print(f"  Per cls: {per_class}")
    print(f"  Seed:    {seed}")
    print("")

    rows: list[dict[str, Any]] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(chosen), bs):
        batch = chosen[start:start + bs]
        x = torch.stack(
            [preprocess_bgr(load_image_bgr(data_dir / id_to_meta[sid].image_path)) for sid, _ in batch]
        ).to(device)
        pred_idx = predict_probs(model, x).argmax(axis=1)
        for (sid, gt), pi in zip(batch, pred_idx, strict=True):
            pred = class_names[int(pi)]
            rows.append({"id": sid, "img": Path(id_to_meta[sid].image_path).name, "gt": gt, "pred": pred, "ok": pred == gt})

    correct = sum(r["ok"] for r in rows)
    seen = len(rows)
    return {
        "model_path": str(model_path),
        "dataset_path": str(data_dir),
        "split": split,
        "rows": rows,
        "correct": correct,
        "seen": seen,
        "accuracy": correct / max(1, seen),
    }


def print_sample_check(result: dict[str, Any]) -> None:
    """Human-readable table + summary of :func:`sample_check`."""
    print(f"{'id':<34} {'img':<12} {'gt':<10} {'pred':<10} {'ok'}")
    for r in result["rows"]:
        print(f"{r['id']:<34} {r['img']:<12} {r['gt']:<10} {r['pred']:<10} {str(r['ok'])}")
    print("")
    print(f"Summary: {result['correct']}/{result['seen']} correct  (acc={result['accuracy']:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample-check predictions vs GT")
    ap.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="Dataset dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=5, help="How many samples per class")
    ap.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_json_flag(ap)
    args = ap.parse_args()

    def run() -> dict[str, Any]:
        result = sample_check(
            Path(args.model), Path(args.data), split=args.split, per_class=args.per_class,
            seed=args.seed, batch_size=args.batch_size, device=args.device,
        )
        print_sample_check(result)
        return result

    run_cli("sample_check", run, args.json)


if __name__ == "__main__":
    main()
