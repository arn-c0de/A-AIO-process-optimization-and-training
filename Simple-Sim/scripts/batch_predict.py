#!/usr/bin/env python3
"""Batch predict on a dataset split and report accuracy/metrics + history compare.

This is an "evaluate with more context" script:
- runs predictions with a trained checkpoint
- compares to ground-truth labels.jsonl
- writes a timestamped report + per-sample predictions
- prints deltas vs previous reports for the same model stem

Example:
  .venv/bin/python scripts/batch_predict.py \\
    --model outputs/models/run_0001.pt \\
    --data outputs/sim_data/runs/run_0001 \\
    --split test
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.data_loader import ROIDataset
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.manifest import read_dataset_manifest
from simple_sim.model_bundle import bundle_checkpoint_path


@dataclass
class RunInfo:
    dataset_dir: Path
    split: str
    total_samples: int
    split_sizes: Dict[str, int]
    class_distribution: Dict[str, int]
    class_list: List[str]


def load_checkpoint_model(model_path: Path, device: torch.device) -> Tuple[torch.nn.Module, List[str], Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, checkpoint


def dataset_info(data_dir: Path) -> RunInfo:
    data_dir = Path(data_dir)
    meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)

    split_sizes: Dict[str, int] = {}
    for s in ["train", "val", "test"]:
        p = data_dir / "splits" / f"{s}.txt"
        if p.exists():
            split_sizes[s] = sum(1 for _ in p.open("r", encoding="utf-8") if _.strip())
        else:
            split_sizes[s] = 0

    class_dist: Dict[str, int] = {}
    for r in label_rows:
        class_dist[r.class_name] = class_dist.get(r.class_name, 0) + 1

    # split is filled later by caller
    return RunInfo(
        dataset_dir=data_dir,
        split="-",
        total_samples=len(meta_rows),
        split_sizes=split_sizes,
        class_distribution=class_dist,
        class_list=list(class_dist.keys()),
    )


def build_dataset(data_dir: Path, split: str, class_names: List[str], return_id: bool = True):
    if split in ("train", "val", "test"):
        return ROIDataset(data_dir, split, class_names, return_id=return_id)
    if split == "all":
        parts = [ROIDataset(data_dir, s, class_names, return_id=return_id) for s in ("train", "val", "test")]
        return ConcatDataset(parts)
    raise ValueError(f"Unknown split: {split}")


def _now_tag() -> str:
    # Include microseconds so repeated runs within the same second never collide.
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _unique_path(p: Path) -> Path:
    """Return a non-existing path by appending an incrementing suffix if needed."""
    p = Path(p)
    if not p.exists():
        return p
    stem = p.stem
    suf = p.suffix
    parent = p.parent
    for i in range(1, 10000):
        cand = parent / f"{stem}_{i:03d}{suf}"
        if not cand.exists():
            return cand
    # Extremely unlikely; fall back to ns timestamp.
    return parent / f"{stem}_{time.time_ns()}{suf}"


def load_previous_reports(search_dirs: List[Path], model_stem: str) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for d in search_dirs:
        if not d.exists():
            continue
        # Include:
        # - eval.py outputs: report_<stem>.json
        # - this script outputs: batch_report_<stem>_*.json
        for p in sorted(list(d.glob(f"report_{model_stem}.json")) + list(d.glob(f"batch_report_{model_stem}_*.json"))):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict) or "metrics" not in obj:
                continue
            obj["_path"] = str(p)
            reports.append(obj)
    return reports


def load_all_reports(search_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Load any eval/batch reports from the given directories."""
    reports: List[Dict[str, Any]] = []
    seen_paths = set()
    for d in search_dirs:
        if not d.exists():
            continue
        cand = []
        cand.extend(d.glob("report_*.json"))
        cand.extend(d.glob("batch_report_*.json"))
        for p in sorted(cand):
            sp = str(p)
            if sp in seen_paths:
                continue
            seen_paths.add(sp)
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(obj, dict) or "metrics" not in obj:
                continue
            obj["_path"] = sp
            reports.append(obj)
    return reports


def metric_value(report: Dict[str, Any], key: str, critical_class: Optional[str]) -> Optional[float]:
    """Extract a comparable scalar value from a report."""
    try:
        metrics = report["metrics"]
    except Exception:
        return None

    if key in ("accuracy", "macro_f1"):
        try:
            return float(metrics[key])
        except Exception:
            return None

    if key == "critical_fn_rate":
        if not critical_class:
            return None
        try:
            return float(metrics["critical_fn_rates"].get(critical_class))
        except Exception:
            return None

    return None


def parse_iso_ts(s: str) -> float:
    # Best-effort; unknown formats sort last.
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def _basename(p: Optional[str]) -> str:
    if not p:
        return "-"
    try:
        return Path(p).name
    except Exception:
        return str(p)


def best_report(reports: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    best = None
    best_v = None
    for r in reports:
        try:
            v = float(r["metrics"][key])
        except Exception:
            continue
        if best is None or v > float(best_v):
            best = r
            best_v = v
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch predict and evaluate on a dataset split")
    ap.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="Dataset dir (has meta.jsonl/labels.jsonl/splits/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to evaluate")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size (default: from checkpoint.eval.batch_size or 64)")
    ap.add_argument("--num-workers", type=int, default=None, help="Override num_workers (default: from checkpoint.eval.num_workers or 0)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick checks")
    ap.add_argument("--out-dir", default=None, help="Output dir for reports (default: outputs/models/history)")
    ap.add_argument("--save-preds", action="store_true", help="Write per-sample predictions JSONL")
    ap.add_argument("--topk", type=int, default=4, help="Top-k probabilities to store if --save-preds")
    ap.add_argument("--history-scope", default="model", choices=["model", "dataset", "all"],
                    help="Which previous reports to compare against (default: model)")
    ap.add_argument("--history-metric", default="macro_f1", choices=["accuracy", "macro_f1", "critical_fn_rate"],
                    help="Metric to rank reports by")
    ap.add_argument("--history-critical-class", default="MISALIGNED",
                    help="Used when --history-metric=critical_fn_rate (default: MISALIGNED)")
    ap.add_argument("--history-split", default="same", choices=["same", "any"],
                    help="Compare only same split as current or any split")
    ap.add_argument("--history-limit", type=int, default=10, help="How many history rows to print")
    ap.add_argument("--history-dirs", default=None,
                    help="Comma-separated list of dirs to search for reports (default: outputs/models and outputs/models/history)")
    args = ap.parse_args()

    requested_model_path = Path(args.model)
    data_dir = Path(args.data)
    device = torch.device(args.device)

    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    effective_model_path = requested_model_path
    dataset_profile_id: Optional[str] = None
    if requested_model_path.exists() and requested_model_path.is_dir():
        manifest_path = data_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        manifest = read_dataset_manifest(manifest_path)
        dataset_profile_id = manifest["component_profile"]["profile_id"]
        resolved = bundle_checkpoint_path(requested_model_path, dataset_profile_id, kind="best")
        if not resolved.exists():
            raise FileNotFoundError(
                f"Multi-model bundle has no checkpoint for profile '{dataset_profile_id}':\n"
                f"  bundle: {requested_model_path}\n"
                f"  expected: {resolved}"
            )
        effective_model_path = resolved

    model, class_names, checkpoint = load_checkpoint_model(effective_model_path, device)
    cfg = checkpoint.get("config", {}) or {}
    eval_cfg = (cfg.get("eval") or {}) if isinstance(cfg, dict) else {}

    batch_size = args.batch_size if args.batch_size is not None else int(eval_cfg.get("batch_size", 64))
    num_workers = args.num_workers if args.num_workers is not None else int(eval_cfg.get("num_workers", 0))
    critical_classes = eval_cfg.get("critical_classes")

    info = dataset_info(data_dir)
    info.split = args.split

    dataset_classes = info.class_list
    if set(dataset_classes) != set(class_names):
        raise ValueError(
            f"Model classes {class_names} do not match dataset classes {dataset_classes}. "
            "Train a checkpoint on this dataset before running batch_predict."
        )

    ds = build_dataset(data_dir, args.split, class_names, return_id=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true: List[int] = []
    y_pred: List[int] = []

    preds_rows: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    ema_ips = None

    progress = tqdm(loader, desc=f"Predict({args.split})")
    seen = 0
    for batch in progress:
        # ROIDataset(return_id=True) returns: (img, label, id, image_rel_path)
        images, labels, sample_ids, image_paths = batch

        bt0 = time.perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            pred_idx = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

        dt = max(1e-9, time.perf_counter() - bt0)
        bs = int(images.shape[0])
        ips = bs / dt
        ema_ips = ips if ema_ips is None else (0.9 * ema_ips + 0.1 * ips)
        progress.set_postfix({"img/s": f"{ema_ips:.1f}", "last_img": str(image_paths[-1]).split("/")[-1]})

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred_idx.detach().cpu().numpy().tolist())
        seen += bs

        if args.save_preds:
            topk = min(args.topk, len(class_names))
            top_vals, top_inds = torch.topk(probs, k=topk, dim=1)
            top_vals = top_vals.detach().cpu().numpy()
            top_inds = top_inds.detach().cpu().numpy()
            pred_idx_np = pred_idx.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            for i in range(bs):
                tid = str(sample_ids[i])
                gt_name = class_names[int(labels_np[i])]
                pred_name = class_names[int(pred_idx_np[i])]
                row = {
                    "id": tid,
                    "image_path": str(image_paths[i]),
                    "gt": gt_name,
                    "pred": pred_name,
                    "correct": bool(pred_name == gt_name),
                }
                row["topk"] = [
                    {"class": class_names[int(top_inds[i, j])], "prob": float(top_vals[i, j])}
                    for j in range(topk)
                ]
                preds_rows.append(row)

        if args.max_samples is not None and seen >= args.max_samples:
            break

    elapsed = max(1e-9, time.perf_counter() - t0)
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), class_names, critical_classes=critical_classes)

    print("\n" + "=" * 60)
    print("BATCH PREDICT REPORT")
    print("=" * 60)
    print(f"Model:       {requested_model_path}")
    if effective_model_path.resolve() != requested_model_path.resolve():
        print(f"Resolved:    {effective_model_path}")
    print(f"Dataset:     {data_dir}")
    print(f"Split:       {args.split}")
    print(f"Seen:        {seen}")
    print(f"Speed:       {seen / elapsed:.1f} images/s  (time: {elapsed:.1f}s)")
    print(f"Dataset size: total={info.total_samples} splits={info.split_sizes}")
    print("")
    print(format_metrics(metrics, class_names))

    # Write outputs
    out_dir = Path(args.out_dir) if args.out_dir else (requested_model_path.parent / "history")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = _now_tag()
    model_stem = requested_model_path.stem
    report_path = _unique_path(out_dir / f"batch_report_{model_stem}_{args.split}_{tag}.json")
    preds_path = _unique_path(out_dir / f"batch_preds_{model_stem}_{args.split}_{tag}.jsonl")

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "batch_predict",
        # Persist the *requested* model path for history grouping.
        # For bundles, this is the bundle directory; the resolved per-profile checkpoint is stored separately.
        "model_path": str(requested_model_path),
        "model_stem": model_stem,
        "resolved_model_path": str(effective_model_path) if str(effective_model_path) != str(requested_model_path) else None,
        "bundle_profile_id": dataset_profile_id,
        "dataset_path": str(data_dir),
        "split": args.split,
        "seen_samples": int(seen),
        "elapsed_s": float(elapsed),
        "img_per_s": float(seen / elapsed),
        "dataset_total_samples": int(info.total_samples),
        "dataset_split_sizes": info.split_sizes,
        "dataset_class_distribution": info.class_distribution,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_val_accuracy": float(checkpoint.get("val_accuracy", 0.0)),
        "checkpoint_val_f1": float(checkpoint.get("val_f1", 0.0)),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    if args.save_preds:
        with open(preds_path, "w", encoding="utf-8") as f:
            for row in preds_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"Predictions saved to: {preds_path}")

    # Compare to previous rounds (reports)
    if args.history_dirs:
        compare_dirs = [Path(p.strip()) for p in args.history_dirs.split(",") if p.strip()]
    else:
        compare_dirs = [requested_model_path.parent, requested_model_path.parent / "history"]

    all_prev = load_all_reports(compare_dirs)
    # Scope filters
    if args.history_scope == "model":
        all_prev = [
            r for r in all_prev
            if r.get("model_stem") == model_stem or _basename(r.get("model_path")) == requested_model_path.name
        ]
    elif args.history_scope == "dataset":
        all_prev = [r for r in all_prev if r.get("dataset_path") == str(data_dir)]

    if args.history_split == "same":
        all_prev = [r for r in all_prev if r.get("split") == args.split]

    # Remove the current report itself
    all_prev = [r for r in all_prev if r.get("_path") != str(report_path)]

    key = args.history_metric
    crit = args.history_critical_class if key == "critical_fn_rate" else None

    rows = []
    for r in all_prev:
        v = metric_value(r, key=key, critical_class=crit)
        if v is None:
            continue
        rows.append((v, r))

    # Sorting direction: higher is better except FN rate where lower is better.
    reverse = True
    if key == "critical_fn_rate":
        reverse = False
    rows.sort(key=lambda t: t[0], reverse=reverse)

    print("\n" + "=" * 60)
    title = f"HISTORY (scope={args.history_scope}, split={args.history_split}, metric={key}"
    if crit:
        title += f", class={crit}"
    title += ")"
    print(title)
    print("=" * 60)

    if not rows:
        print("No previous reports found for the chosen filters.")
        return

    limit = max(1, int(args.history_limit))
    show = rows[:limit]

    # Header
    print(f"{'rank':>4}  {'value':>8}  {'when':<19}  {'split':<5}  {'seen':>5}  {'ds_total':>7}  {'model':<12}  {'dataset':<12}  {'report'}")
    for i, (v, r) in enumerate(show, 1):
        when = r.get("timestamp") or "-"
        when = when[:19]
        split = str(r.get("split") or "-")
        seen_s = str(r.get("seen_samples") or r.get("test_size") or "-")
        ds_total = str(r.get("dataset_total_samples") or "-")
        mstem = str(r.get("model_stem") or _basename(r.get("model_path")))
        dsname = _basename(r.get("dataset_path"))
        path = str(r.get("_path") or "-")
        print(f"{i:>4}  {v:>8.4f}  {when:<19}  {split:<5}  {seen_s:>5}  {ds_total:>7}  {mstem:<12.12}  {dsname:<12.12}  {path}")


if __name__ == "__main__":
    main()
