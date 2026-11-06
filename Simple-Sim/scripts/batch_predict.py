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
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.cli_utils import add_json_flag, emit_result
from simple_sim.data_loader import ROIDataset
from simple_sim.io_utils import append_jsonl, atomic_write_json, now_tag, unique_path
from simple_sim.manifest import read_dataset_manifest, resolve_dataset_profile
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.model_bundle import bundle_checkpoint_path
from simple_sim.models import load_checkpoint_model
from simple_sim.reports import load_reports, metric_value
from simple_sim.schema import LabelRow, MetaRow, read_jsonl

COMMAND = "batch_predict"


@dataclass
class RunInfo:
    dataset_dir: Path
    split: str
    total_samples: int
    split_sizes: dict[str, int]
    class_distribution: dict[str, int]
    class_list: list[str]


@dataclass
class PredictResult:
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    profile_gt: list[str] = field(default_factory=list)
    profile_pred: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0
    elapsed: float = 0.0


def dataset_info(data_dir: Path) -> RunInfo:
    data_dir = Path(data_dir)
    meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)

    split_sizes: dict[str, int] = {}
    for s in ["train", "val", "test"]:
        p = data_dir / "splits" / f"{s}.txt"
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                split_sizes[s] = sum(1 for line in fh if line.strip())
        else:
            split_sizes[s] = 0

    class_dist: dict[str, int] = {}
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


def build_dataset(data_dir: Path, split: str, class_names: list[str], return_id: bool = True) -> Dataset:
    if split in ("train", "val", "test"):
        return ROIDataset(data_dir, split, class_names, return_id=return_id)
    if split == "all":
        return ConcatDataset([ROIDataset(data_dir, s, class_names, return_id=return_id) for s in ("train", "val", "test")])
    raise ValueError(f"Unknown split: {split}")


def _basename(p: str | None) -> str:
    return Path(p).name if p else "-"


def resolve_model_path(requested: Path, data_dir: Path) -> tuple[Path, str | None]:
    """Return ``(checkpoint_path, bundle_profile_id)``; a bundle dir resolves via the dataset's profile."""
    if not requested.is_dir():
        return requested, None
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    profile_id, _ = resolve_dataset_profile(read_dataset_manifest(manifest_path))
    resolved = bundle_checkpoint_path(requested, profile_id, kind="best")
    if not resolved.exists():
        # Help users diagnose what's actually inside the bundle dir.
        avail = sorted(p.name for p in requested.glob("*.pt"))
        raise FileNotFoundError(
            f"Multi-model bundle has no checkpoint for profile '{profile_id}':\n"
            f"  bundle: {requested}\n"
            f"  expected: {resolved}\n"
            f"  available: {', '.join(avail[:12]) if avail else '(none)'}\n\n"
            f"To add it, train this dataset into the same bundle:\n"
            f"  ./.venv/bin/python scripts/train.py --data {data_dir} --out {requested}"
        )
    return resolved, profile_id


def _gt_profile_map(data_dir: Path) -> dict[str, str]:
    """Sample id -> GT profile_id from v2 labels (empty for legacy datasets)."""
    try:
        return {lr.id: lr.profile_id for lr in read_jsonl(data_dir / "labels.jsonl", LabelRow) if lr.profile_id}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) and ("cuda" in msg or "cudnn" in msg)


def predict_once(
    ds: Dataset,
    *,
    device: torch.device,
    model: torch.nn.Module,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    split: str,
    max_samples: int | None,
    save_preds: bool,
    topk: int,
    gt_profile_map: dict[str, str],
    profile_model: torch.nn.Module | None,
    profile_class_names: list[str],
    ckpt_profile_id: str,
) -> PredictResult:
    """One pass over *ds*; collects labels/preds, profile labels and (optionally) per-sample rows."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    res = PredictResult()
    t0 = time.perf_counter()
    ema_ips: float | None = None

    progress = tqdm(loader, desc=f"Predict({split})")
    for images, labels, sample_ids, image_paths in progress:
        if max_samples is not None:
            # Truncate the last batch so exactly --max-samples samples are evaluated.
            remaining = int(max_samples) - res.seen
            if remaining < int(images.shape[0]):
                images, labels = images[:remaining], labels[:remaining]
                sample_ids, image_paths = sample_ids[:remaining], image_paths[:remaining]

        bt0 = time.perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(images)
            pred_idx = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

        profile_pred_idx_np = None
        profile_probs_np = None
        if profile_model is not None:
            with torch.no_grad():
                profile_logits = profile_model(images)
                profile_probs_np = torch.softmax(profile_logits, dim=1).cpu().numpy()
                profile_pred_idx_np = torch.argmax(profile_logits, dim=1).cpu().numpy()

        bs = int(images.shape[0])
        ips = bs / max(1e-9, time.perf_counter() - bt0)
        ema_ips = ips if ema_ips is None else (0.9 * ema_ips + 0.1 * ips)
        progress.set_postfix({"img/s": f"{ema_ips:.1f}", "last_img": str(image_paths[-1]).split("/")[-1]})

        res.y_true.extend(labels.cpu().numpy().tolist())
        res.y_pred.extend(pred_idx.cpu().numpy().tolist())
        res.seen += bs

        def pred_profile(i: int) -> str:
            if profile_pred_idx_np is not None:
                return profile_class_names[int(profile_pred_idx_np[i])]
            return ckpt_profile_id

        # Profile metrics are collected independently of --save-preds.
        if profile_model is not None or ckpt_profile_id:
            for i in range(bs):
                gt_prof = str(gt_profile_map.get(str(sample_ids[i]), "") or "")
                if gt_prof:
                    res.profile_pred.append(pred_profile(i))
                    res.profile_gt.append(gt_prof)

        if save_preds:
            k = min(topk, len(class_names))
            top_vals, top_inds = (t.cpu().numpy() for t in torch.topk(probs, k=k, dim=1))
            pred_idx_np = pred_idx.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for i in range(bs):
                tid = str(sample_ids[i])
                gt_name = class_names[int(labels_np[i])]
                pred_name = class_names[int(pred_idx_np[i])]
                row: dict[str, Any] = {
                    "id": tid,
                    "image_path": str(image_paths[i]),
                    "gt": gt_name,
                    "pred": pred_name,
                    "correct": bool(pred_name == gt_name),
                    "topk": [{"class": class_names[int(top_inds[i, j])], "prob": float(top_vals[i, j])} for j in range(k)],
                }
                gt_prof = str(gt_profile_map.get(tid, "") or "")
                if gt_prof:
                    row["gt_profile"] = gt_prof
                if profile_model is not None or ckpt_profile_id:
                    pred_prof = pred_profile(i)
                    row["pred_profile"] = pred_prof
                    row["gt_profile"] = gt_prof
                    row["profile_correct"] = bool(pred_prof == gt_prof) if gt_prof else None
                    row["profile_confidence"] = (
                        float(profile_probs_np[i, int(profile_pred_idx_np[i])]) if profile_probs_np is not None else 1.0
                    )
                res.rows.append(row)

        if max_samples is not None and res.seen >= max_samples:
            break

    res.elapsed = max(1e-9, time.perf_counter() - t0)
    return res


def predict_with_oom_retry(device: torch.device, batch_size: int, **kwargs: Any) -> PredictResult:
    """Run :func:`predict_once`, halving the batch size on CUDA OOM down to 1."""
    bs = max(1, int(batch_size))
    while True:
        try:
            return predict_once(device=device, batch_size=bs, **kwargs)
        except RuntimeError as e:
            if device.type != "cuda" or not _is_cuda_oom(e):
                raise
            if bs == 1:
                raise RuntimeError(
                    "CUDA OOM even at batch_size=1. Keep GPU selected but reduce memory pressure "
                    "(close other GPU jobs, use a smaller model, or lower input size)."
                ) from e
            new_bs = max(1, bs // 2)
            print(f"[warn] CUDA OOM at batch_size={bs}. Retrying on GPU with batch_size={new_bs}.")
            bs = new_bs
            torch.cuda.empty_cache()


def profile_classification_metrics(gt: list[str], pred: list[str]) -> tuple[dict[str, Any], list[str]]:
    names = sorted(set(gt) | set(pred))
    idx = {n: i for i, n in enumerate(names)}
    m = compute_metrics(np.array([idx[n] for n in gt]), np.array([idx[n] for n in pred]), names, critical_classes=[])
    return m, names


def print_profile_metrics(profile_metrics: dict[str, Any], names: list[str]) -> None:
    print("")
    print("=" * 60)
    print("PROFILE CLASSIFICATION")
    print("=" * 60)
    print(f"Profile Accuracy: {profile_metrics['accuracy']:.4f}")
    print(f"Profile Macro F1: {profile_metrics['macro_f1']:.4f}")
    pc = profile_metrics.get("per_class") or {}
    if pc:
        print(f"{'Profile':<30} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
        for pn in names:
            m = pc.get(pn, {})
            print(f"{pn:<30} {m.get('precision',0):.4f}   {m.get('recall',0):.4f}   {m.get('f1',0):.4f}   {m.get('support',0):>5}")


def history_rows(
    compare_dirs: list[Path],
    *,
    scope: str,
    split_mode: str,
    metric: str,
    critical_class: str | None,
    model_path: Path,
    data_dir: Path,
    split: str,
    exclude_path: Path,
) -> list[tuple[float, dict[str, Any]]]:
    """Previous reports matching the filters as ``(value, report)``, best first (FN rate: lowest first)."""
    reports = load_reports(compare_dirs)
    if scope == "model":
        reports = [
            r for r in reports
            if r.get("model_stem") == model_path.stem or _basename(r.get("model_path")) == model_path.name
        ]
    elif scope == "dataset":
        reports = [r for r in reports if r.get("dataset_path") == str(data_dir)]
    if split_mode == "same":
        reports = [r for r in reports if r.get("split") == split]
    reports = [r for r in reports if r.get("_path") != str(exclude_path)]

    rows = [(v, r) for r in reports if (v := metric_value(r, metric, critical_class)) is not None]
    rows.sort(key=lambda t: t[0], reverse=(metric != "critical_fn_rate"))
    return rows


def print_history(rows: list[tuple[float, dict[str, Any]]], *, scope: str, split_mode: str, metric: str, crit: str | None, limit: int) -> None:
    print("\n" + "=" * 60)
    title = f"HISTORY (scope={scope}, split={split_mode}, metric={metric}"
    if crit:
        title += f", class={crit}"
    print(title + ")")
    print("=" * 60)

    if not rows:
        print("No previous reports found for the chosen filters.")
        return

    print(f"{'rank':>4}  {'value':>8}  {'when':<19}  {'split':<5}  {'seen':>5}  {'ds_total':>7}  {'model':<12}  {'dataset':<12}  {'report'}")
    for i, (v, r) in enumerate(rows[: max(1, int(limit))], 1):
        when = (r.get("timestamp") or "-")[:19]
        split = str(r.get("split") or "-")
        seen_s = str(r.get("seen_samples") or r.get("test_size") or "-")
        ds_total = str(r.get("dataset_total_samples") or "-")
        mstem = str(r.get("model_stem") or _basename(r.get("model_path")))
        dsname = _basename(r.get("dataset_path"))
        path = str(r.get("_path") or "-")
        print(f"{i:>4}  {v:>8.4f}  {when:<19}  {split:<5}  {seen_s:>5}  {ds_total:>7}  {mstem:<12.12}  {dsname:<12.12}  {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_model_path = Path(args.model)
    data_dir = Path(args.data)
    device = torch.device(args.device)

    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    effective_model_path, dataset_profile_id = resolve_model_path(requested_model_path, data_dir)

    model, class_names, checkpoint = load_checkpoint_model(effective_model_path, device)
    cfg = checkpoint.get("config", {}) or {}
    eval_cfg = (cfg.get("eval") or {}) if isinstance(cfg, dict) else {}
    component_profile = checkpoint.get("component_profile")
    ckpt_profile_id = str(component_profile.get("profile_id") or "") if isinstance(component_profile, dict) else ""

    batch_size = args.batch_size if args.batch_size is not None else int(eval_cfg.get("batch_size", 64))
    num_workers = args.num_workers if args.num_workers is not None else int(eval_cfg.get("num_workers", 0))
    critical_classes = eval_cfg.get("critical_classes")

    # GT profile mapping is built regardless of --profile-model so reports/preds carry GT profile metadata.
    gt_profile_map = _gt_profile_map(data_dir)

    # Profile classifier (optional two-stage)
    profile_model = None
    profile_class_names: list[str] = []
    if args.profile_model:
        profile_model_path = Path(args.profile_model)
        if not profile_model_path.exists():
            raise FileNotFoundError(f"Profile model not found: {profile_model_path}")
        profile_model, profile_class_names, _ = load_checkpoint_model(profile_model_path, device)

    info = dataset_info(data_dir)
    info.split = args.split

    dataset_classes = info.class_list
    if set(dataset_classes) != set(class_names):
        raise ValueError(
            f"Model classes {class_names} do not match dataset classes {dataset_classes}. "
            "Train a checkpoint on this dataset before running batch_predict."
        )

    pr = predict_with_oom_retry(
        device,
        batch_size,
        ds=build_dataset(data_dir, args.split, class_names, return_id=True),
        model=model,
        class_names=class_names,
        num_workers=num_workers,
        split=args.split,
        max_samples=args.max_samples,
        save_preds=bool(args.save_preds),
        topk=int(args.topk),
        gt_profile_map=gt_profile_map,
        profile_model=profile_model,
        profile_class_names=profile_class_names,
        ckpt_profile_id=ckpt_profile_id,
    )
    seen, elapsed = pr.seen, pr.elapsed

    metrics = compute_metrics(np.array(pr.y_true), np.array(pr.y_pred), class_names, critical_classes=critical_classes)

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

    # Profile metrics (when profile preds and GT labels were available)
    profile_metrics: dict[str, Any] | None = None
    if pr.profile_gt:
        profile_metrics, profile_names = profile_classification_metrics(pr.profile_gt, pr.profile_pred)
        print_profile_metrics(profile_metrics, profile_names)

    # Write outputs
    out_dir = Path(args.out_dir) if args.out_dir else (requested_model_path.parent / "history")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = now_tag()
    model_stem = requested_model_path.stem
    report_path = unique_path(out_dir / f"batch_report_{model_stem}_{args.split}_{tag}.json")
    preds_path = unique_path(out_dir / f"batch_preds_{model_stem}_{args.split}_{tag}.jsonl")

    report: dict[str, Any] = {
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
    if profile_metrics is not None:
        report["profile_accuracy"] = profile_metrics["accuracy"]
        report["profile_metrics"] = profile_metrics
    atomic_write_json(report_path, report)
    print(f"\nReport saved to: {report_path}")

    if args.save_preds:
        append_jsonl(preds_path, pr.rows)
        print(f"Predictions saved to: {preds_path}")

    # Compare to previous rounds (reports)
    if args.history_dirs:
        compare_dirs = [Path(p.strip()) for p in args.history_dirs.split(",") if p.strip()]
    else:
        compare_dirs = [requested_model_path.parent, requested_model_path.parent / "history"]
    crit = args.history_critical_class if args.history_metric == "critical_fn_rate" else None
    hist = history_rows(
        compare_dirs,
        scope=args.history_scope,
        split_mode=args.history_split,
        metric=args.history_metric,
        critical_class=crit,
        model_path=requested_model_path,
        data_dir=data_dir,
        split=args.split,
        exclude_path=report_path,
    )
    print_history(hist, scope=args.history_scope, split_mode=args.history_split, metric=args.history_metric, crit=crit, limit=args.history_limit)

    n_mismatch = int(sum(1 for t, p in zip(pr.y_true, pr.y_pred) if t != p))
    return {
        "report_path": str(report_path),
        "preds_path": str(preds_path) if args.save_preds else None,
        "model_path": str(requested_model_path),
        "resolved_model_path": report["resolved_model_path"],
        "dataset_path": str(data_dir),
        "split": args.split,
        "n_samples": int(seen),
        "n_mismatches": n_mismatch,
        "metrics": {k: metrics.get(k) for k in ("accuracy", "macro_f1", "critical_fn_rates")},
        "profile_accuracy": report.get("profile_accuracy"),
        "history_count": len(hist),
        "history_best": (
            {"value": hist[0][0], "report": hist[0][1].get("_path")} if hist else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--profile-model", default=None,
                    help="Optional path to profile classifier checkpoint (.pt) for two-stage evaluation")
    add_json_flag(ap)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (Exception, SystemExit) as exc:
        if not args.json:
            raise
        emit_result(COMMAND, ok=False, error=str(exc))
        sys.exit(1)
    if args.json:
        emit_result(COMMAND, result)


if __name__ == "__main__":
    main()
