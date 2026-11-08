"""Batch prediction on a dataset split with report, per-sample predictions and history (plan L3).

:func:`run_batch_predict` takes a :class:`BatchPredictOptions` (the ``batch-predict`` command
builds it from its arguments), runs the model over the split with CUDA-OOM batch halving
(:func:`simple_sim.eval_utils.oom_retry`), optionally a profile classifier for two-stage
evaluation, writes ``history/batch_report_<stem>_<split>_<tag>.json`` (+ ``batch_preds_*.jsonl``)
and ranks it against previous reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from simple_sim.data_loader import ROIDataset
from simple_sim.dataset_index import DatasetIndex
from simple_sim.devices import resolve_device
from simple_sim.eval_utils import ThroughputMeter, dataset_profile, oom_retry
from simple_sim.evaluation import write_report
from simple_sim.io_utils import append_jsonl, now_iso, now_tag, unique_path
from simple_sim.metrics import compute_metrics, format_metrics
from simple_sim.model_bundle import resolve_bundle_checkpoint
from simple_sim.models import load_checkpoint_model
from simple_sim.reports import load_reports, metric_value, report_path

__all__ = [
    "BatchPredictOptions",
    "HISTORY_METRICS",
    "HISTORY_SCOPES",
    "PredictResult",
    "RunInfo",
    "SPLITS",
    "build_dataset",
    "dataset_info",
    "history_rows",
    "predict_once",
    "resolve_model_path",
    "run_batch_predict",
    "train_hint",
]

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test", "all")
HISTORY_SCOPES = ("model", "dataset", "all")
HISTORY_METRICS = ("accuracy", "macro_f1", "critical_fn_rate")

_RULE = "=" * 60


@dataclass(frozen=True, slots=True)
class BatchPredictOptions:
    """Inputs of :func:`run_batch_predict` (defaults = the ``batch-predict`` command defaults).

    ``batch_size`` / ``num_workers`` default to the checkpoint's ``eval`` section (64 / 0);
    ``out_dir`` defaults to ``<model dir>/history``; ``history_dirs`` to the model dir and its
    ``history`` subdir.
    """

    model: Path
    data: Path
    split: str = "test"
    device: str | torch.device = "cpu"
    batch_size: int | None = None
    num_workers: int | None = None
    max_samples: int | None = None
    out_dir: Path | None = None
    save_preds: bool = False
    topk: int = 4
    history_scope: str = "model"
    history_metric: str = "macro_f1"
    history_critical_class: str = "MISALIGNED"
    history_split: str = "same"
    history_limit: int = 10
    history_dirs: tuple[Path, ...] = ()
    profile_model: Path | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if self.history_scope not in HISTORY_SCOPES:
            raise ValueError(f"history_scope must be one of {HISTORY_SCOPES}, got {self.history_scope!r}")
        if self.history_metric not in HISTORY_METRICS:
            raise ValueError(f"history_metric must be one of {HISTORY_METRICS}, got {self.history_metric!r}")
        if self.history_split not in ("same", "any"):
            raise ValueError(f"history_split must be 'same' or 'any', got {self.history_split!r}")


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
    """Sizes and class distribution of *data_dir* (via the shared :class:`DatasetIndex`)."""
    data_dir = Path(data_dir)
    index = DatasetIndex(data_dir)
    meta_rows = index.meta_rows()
    index.label_rows()  # validated (raises) before the split files are looked at, as before
    split_sizes = index.split_sizes()
    class_dist = index.class_distribution()

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


def train_hint(data_dir: Path, bundle: Path) -> str:
    """The command that trains *data_dir* into *bundle* (shown when the bundle lacks the profile)."""
    return f"  python -m simple_sim train --data {data_dir} --out {bundle}"


def resolve_model_path(requested: Path, data_dir: Path) -> tuple[Path, str | None]:
    """Return ``(checkpoint_path, bundle_profile_id)``; a bundle dir resolves via the dataset's profile."""
    requested, data_dir = Path(requested), Path(data_dir)
    if not requested.is_dir():
        return requested, None
    profile_id, _ = dataset_profile(data_dir)
    # The error lists what is actually inside the bundle dir.
    resolved = resolve_bundle_checkpoint(
        requested, profile_id, show_available=True,
        hint="\nTo add it, train this dataset into the same bundle:\n" + train_hint(data_dir, requested),
    )
    return resolved, profile_id


def _gt_profile_map(data_dir: Path) -> dict[str, str]:
    """Sample id -> GT profile_id from v2 labels (empty for legacy datasets)."""
    try:
        return DatasetIndex(data_dir).profile_map()
    except (OSError, ValueError, TypeError, KeyError):
        return {}


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
    meter = ThroughputMeter()

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
        ema_ips = meter.update(bs, time.perf_counter() - bt0)
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


def profile_classification_metrics(gt: list[str], pred: list[str]) -> tuple[dict[str, Any], list[str]]:
    names = sorted(set(gt) | set(pred))
    idx = {n: i for i, n in enumerate(names)}
    m = compute_metrics(np.array([idx[n] for n in gt]), np.array([idx[n] for n in pred]), names, critical_classes=[])
    return m, names


def format_profile_metrics(profile_metrics: dict[str, Any], names: list[str]) -> str:
    lines = ["", _RULE, "PROFILE CLASSIFICATION", _RULE,
             f"Profile Accuracy: {profile_metrics['accuracy']:.4f}",
             f"Profile Macro F1: {profile_metrics['macro_f1']:.4f}"]
    pc = profile_metrics.get("per_class") or {}
    if pc:
        lines.append(f"{'Profile':<30} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
        for pn in names:
            m = pc.get(pn, {})
            lines.append(f"{pn:<30} {m.get('precision',0):.4f}   {m.get('recall',0):.4f}   {m.get('f1',0):.4f}   {m.get('support',0):>5}")
    return "\n".join(lines)


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
    reports = [r for r in reports if report_path(r) != str(exclude_path)]

    rows = [(v, r) for r in reports if (v := metric_value(r, metric, critical_class)) is not None]
    rows.sort(key=lambda t: t[0], reverse=(metric != "critical_fn_rate"))
    return rows


def format_history(rows: list[tuple[float, dict[str, Any]]], *, scope: str, split_mode: str, metric: str, crit: str | None, limit: int) -> str:
    title = f"HISTORY (scope={scope}, split={split_mode}, metric={metric}"
    if crit:
        title += f", class={crit}"
    lines = ["\n" + _RULE, title + ")", _RULE]
    if not rows:
        lines.append("No previous reports found for the chosen filters.")
        return "\n".join(lines)

    lines.append(f"{'rank':>4}  {'value':>8}  {'when':<19}  {'split':<5}  {'seen':>5}  {'ds_total':>7}  {'model':<12}  {'dataset':<12}  {'report'}")
    for i, (v, r) in enumerate(rows[: max(1, int(limit))], 1):
        when = (r.get("timestamp") or "-")[:19]
        split = str(r.get("split") or "-")
        seen_s = str(r.get("seen_samples") or r.get("test_size") or "-")
        ds_total = str(r.get("dataset_total_samples") or "-")
        mstem = str(r.get("model_stem") or _basename(r.get("model_path")))
        dsname = _basename(r.get("dataset_path"))
        path = str(report_path(r) or "-")
        lines.append(f"{i:>4}  {v:>8.4f}  {when:<19}  {split:<5}  {seen_s:>5}  {ds_total:>7}  {mstem:<12.12}  {dsname:<12.12}  {path}")
    return "\n".join(lines)


def run_batch_predict(opts: BatchPredictOptions) -> dict[str, Any]:
    """Predict *opts.split* of *opts.data* with *opts.model*; write the report and compare history.

    Returns the ``--json`` payload (report / preds paths, metrics, history best, ...).
    """
    requested_model_path = Path(opts.model)
    data_dir = Path(opts.data)
    device = resolve_device(opts.device)

    # Multi-model bundle support: --model can be a directory containing per-profile checkpoints.
    effective_model_path, dataset_profile_id = resolve_model_path(requested_model_path, data_dir)

    model, class_names, checkpoint = load_checkpoint_model(effective_model_path, device)
    cfg = checkpoint.get("config", {}) or {}
    eval_cfg = (cfg.get("eval") or {}) if isinstance(cfg, dict) else {}
    component_profile = checkpoint.get("component_profile")
    ckpt_profile_id = str(component_profile.get("profile_id") or "") if isinstance(component_profile, dict) else ""

    batch_size = opts.batch_size if opts.batch_size is not None else int(eval_cfg.get("batch_size", 64))
    num_workers = opts.num_workers if opts.num_workers is not None else int(eval_cfg.get("num_workers", 0))
    critical_classes = eval_cfg.get("critical_classes")

    # GT profile mapping is built regardless of --profile-model so reports/preds carry GT profile metadata.
    gt_profile_map = _gt_profile_map(data_dir)

    # Profile classifier (optional two-stage)
    profile_model = None
    profile_class_names: list[str] = []
    if opts.profile_model:
        profile_model_path = Path(opts.profile_model)
        if not profile_model_path.exists():
            raise FileNotFoundError(f"Profile model not found: {profile_model_path}")
        profile_model, profile_class_names, _ = load_checkpoint_model(profile_model_path, device)

    info = dataset_info(data_dir)
    info.split = opts.split

    dataset_classes = info.class_list
    unknown = sorted(set(dataset_classes) - set(class_names))
    if unknown:
        raise ValueError(
            f"Model classes {class_names} do not match dataset classes {dataset_classes}: "
            f"the dataset has labels the model was not trained on ({unknown}). "
            "Train a checkpoint on this dataset before running batch_predict."
        )
    absent = sorted(set(class_names) - set(dataset_classes))
    if absent:
        # A configured class without samples in this dataset is fine; it just has support 0.
        log.warning(f"Model classes without samples in this dataset: {absent}")

    ds = build_dataset(data_dir, opts.split, class_names, return_id=True)
    pr = oom_retry(
        lambda bs: predict_once(
            ds,
            device=device,
            batch_size=bs,
            model=model,
            class_names=class_names,
            num_workers=num_workers,
            split=opts.split,
            max_samples=opts.max_samples,
            save_preds=bool(opts.save_preds),
            topk=int(opts.topk),
            gt_profile_map=gt_profile_map,
            profile_model=profile_model,
            profile_class_names=profile_class_names,
            ckpt_profile_id=ckpt_profile_id,
        ),
        batch_size, device,
    )
    seen, elapsed = pr.seen, pr.elapsed

    metrics = compute_metrics(np.array(pr.y_true), np.array(pr.y_pred), class_names, critical_classes=critical_classes)

    log.info("\n" + _RULE)
    log.info("BATCH PREDICT REPORT")
    log.info(_RULE)
    log.info(f"Model:       {requested_model_path}")
    if effective_model_path.resolve() != requested_model_path.resolve():
        log.info(f"Resolved:    {effective_model_path}")
    log.info(f"Dataset:     {data_dir}")
    log.info(f"Split:       {opts.split}")
    log.info(f"Seen:        {seen}")
    log.info(f"Speed:       {seen / elapsed:.1f} images/s  (time: {elapsed:.1f}s)")
    log.info(f"Dataset size: total={info.total_samples} splits={info.split_sizes}")
    log.info("")
    log.info(format_metrics(metrics, class_names))

    # Profile metrics (when profile preds and GT labels were available)
    profile_metrics: dict[str, Any] | None = None
    if pr.profile_gt:
        profile_metrics, profile_names = profile_classification_metrics(pr.profile_gt, pr.profile_pred)
        log.info(format_profile_metrics(profile_metrics, profile_names))

    # Write outputs
    out_dir = Path(opts.out_dir) if opts.out_dir else (requested_model_path.parent / "history")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = now_tag()
    model_stem = requested_model_path.stem
    out_report = unique_path(out_dir / f"batch_report_{model_stem}_{opts.split}_{tag}.json")
    preds_path = unique_path(out_dir / f"batch_preds_{model_stem}_{opts.split}_{tag}.jsonl")

    report: dict[str, Any] = {
        "timestamp": now_iso(),
        "mode": "batch_predict",
        # Persist the *requested* model path for history grouping.
        # For bundles, this is the bundle directory; the resolved per-profile checkpoint is stored separately.
        "model_path": str(requested_model_path),
        "model_stem": model_stem,
        "resolved_model_path": str(effective_model_path) if str(effective_model_path) != str(requested_model_path) else None,
        "bundle_profile_id": dataset_profile_id,
        "dataset_path": str(data_dir),
        "split": opts.split,
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
    write_report(out_report, report)

    if opts.save_preds:
        append_jsonl(preds_path, pr.rows)
        log.info(f"Predictions saved to: {preds_path}")

    # Compare to previous rounds (reports)
    compare_dirs = [Path(p) for p in opts.history_dirs] or [requested_model_path.parent, requested_model_path.parent / "history"]
    crit = opts.history_critical_class if opts.history_metric == "critical_fn_rate" else None
    hist = history_rows(
        compare_dirs,
        scope=opts.history_scope,
        split_mode=opts.history_split,
        metric=opts.history_metric,
        critical_class=crit,
        model_path=requested_model_path,
        data_dir=data_dir,
        split=opts.split,
        exclude_path=out_report,
    )
    log.info(format_history(hist, scope=opts.history_scope, split_mode=opts.history_split,
                            metric=opts.history_metric, crit=crit, limit=opts.history_limit))

    n_mismatch = int(sum(1 for t, p in zip(pr.y_true, pr.y_pred, strict=True) if t != p))
    return {
        "report_path": str(out_report),
        "preds_path": str(preds_path) if opts.save_preds else None,
        "model_path": str(requested_model_path),
        "resolved_model_path": report["resolved_model_path"],
        "dataset_path": str(data_dir),
        "split": opts.split,
        "n_samples": int(seen),
        "n_mismatches": n_mismatch,
        "metrics": {k: metrics.get(k) for k in ("accuracy", "macro_f1", "critical_fn_rates")},
        "profile_accuracy": report.get("profile_accuracy"),
        "history_count": len(hist),
        "history_best": (
            {"value": hist[0][0], "report": report_path(hist[0][1])} if hist else None
        ),
    }
