"""Batch predict on a dataset split and report accuracy/metrics + history compare.

This is an "evaluate with more context" command:
- runs predictions with a trained checkpoint
- compares to ground-truth labels.jsonl
- writes a timestamped report + per-sample predictions
- prints deltas vs previous reports for the same model stem

Example:
  python -m simple_sim batch-predict \\
    --model outputs/models/run_0001.pt \\
    --data outputs/sim_data/runs/run_0001 \\
    --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from simple_sim.cli_utils import add_data_arg, add_json_flag, add_model_arg, run_cli
from simple_sim.devices import default_device_name

if TYPE_CHECKING:
    from simple_sim.batch_predict import BatchPredictOptions

COMMAND = "batch_predict"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Batch predict and evaluate on a dataset split")
    add_model_arg(ap, "Path to checkpoint (.pt)")
    add_data_arg(ap, "Dataset dir (has meta.jsonl/labels.jsonl/splits/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to evaluate")
    ap.add_argument("--device", default=default_device_name())
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


def options_from_args(args: argparse.Namespace) -> BatchPredictOptions:
    """Map parsed ``batch-predict`` arguments to :class:`~simple_sim.batch_predict.BatchPredictOptions`."""
    from simple_sim.batch_predict import BatchPredictOptions

    return BatchPredictOptions(
        model=Path(args.model),
        data=Path(args.data),
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        save_preds=bool(args.save_preds),
        topk=int(args.topk),
        history_scope=args.history_scope,
        history_metric=args.history_metric,
        history_critical_class=args.history_critical_class,
        history_split=args.history_split,
        history_limit=int(args.history_limit),
        history_dirs=tuple(Path(p.strip()) for p in (args.history_dirs or "").split(",") if p.strip()),
        profile_model=Path(args.profile_model) if args.profile_model else None,
    )


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.batch_predict import run_batch_predict

    run_cli(COMMAND, lambda: run_batch_predict(options_from_args(args)), args.json)
    return 0
