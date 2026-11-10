"""Report list of the weights tab: model matching and table cells (Tk-free)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def report_matches_model(report: Mapping[str, Any], model: Path) -> bool:
    """Whether *report* was written for *model* (same file name or stem, or a checkpoint inside the bundle dir)."""
    report_model = str(report.get("model_path") or "")
    if Path(report_model).name == model.name or str(report.get("model_stem") or "") == model.stem:
        return True
    if model.is_dir():
        try:
            return Path(report_model).resolve().parent == model.resolve()
        except Exception:
            return False
    return False


def report_cells(report: Mapping[str, Any], rel_path: Callable[[Path], str]) -> tuple[str, ...]:
    """``(when, split, seen, accuracy, macro_f1, model, dataset, report path)`` as shown in the reports table."""
    when = (report.get("timestamp") or "-").replace("T", " ")[:19]
    seen_v = report.get("seen_samples") or report.get("test_size")
    metrics = report.get("metrics", {})
    acc, f1 = metrics.get("accuracy"), metrics.get("macro_f1")
    return (
        when,
        str(report.get("split") or "-"),
        str(int(seen_v)) if seen_v is not None else "-",
        f"{float(acc):.4f}" if acc is not None else "-",
        f"{float(f1):.4f}" if f1 is not None else "-",
        Path(str(report.get("model_path") or "-")).name,
        Path(str(report.get("dataset_path") or "-")).name,
        rel_path(Path(str(report.get("_path") or "-"))) if report.get("_path") != "-" else "-",
    )
