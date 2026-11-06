"""Logic for the Datasets Tab (tk-free): created-date and details text for a dataset."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def _load_manifest(ds: Path) -> dict:
    manifest_path = ds / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def dataset_created_display(ds: Path) -> str:
    """``YYYY-MM-DD HH:MM`` from the manifest's ``created_at``, else the folder mtime, else ``-``."""
    manifest_path = ds / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            created = str(manifest.get("created_at") or "").strip()
            if created:
                # Supports ISO "....Z" format from manifests.
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    try:
        dt = datetime.fromtimestamp(ds.stat().st_mtime)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def dataset_details_text(ds: Path) -> str:
    """Human-readable summary: path, sample/split counts and samples per component profile."""
    lines: list[str] = []
    lines.append(f"Dataset: {ds.name}")
    lines.append(f"Path: {ds}")
    lines.append("")

    manifest = _load_manifest(ds)

    total_samples = None
    split_stats = {}
    ds_stats = manifest.get("dataset_stats") or {}
    if isinstance(ds_stats, dict):
        total_samples = ds_stats.get("total_samples")
        split_stats = ds_stats.get("splits") or {}

    labels_path = ds / "labels.jsonl"
    if total_samples is None and labels_path.exists():
        try:
            with labels_path.open("r", encoding="utf-8") as fh:
                total_samples = sum(1 for line in fh if line.strip())
        except Exception:
            total_samples = None

    lines.append(f"Total samples: {total_samples if total_samples is not None else 'unknown'}")

    if isinstance(split_stats, dict) and split_stats:
        tr = int(split_stats.get("train", 0) or 0)
        va = int(split_stats.get("val", 0) or 0)
        te = int(split_stats.get("test", 0) or 0)
        lines.append(f"Splits: train={tr}, val={va}, test={te}")
    lines.append("")

    profile_counts: Counter[str] = Counter()
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                pid = str(row.get("profile_id") or "").strip()
                if not pid:
                    pid = "unknown_or_legacy"
                profile_counts[pid] += 1

    if not profile_counts:
        comp = manifest.get("component_profile")
        if isinstance(comp, dict):
            pid = str(comp.get("profile_id") or "").strip()
            if pid:
                if isinstance(total_samples, int) and total_samples >= 0:
                    profile_counts[pid] = total_samples
                else:
                    profile_counts[pid] = 0
        profiles = manifest.get("profiles")
        if isinstance(profiles, list):
            for p in profiles:
                if isinstance(p, dict):
                    pid = str(p.get("profile_id") or "").strip()
                    if pid and pid not in profile_counts:
                        profile_counts[pid] = 0

    lines.append("Samples per profile:")
    if profile_counts:
        for pid, cnt in sorted(profile_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {pid}: {cnt}")
    else:
        lines.append("- (no profile data found)")

    return "\n".join(lines)
