"""Dataset edit operations for GUI workflows.

This module centralizes delete/move operations so both Analysis and Pipeline
views can edit datasets with consistent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from simple_sim.schema import LabelRow, MetaRow, read_jsonl, write_jsonl


@dataclass(frozen=True)
class DatasetSampleInfo:
    """Lightweight view model for sample selection dialogs."""

    sample_id: str
    class_name: str
    split: str
    image_path: str


@dataclass(frozen=True)
class DatasetEditSummary:
    """Summary of a completed dataset edit operation."""

    requested: int
    affected: int


def list_dataset_samples(dataset_dir: Path) -> List[DatasetSampleInfo]:
    """Return dataset samples for UI selection lists."""
    dataset_dir = Path(dataset_dir)
    meta_rows, label_rows = _read_rows(dataset_dir)
    label_by_id = {row.id: row.class_name for row in label_rows}

    out: List[DatasetSampleInfo] = []
    for row in meta_rows:
        out.append(
            DatasetSampleInfo(
                sample_id=row.id,
                class_name=label_by_id.get(row.id, "?"),
                split=row.split,
                image_path=row.image_path,
            )
        )
    return out


def delete_samples(dataset_dir: Path, sample_ids: Iterable[str]) -> DatasetEditSummary:
    """Delete samples from a dataset.

    The operation updates JSONL files first and removes image files afterwards.
    """
    dataset_dir = Path(dataset_dir)
    selected = _normalize_selection(sample_ids)
    if not selected:
        return DatasetEditSummary(requested=0, affected=0)

    meta_rows, label_rows = _read_rows(dataset_dir)
    meta_by_id = {row.id: row for row in meta_rows}
    _ensure_ids_exist(selected, meta_by_id, "delete")

    keep_meta = [row for row in meta_rows if row.id not in selected]
    keep_labels = [row for row in label_rows if row.id not in selected]

    write_jsonl(dataset_dir / "meta.jsonl", keep_meta)
    write_jsonl(dataset_dir / "labels.jsonl", keep_labels)
    recompute_manifest_stats(dataset_dir)

    for sample_id in selected:
        image_path = dataset_dir / meta_by_id[sample_id].image_path
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            # Keep row changes durable even if an orphan image cannot be deleted.
            pass

    return DatasetEditSummary(requested=len(selected), affected=len(selected))


def move_samples(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    sample_ids: Iterable[str],
    *,
    enforce_profile_match: bool = True,
) -> DatasetEditSummary:
    """Move samples from one dataset to another with JSONL + manifest updates."""
    src_dataset_dir = Path(src_dataset_dir)
    dst_dataset_dir = Path(dst_dataset_dir)
    if src_dataset_dir.resolve() == dst_dataset_dir.resolve():
        raise ValueError("Source and target dataset are identical.")

    selected = _normalize_selection(sample_ids)
    if not selected:
        return DatasetEditSummary(requested=0, affected=0)

    if enforce_profile_match:
        _ensure_profile_compatibility(src_dataset_dir, dst_dataset_dir)

    src_meta, src_labels = _read_rows(src_dataset_dir)
    dst_meta, dst_labels = _read_rows(dst_dataset_dir)

    src_meta_by_id = {row.id: row for row in src_meta}
    src_label_by_id = {row.id: row for row in src_labels}
    _ensure_ids_exist(selected, src_meta_by_id, "move")

    dst_ids = {row.id for row in dst_meta}
    dst_image_paths = {row.image_path for row in dst_meta}

    moved_pairs: List[Tuple[Path, Path]] = []
    selected_set = set(selected)

    new_src_meta = [row for row in src_meta if row.id not in selected_set]
    new_src_labels = [row for row in src_labels if row.id not in selected_set]

    new_dst_meta = list(dst_meta)
    new_dst_labels = list(dst_labels)

    try:
        for src_id in selected:
            src_meta_row = src_meta_by_id[src_id]
            src_label_row = src_label_by_id[src_id]

            new_id = _dedupe_sample_id(src_meta_row.id, dst_ids)
            dst_ids.add(new_id)

            dst_image_rel = _dedupe_image_path(src_meta_row.image_path, dst_image_paths)
            dst_image_paths.add(dst_image_rel)

            src_file = src_dataset_dir / src_meta_row.image_path
            dst_file = dst_dataset_dir / dst_image_rel
            if not src_file.exists():
                raise FileNotFoundError(f"Missing source image for sample '{src_id}': {src_file}")

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_file), str(dst_file))
            moved_pairs.append((src_file, dst_file))

            new_dst_meta.append(
                MetaRow(
                    schema_version=src_meta_row.schema_version,
                    id=new_id,
                    run_id=src_meta_row.run_id,
                    domain=src_meta_row.domain,
                    split=src_meta_row.split,
                    seed=src_meta_row.seed,
                    image_path=dst_image_rel,
                    render_backend=src_meta_row.render_backend,
                    footprint=src_meta_row.footprint,
                    nominal=dict(src_meta_row.nominal),
                    defect=dict(src_meta_row.defect),
                    augment=dict(src_meta_row.augment),
                    render_meta=dict(src_meta_row.render_meta),
                )
            )
            new_dst_labels.append(
                LabelRow(
                    schema_version=src_label_row.schema_version,
                    id=new_id,
                    class_name=src_label_row.class_name,
                    profile_id=src_label_row.profile_id,
                )
            )

        write_jsonl(src_dataset_dir / "meta.jsonl", new_src_meta)
        write_jsonl(src_dataset_dir / "labels.jsonl", new_src_labels)
        write_jsonl(dst_dataset_dir / "meta.jsonl", new_dst_meta)
        write_jsonl(dst_dataset_dir / "labels.jsonl", new_dst_labels)

        recompute_manifest_stats(src_dataset_dir)
        recompute_manifest_stats(dst_dataset_dir)
    except Exception:
        for src_file, dst_file in reversed(moved_pairs):
            try:
                src_file.parent.mkdir(parents=True, exist_ok=True)
                if dst_file.exists():
                    shutil.move(str(dst_file), str(src_file))
            except Exception:
                pass
        raise

    return DatasetEditSummary(requested=len(selected), affected=len(selected))


def recompute_manifest_stats(dataset_dir: Path) -> None:
    """Recompute dataset_stats in dataset_manifest.json when present."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return

    meta_rows, label_rows = _read_rows(dataset_dir)

    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in meta_rows:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1

    class_counts: Dict[str, int] = {}
    for row in label_rows:
        class_counts[row.class_name] = class_counts.get(row.class_name, 0) + 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["dataset_stats"] = {
        "total_samples": len(meta_rows),
        "splits": split_counts,
        "classes": class_counts,
    }

    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)


def _read_rows(dataset_dir: Path) -> Tuple[List[MetaRow], List[LabelRow]]:
    dataset_dir = Path(dataset_dir)
    meta_rows = read_jsonl(dataset_dir / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(dataset_dir / "labels.jsonl", LabelRow)
    return meta_rows, label_rows


def _normalize_selection(sample_ids: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for sample_id in sample_ids:
        key = str(sample_id).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _ensure_ids_exist(sample_ids: List[str], meta_by_id: Dict[str, MetaRow], action: str) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in meta_by_id]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise ValueError(f"Cannot {action}: sample ID(s) not found: {preview}{suffix}")


def _dedupe_sample_id(sample_id: str, existing_ids: set[str]) -> str:
    if sample_id not in existing_ids:
        return sample_id

    parts = sample_id.split("/")
    base_leaf = parts[-1]
    prefix = "/".join(parts[:-1])
    idx = 2
    while True:
        leaf = f"{base_leaf}__moved{idx}"
        candidate = f"{prefix}/{leaf}" if prefix else leaf
        if candidate not in existing_ids:
            return candidate
        idx += 1


def _dedupe_image_path(image_path: str, existing_image_paths: set[str]) -> str:
    normalized = str(Path(image_path).as_posix())
    if normalized not in existing_image_paths:
        return normalized

    p = Path(normalized)
    base_dir = str(p.parent).replace("\\", "/")
    stem = p.stem
    suffix = p.suffix
    idx = 2
    while True:
        name = f"{stem}__moved{idx}{suffix}"
        candidate = name if base_dir in ("", ".") else f"{base_dir}/{name}"
        if candidate not in existing_image_paths:
            return candidate
        idx += 1


def _ensure_profile_compatibility(src_dataset_dir: Path, dst_dataset_dir: Path) -> None:
    src = _read_profile_id(src_dataset_dir)
    dst = _read_profile_id(dst_dataset_dir)
    if src and dst and src != dst:
        raise ValueError(
            "Profile mismatch between datasets. "
            f"Source profile='{src}', target profile='{dst}'."
        )


def _read_profile_id(dataset_dir: Path) -> Optional[str]:
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = data.get("component_profile")
        if isinstance(profile, dict):
            value = profile.get("profile_id")
            return str(value).strip() if value else None
    except Exception:
        return None
    return None
