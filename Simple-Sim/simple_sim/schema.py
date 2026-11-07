"""Data contract schemas for JSONL metadata and labels.

Schema notes:
- meta.jsonl schema_version 1: original fields only
- meta.jsonl schema_version 2: adds render_meta (backend-specific render details)
"""

import dataclasses
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, TypeVar
from simple_sim.classes import VALID_CLASSES
from simple_sim.io_utils import append_jsonl, atomic_write_jsonl

T = TypeVar('T')


@dataclass
class MetaRow:
    """Schema for meta.jsonl - one row per sample."""
    schema_version: int
    id: str
    run_id: str
    domain: str
    split: str
    seed: int
    image_path: str
    render_backend: str
    footprint: str
    nominal: dict[str, float]
    defect: dict[str, Any]
    augment: dict[str, float]
    # Backend-specific info (camera pose, lighting, renderer version, etc.).
    render_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate required fields."""
        if self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not self.domain:
            raise ValueError("domain cannot be empty")
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        # 0 is a valid seed for NumPy; require only non-negative.
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.image_path:
            raise ValueError("image_path cannot be empty")
        if not self.render_backend:
            raise ValueError("render_backend cannot be empty")
        if not self.footprint:
            raise ValueError("footprint cannot be empty")
        if self.render_meta is None:
            raise ValueError("render_meta cannot be null")
        if not isinstance(self.render_meta, dict):
            raise ValueError("render_meta must be a dict")

        # Validate nominal geometry (must contain at least these keys; extras like pad_spacing_y are allowed)
        required_nominal = {'pad_width', 'pad_height', 'pad_spacing', 'component_length', 'component_width'}
        missing = required_nominal - set(self.nominal.keys())
        if missing:
            raise ValueError(f"nominal is missing required keys: {missing}")

        # Validate defect params
        required_defect = {'type', 'shift_x', 'shift_y', 'rotation_deg', 'tilt_deg'}
        if set(self.defect.keys()) != required_defect:
            raise ValueError(f"defect must have exactly {required_defect}")
        if self.defect['type'] not in VALID_CLASSES:
            raise ValueError(f"Invalid defect type: {self.defect['type']}")

        # Validate augment params
        required_augment = {'blur_sigma', 'noise_stddev', 'brightness_factor', 'contrast_factor', 'rotation_deg'}
        if set(self.augment.keys()) != required_augment:
            raise ValueError(f"augment must have exactly {required_augment}")


@dataclass
class LabelRow:
    """Schema for labels.jsonl - one row per sample.

    schema_version 1: original (id + class_name only)
    schema_version 2: adds profile_id for multi-profile datasets
    """
    schema_version: int
    id: str
    class_name: str
    profile_id: str = ""

    def __post_init__(self):
        """Validate required fields."""
        if self.schema_version not in (1, 2):
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.id:
            raise ValueError("id cannot be empty")
        if self.class_name not in VALID_CLASSES:
            raise ValueError(f"Invalid class: {self.class_name}")
        if self.schema_version == 2 and not self.profile_id:
            raise ValueError("schema_version=2 requires non-empty profile_id")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    """Write rows to JSONL file with atomic operation and validation.

    Args:
        path: Output JSONL file path
        rows: List of dataclass instances (MetaRow or LabelRow)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, (asdict(row) for row in rows))


def append_jsonl_rows(path: Path, rows: list[Any]) -> None:
    """Append dataclass rows (already validated on construction) to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, (asdict(row) for row in rows))


def read_jsonl(path: Path, schema_cls: type[T]) -> list[T]:
    """Read and validate JSONL file.

    Args:
        path: Input JSONL file path
        schema_cls: Dataclass type (MetaRow or LabelRow)

    Returns:
        List of validated dataclass instances
    """
    return list(iter_jsonl(path, schema_cls))


def iter_jsonl(path: Path, schema_cls: type[T]) -> Iterator[T]:
    """:func:`read_jsonl` row by row (same validation and errors, nothing kept in memory).

    The missing-file check runs on the call, not on the first ``next()``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return _iter_jsonl(path, schema_cls)


def _iter_jsonl(path: Path, schema_cls: type[T]) -> Iterator[T]:
    # Per-file constants (hoisted out of the per-line loop; plan S1). The dataclass check stays
    # lazy so an empty file still returns [] for any class, as before.
    expected_fields: frozenset[str] | None = None
    is_label = schema_cls == LabelRow
    is_meta = schema_cls == MetaRow
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

            # Check for unknown fields using dataclass introspection
            if expected_fields is None:
                if not dataclasses.is_dataclass(schema_cls):
                    raise ValueError(f"Unknown schema class: {schema_cls}")
                expected_fields = frozenset(f.name for f in dataclasses.fields(schema_cls))

            unknown_fields = data.keys() - expected_fields
            if unknown_fields:
                raise ValueError(f"Unknown fields at line {line_num}: {unknown_fields}")

            # Backward compat: default profile_id for v1 LabelRows
            if is_label and 'profile_id' not in data:
                data['profile_id'] = ""
            # Backward compat: meta.jsonl v1 doesn't have render_meta
            if is_meta and 'render_meta' not in data:
                data['render_meta'] = {}

            try:
                row = schema_cls(**data)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Schema validation failed at line {line_num}: {e}") from e
            yield row


def validate_jsonl_pair(
    meta_path: Path,
    labels_path: Path,
    *,
    on_meta: Callable[[MetaRow], None] | None = None,
    on_label: Callable[[LabelRow], None] | None = None,
) -> None:
    """Validate that meta.jsonl and labels.jsonl are consistent.

    Streams both files (only the ids are kept). *on_meta* / *on_label* see every validated row in
    file order (meta first), so a caller can collect what it needs in the same pass.

    Args:
        meta_path: Path to meta.jsonl
        labels_path: Path to labels.jsonl

    Raises:
        ValueError: If validation fails
    """
    meta_ids: set[str] = set()
    n_meta = 0
    for row in iter_jsonl(meta_path, MetaRow):
        meta_ids.add(row.id)
        n_meta += 1
        if on_meta is not None:
            on_meta(row)
    label_ids: set[str] = set()
    n_labels = 0
    for lrow in iter_jsonl(labels_path, LabelRow):
        label_ids.add(lrow.id)
        n_labels += 1
        if on_label is not None:
            on_label(lrow)

    # Check counts match
    if n_meta != n_labels:
        raise ValueError(f"Row count mismatch: meta={n_meta}, labels={n_labels}")

    # Check IDs match

    if meta_ids != label_ids:
        missing_in_labels = meta_ids - label_ids
        missing_in_meta = label_ids - meta_ids
        msg = []
        if missing_in_labels:
            msg.append(f"IDs in meta but not labels: {missing_in_labels}")
        if missing_in_meta:
            msg.append(f"IDs in labels but not meta: {missing_in_meta}")
        raise ValueError("; ".join(msg))
