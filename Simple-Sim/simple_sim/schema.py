"""Data contract schemas for JSONL metadata and labels.

Schema notes:
- meta.jsonl schema_version 1: original fields only
- meta.jsonl schema_version 2: adds render_meta (backend-specific render details)
"""

import dataclasses
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar
from simple_sim.config import VALID_CLASSES

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
    nominal: Dict[str, float]
    defect: Dict[str, Any]
    augment: Dict[str, float]
    # Backend-specific info (camera pose, lighting, renderer version, etc.).
    render_meta: Dict[str, Any] = field(default_factory=dict)

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


def write_jsonl(path: Path, rows: List[Any]) -> None:
    """Write rows to JSONL file with atomic operation and validation.

    Args:
        path: Output JSONL file path
        rows: List of dataclass instances (MetaRow or LabelRow)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    temp_path = path.with_suffix('.tmp')

    with open(temp_path, 'w') as f:
        for row in rows:
            json_line = json.dumps(asdict(row))
            f.write(json_line + '\n')

    # Atomic rename
    temp_path.replace(path)


def read_jsonl(path: Path, schema_cls: Type[T]) -> List[T]:
    """Read and validate JSONL file.

    Args:
        path: Input JSONL file path
        schema_cls: Dataclass type (MetaRow or LabelRow)

    Returns:
        List of validated dataclass instances
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

            # Check for unknown fields using dataclass introspection
            if not dataclasses.is_dataclass(schema_cls):
                raise ValueError(f"Unknown schema class: {schema_cls}")
            expected_fields = {f.name for f in dataclasses.fields(schema_cls)}

            unknown_fields = set(data.keys()) - expected_fields
            if unknown_fields:
                raise ValueError(f"Unknown fields at line {line_num}: {unknown_fields}")

            # Backward compat: default profile_id for v1 LabelRows
            if schema_cls == LabelRow and 'profile_id' not in data:
                data['profile_id'] = ""
            # Backward compat: meta.jsonl v1 doesn't have render_meta
            if schema_cls == MetaRow and 'render_meta' not in data:
                data['render_meta'] = {}

            try:
                row = schema_cls(**data)
                rows.append(row)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Schema validation failed at line {line_num}: {e}")

    return rows


def validate_jsonl_pair(meta_path: Path, labels_path: Path) -> None:
    """Validate that meta.jsonl and labels.jsonl are consistent.

    Args:
        meta_path: Path to meta.jsonl
        labels_path: Path to labels.jsonl

    Raises:
        ValueError: If validation fails
    """
    meta_rows = read_jsonl(meta_path, MetaRow)
    label_rows = read_jsonl(labels_path, LabelRow)

    # Check counts match
    if len(meta_rows) != len(label_rows):
        raise ValueError(f"Row count mismatch: meta={len(meta_rows)}, labels={len(label_rows)}")

    # Check IDs match
    meta_ids = {row.id for row in meta_rows}
    label_ids = {row.id for row in label_rows}

    if meta_ids != label_ids:
        missing_in_labels = meta_ids - label_ids
        missing_in_meta = label_ids - meta_ids
        msg = []
        if missing_in_labels:
            msg.append(f"IDs in meta but not labels: {missing_in_labels}")
        if missing_in_meta:
            msg.append(f"IDs in labels but not meta: {missing_in_meta}")
        raise ValueError("; ".join(msg))
