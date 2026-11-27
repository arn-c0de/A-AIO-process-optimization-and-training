"""Dataset quality and consistency checks (files, schemas, IDs, images, splits, seeds).

Library behind ``tools/validate_dataset.py`` and the GUI validation suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import yaml

from simple_sim.cli_utils import RULE
from simple_sim.io_utils import load_yaml
from simple_sim.manifest import read_dataset_manifest
from simple_sim.rng import derive_sample_seed, parse_sample_id
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, validate_jsonl_pair
from simple_sim.splits import assert_no_overlap, missing_classes, read_split
from simple_sim.telemetry import emit

PASS, FAIL, WARN = "pass", "fail", "warn"
_ICON = {PASS: "✓", FAIL: "✗ FAIL:", WARN: "⚠ WARNING:"}
_LIST_LIMIT = 5


class _Checks:
    """Collects check results (for --json) and prints them as they are recorded."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str, examples: list[str] | None = None) -> bool:
        self.items.append({"name": name, "status": status, "detail": detail})
        print(f"  {_ICON[status]} {detail}")
        for ex in (examples or [])[:_LIST_LIMIT]:
            print(f"    - {ex}")
        if examples and len(examples) > _LIST_LIMIT:
            print(f"    ... and {len(examples) - _LIST_LIMIT} more")
        return status != FAIL

    @property
    def ok(self) -> bool:
        return all(c["status"] != FAIL for c in self.items)


def validate_dataset(data_dir: Path, seed_mismatch_log_limit: int = 5) -> bool:
    """Validate a complete dataset; return True if all checks pass."""
    return run_validation(data_dir, seed_mismatch_log_limit)[0]


def run_validation(data_dir: Path, seed_mismatch_log_limit: int = 5) -> tuple[bool, list[dict[str, Any]]]:
    """Validate *data_dir*; return ``(ok, checks)`` with one ``{name, status, detail}`` per check."""
    data_dir = Path(data_dir)
    emit("validate_start", dataset_dir=str(data_dir))
    checks = _Checks()
    try:
        _run_checks(data_dir, seed_mismatch_log_limit, checks)
        return checks.ok, checks.items
    finally:
        # Always emit validate_done (also on early returns/exceptions) so the GUI leaves "validating".
        emit("validate_done", dataset_dir=str(data_dir), ok=checks.ok)


def _run_checks(data_dir: Path, seed_mismatch_log_limit: int, c: _Checks) -> None:
    print(RULE)
    print("DATASET VALIDATION")
    print(RULE)
    print(f"Dataset: {data_dir}\n")
    try:
        _checks_body(data_dir, seed_mismatch_log_limit, c)
    finally:
        print("\n" + RULE)
        print("✓ ALL VALIDATIONS PASSED" if c.ok else "✗ SOME VALIDATIONS FAILED")
        print(RULE)


def _checks_body(data_dir: Path, seed_mismatch_log_limit: int, c: _Checks) -> None:
    if not _check_files(data_dir, c):
        return  # Can't continue without images
    rows = _check_schemas(data_dir, c)
    if rows is None:
        return
    meta_rows, label_rows = rows
    _check_row_counts(data_dir, meta_rows, label_rows, c)
    _check_images(data_dir, meta_rows, c)
    splits = _check_splits(data_dir / "splits", meta_rows, c)
    _check_class_distribution(splits, label_rows, c)

    # 8: Determinism (seed replay on the first 20 samples)
    print("\n8. Checking determinism (sample verification)...")
    _check_seeds(data_dir, meta_rows, seed_mismatch_log_limit, c)


def _check_files(data_dir: Path, c: _Checks) -> bool:
    """1: Required files/directories; False when ``images/`` is missing."""
    print("1. Checking required files...")
    for filename in ("meta.jsonl", "labels.jsonl", "config.yaml"):
        exists = (data_dir / filename).exists()
        c.add(f"file:{filename}", PASS if exists else FAIL, f"{filename} exists" if exists else f"Missing {filename}")
    splits_exist = (data_dir / "splits").exists()
    c.add("dir:splits", PASS if splits_exist else FAIL,
          "splits/ directory exists" if splits_exist else "Missing splits directory")
    images_exist = (data_dir / "images").exists()
    return c.add("dir:images", PASS if images_exist else FAIL,
                 "images/ directory exists" if images_exist else "Missing images directory")


def _check_schemas(data_dir: Path, c: _Checks) -> tuple[list[MetaRow], list[LabelRow]] | None:
    """2: Schema validation; None when a file cannot be parsed."""
    print("\n2. Validating JSONL schemas...")
    try:
        meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    except Exception as e:  # noqa: BLE001 - any parse error is a failed check
        c.add("schema:meta", FAIL, f"meta.jsonl invalid - {e}")
        return None
    c.add("schema:meta", PASS, f"meta.jsonl valid ({len(meta_rows)} rows)")
    try:
        label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)
    except Exception as e:  # noqa: BLE001
        c.add("schema:labels", FAIL, f"labels.jsonl invalid - {e}")
        return None
    c.add("schema:labels", PASS, f"labels.jsonl valid ({len(label_rows)} rows)")
    if not meta_rows:
        # Every later check passes vacuously on zero rows; an empty dataset cannot be trained on.
        c.add("non_empty", FAIL, "Dataset has no samples (meta.jsonl is empty)")
    return meta_rows, label_rows


def _check_row_counts(data_dir: Path, meta_rows: list[MetaRow], label_rows: list[LabelRow], c: _Checks) -> None:
    """3: Row counts, 4: ID consistency."""
    print("\n3. Checking row counts...")
    if len(meta_rows) == len(label_rows):
        c.add("row_counts", PASS, f"Row counts match ({len(meta_rows)} samples)")
    else:
        c.add("row_counts", FAIL, f"Row count mismatch (meta={len(meta_rows)}, labels={len(label_rows)})")

    print("\n4. Checking ID consistency...")
    try:
        validate_jsonl_pair(data_dir / "meta.jsonl", data_dir / "labels.jsonl")
        c.add("id_consistency", PASS, "All IDs match between meta and labels")
    except Exception as e:  # noqa: BLE001
        c.add("id_consistency", FAIL, f"ID mismatch - {e}")


def _check_images(data_dir: Path, meta_rows: list[MetaRow], c: _Checks) -> None:
    """5: Images exist and are readable."""
    print("\n5. Checking image files...")
    missing: list[str] = []
    corrupt: list[str] = []
    for row in meta_rows:
        p = data_dir / row.image_path
        if not p.exists():
            missing.append(row.image_path)
        elif cv2.imread(str(p)) is None:
            corrupt.append(row.image_path)
    if missing:
        c.add("images:exist", FAIL, f"{len(missing)} missing images", missing)
    else:
        c.add("images:exist", PASS, f"All {len(meta_rows)} images exist")
    if corrupt:
        c.add("images:readable", FAIL, f"{len(corrupt)} corrupt images", corrupt)
    else:
        c.add("images:readable", PASS, "All images readable")


def _check_splits(splits_dir: Path, meta_rows: list[MetaRow], c: _Checks) -> dict[str, list[str]]:
    """6: Split files, overlap and membership; returns the split files that exist."""
    print("\n6. Checking split overlap...")
    splits: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        if split_file.exists():
            splits[split_name] = read_split(split_file)
        else:
            c.add(f"split:{split_name}", FAIL, f"Missing {split_name}.txt")
    if len(splits) != 3:
        return splits
    try:
        assert_no_overlap(splits)
        sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        c.add("split_overlap", PASS, f"No overlap between splits ({sizes})")
    except ValueError as e:
        c.add("split_overlap", FAIL, f"Split overlap detected - {e}")
    # Split files must name samples of this dataset, and every sample should be in a split.
    meta_ids = {row.id for row in meta_rows}
    split_ids = {sid for ids in splits.values() for sid in ids}
    unknown = sorted(split_ids - meta_ids)
    if unknown:
        c.add("split_membership", FAIL, f"{len(unknown)} split entries are not in meta.jsonl", unknown)
    else:
        c.add("split_membership", PASS, "All split entries are samples of this dataset")
    unassigned = sorted(meta_ids - split_ids)
    if unassigned:
        c.add("split_coverage", WARN, f"{len(unassigned)} samples are in no split file (never trained or evaluated)",
              unassigned)
    return splits


def _check_class_distribution(splits: dict[str, list[str]], label_rows: list[LabelRow], c: _Checks) -> None:
    """7: Class distribution per split (warning only)."""
    print("\n7. Checking class distribution...")
    for split_name, missing in missing_classes(splits, label_rows).items():
        if missing:
            c.add(f"class_coverage:{split_name}", WARN, f"{split_name} missing classes: {sorted(missing)}")
        else:
            c.add(f"class_coverage:{split_name}", PASS, f"{split_name} contains all classes")


def _check_seeds(data_dir: Path, meta_rows: list[MetaRow], log_limit: int, c: _Checks) -> None:
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        c.add("determinism", FAIL, "config.yaml missing, cannot verify seeds")
        return
    try:
        run_seed = (load_yaml(config_path) or {})["run"]["seed"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        c.add("determinism", FAIL, f"cannot read run.seed from config.yaml - {e}")
        return

    manifest_version = None
    extend_len = 0
    generator_script = "scripts/generate.py"
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_dataset_manifest(manifest_path)
            manifest_version = int(manifest.get("manifest_version", 1))
            hist = manifest.get("extend_history") or []
            extend_len = len(hist) if isinstance(hist, list) else 0
            generator = manifest.get("generator")
            if isinstance(generator, dict) and generator.get("script"):
                generator_script = str(generator["script"])
        except (OSError, ValueError):
            pass
    # Extended, multi-profile and merged datasets (other generator scripts) can contain
    # historical rows generated with previous configs/seeds, so strict per-row seed replay
    # can yield false negatives.
    strict = not (manifest_version == 2 or extend_len > 1 or generator_script != "scripts/generate.py")

    checked = mismatches = logged = 0
    for row in meta_rows[:20]:
        try:
            _run_id, domain, _split, index = parse_sample_id(row.id)
        except ValueError as e:
            c.add(f"determinism:{row.id}", FAIL, f"Error checking {row.id}: {e}")
            continue
        if row.seed != derive_sample_seed(run_seed, domain, index):
            mismatches += 1
            if logged < max(0, log_limit):
                print(f"  ✗ Seed mismatch for {row.id}")
                logged += 1
        checked += 1

    if mismatches == 0:
        c.add("determinism", PASS, f"Determinism verified ({checked} samples checked)")
        return
    if mismatches > logged:
        print(f"  … {mismatches - logged} additional seed mismatches suppressed")
    if strict:
        c.add("determinism", FAIL, f"{mismatches}/{checked} samples have seed mismatches")
    else:
        c.add("determinism", WARN, f"{mismatches}/{checked} sampled rows have seed mismatches "
              "(non-strict mode for multi/extended/merged dataset)")
