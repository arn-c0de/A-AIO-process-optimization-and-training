#!/usr/bin/env python3
"""Validate dataset quality and consistency (files, schemas, IDs, images, splits, seeds)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_sim.cli_utils import add_json_flag, emit_result  # noqa: E402
from simple_sim.manifest import read_dataset_manifest  # noqa: E402
from simple_sim.rng import derive_sample_seed, parse_sample_id  # noqa: E402
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, validate_jsonl_pair  # noqa: E402
from simple_sim.splits import assert_no_overlap, read_split  # noqa: E402
from simple_sim.telemetry import emit  # noqa: E402

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
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset: {data_dir}\n")
    try:
        _checks_body(data_dir, seed_mismatch_log_limit, c)
    finally:
        print("\n" + "=" * 60)
        print("✓ ALL VALIDATIONS PASSED" if c.ok else "✗ SOME VALIDATIONS FAILED")
        print("=" * 60)


def _checks_body(data_dir: Path, seed_mismatch_log_limit: int, c: _Checks) -> None:
    # 1: Required files/directories
    print("1. Checking required files...")
    for filename in ("meta.jsonl", "labels.jsonl", "config.yaml"):
        exists = (data_dir / filename).exists()
        c.add(f"file:{filename}", PASS if exists else FAIL, f"{filename} exists" if exists else f"Missing {filename}")
    splits_dir = data_dir / "splits"
    c.add("dir:splits", PASS if splits_dir.exists() else FAIL,
          "splits/ directory exists" if splits_dir.exists() else "Missing splits directory")
    if not c.add("dir:images", PASS if (data_dir / "images").exists() else FAIL,
                 "images/ directory exists" if (data_dir / "images").exists() else "Missing images directory"):
        return  # Can't continue without images

    # 2: Schema validation
    print("\n2. Validating JSONL schemas...")
    try:
        meta_rows = read_jsonl(data_dir / "meta.jsonl", MetaRow)
    except Exception as e:  # noqa: BLE001 - any parse error is a failed check
        c.add("schema:meta", FAIL, f"meta.jsonl invalid - {e}")
        return
    c.add("schema:meta", PASS, f"meta.jsonl valid ({len(meta_rows)} rows)")
    try:
        label_rows = read_jsonl(data_dir / "labels.jsonl", LabelRow)
    except Exception as e:  # noqa: BLE001
        c.add("schema:labels", FAIL, f"labels.jsonl invalid - {e}")
        return
    c.add("schema:labels", PASS, f"labels.jsonl valid ({len(label_rows)} rows)")

    # 3: Row counts
    print("\n3. Checking row counts...")
    if len(meta_rows) == len(label_rows):
        c.add("row_counts", PASS, f"Row counts match ({len(meta_rows)} samples)")
    else:
        c.add("row_counts", FAIL, f"Row count mismatch (meta={len(meta_rows)}, labels={len(label_rows)})")

    # 4: ID consistency
    print("\n4. Checking ID consistency...")
    try:
        validate_jsonl_pair(data_dir / "meta.jsonl", data_dir / "labels.jsonl")
        c.add("id_consistency", PASS, "All IDs match between meta and labels")
    except Exception as e:  # noqa: BLE001
        c.add("id_consistency", FAIL, f"ID mismatch - {e}")

    # 5: Images exist and are readable
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

    # 6: Split files and overlap
    print("\n6. Checking split overlap...")
    splits: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        if split_file.exists():
            splits[split_name] = read_split(split_file)
        else:
            c.add(f"split:{split_name}", FAIL, f"Missing {split_name}.txt")
    if len(splits) == 3:
        try:
            assert_no_overlap(splits)
            sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
            c.add("split_overlap", PASS, f"No overlap between splits ({sizes})")
        except ValueError as e:
            c.add("split_overlap", FAIL, f"Split overlap detected - {e}")

    # 7: Class distribution per split (warning only)
    print("\n7. Checking class distribution...")
    id_to_label = {row.id: row.class_name for row in label_rows}
    all_classes = {row.class_name for row in label_rows}
    for split_name, sample_ids in splits.items():
        missing_classes = all_classes - {id_to_label[s] for s in sample_ids if s in id_to_label}
        if missing_classes:
            c.add(f"class_coverage:{split_name}", WARN, f"{split_name} missing classes: {sorted(missing_classes)}")
        else:
            c.add(f"class_coverage:{split_name}", PASS, f"{split_name} contains all classes")

    # 8: Determinism (seed replay on the first 20 samples)
    print("\n8. Checking determinism (sample verification)...")
    _check_seeds(data_dir, meta_rows, seed_mismatch_log_limit, c)


def _check_seeds(data_dir: Path, meta_rows: list[MetaRow], log_limit: int, c: _Checks) -> None:
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        c.add("determinism", FAIL, "config.yaml missing, cannot verify seeds")
        return
    try:
        run_seed = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})["run"]["seed"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        c.add("determinism", FAIL, f"cannot read run.seed from config.yaml - {e}")
        return

    manifest_version = None
    extend_len = 0
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_dataset_manifest(manifest_path)
            manifest_version = int(manifest.get("manifest_version", 1))
            hist = manifest.get("extend_history") or []
            extend_len = len(hist) if isinstance(hist, list) else 0
        except (OSError, ValueError):
            pass
    # Extended and/or multi-profile datasets can contain historical rows generated
    # with previous configs/seeds, so strict per-row seed replay can yield false negatives.
    strict = not (manifest_version == 2 or extend_len > 1)

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
              "(non-strict mode for multi/extended dataset)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--seed-mismatch-log-limit", type=int, default=5,
                        help="How many individual seed mismatch rows to print (default: 5)")
    add_json_flag(parser)
    args = parser.parse_args()

    ok, checks = run_validation(Path(args.data), seed_mismatch_log_limit=args.seed_mismatch_log_limit)
    if args.json:
        failed = [ch["name"] for ch in checks if ch["status"] == FAIL]
        emit_result(
            "validate",
            {
                "dataset": str(args.data),
                "checks": checks,
                "failed": failed,
                "warnings": [ch["name"] for ch in checks if ch["status"] == WARN],
            },
            ok=ok,
            error=None if ok else f"{len(failed)} check(s) failed: {', '.join(failed)}",
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
