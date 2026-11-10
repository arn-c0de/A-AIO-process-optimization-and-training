#!/usr/bin/env python3
"""Validate dataset quality and consistency (files, schemas, IDs, images, splits, seeds).

CLI of :mod:`simple_sim.dataset_validation`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_sim.cli_utils import add_json_flag, configure_cli_logging, emit_result  # noqa: E402
from simple_sim.dataset_validation import FAIL, WARN, run_validation, validate_dataset  # noqa: E402

__all__ = ["run_validation", "validate_dataset"]  # kept importable from here for existing callers


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--seed-mismatch-log-limit", type=int, default=5,
                        help="How many individual seed mismatch rows to print (default: 5)")
    add_json_flag(parser)
    args = parser.parse_args()
    configure_cli_logging()

    try:
        ok, checks = run_validation(Path(args.data), seed_mismatch_log_limit=args.seed_mismatch_log_limit)
    except Exception as e:  # noqa: BLE001 - with --json the caller reads the result line, not a traceback
        if not args.json:
            raise
        emit_result("validate", {"dataset": str(args.data)}, ok=False, error=f"{type(e).__name__}: {e}")
        sys.exit(1)
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
