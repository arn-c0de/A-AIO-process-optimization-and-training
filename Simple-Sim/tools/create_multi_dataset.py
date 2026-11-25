#!/usr/bin/env python3
"""Create a merged dataset from multiple profile-specific runs.

This aggregates several existing `outputs/sim_data/runs/<name>` folders into
a single dataset folder so you can train a multi-profile checkpoint.

Every label row is written as schema v2 carrying the profile_id of the source
dataset it came from. If the merged sources span 2+ profiles a multi-profile
(manifest_version 2) manifest is written, otherwise a regular v1 manifest.

Usage:
    .venv/bin/python tools/create_multi_dataset.py \
        --datasets outputs/sim_data/runs/run_0001 outputs/sim_data/runs/run_sot23 \
        --out outputs/sim_data/runs/run_multi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if not __package__:  # run as a script (python tools/x.py): make the checkout importable
    sys.path.insert(0, str(ROOT))

from simple_sim.cli_utils import add_json_flag, run_cli  # noqa: E402
from simple_sim.dataset_merge import MergeError, merge_datasets, source_profiles  # noqa: E402,F401  (source_profiles re-exported)
from simple_sim.io_utils import load_yaml  # noqa: E402

SCRIPT_NAME = "tools/create_multi_dataset.py"


def create_multi_dataset(
    datasets: list[Path],
    out_dir: Path,
    *,
    config_path: Path | None = None,
    run_id: str | None = None,
    profiles_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge *datasets* into *out_dir* (:func:`simple_sim.dataset_merge.merge_datasets`); return a summary
    (out, run_id, samples, splits, profiles). Images and the config are copied unchanged; every label class
    must be declared in the config (*config_path*, else the first source's). Run id: *run_id*, else the
    config's ``run.run_id`` (only with *config_path*), else the *out_dir* name."""
    out_dir = Path(out_dir)
    if not run_id and config_path is not None and Path(config_path).exists():
        run_id = str(((load_yaml(Path(config_path)) or {}).get("run") or {}).get("run_id") or "")
    try:
        result = merge_datasets(
            [Path(d) for d in datasets], out_dir, run_id=run_id or out_dir.name,
            config_path=Path(config_path) if config_path is not None else None,
            profiles_dir=Path(profiles_dir) if profiles_dir is not None else ROOT / "configs" / "profiles",
            script_name=SCRIPT_NAME, on_warning=lambda msg: print(f"WARNING: {msg}"),
        )
    except (MergeError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
    return result.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-profile dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Existing dataset folders to merge (e.g. outputs/sim_data/runs/run_0001)",
    )
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional run config for the merged dataset (classes, run_id). "
        "Defaults to the first source dataset's config.yaml.",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for merged samples (default: --config run_id, else the --out folder name).")
    add_json_flag(parser)
    args = parser.parse_args()

    run_cli("create-multi-dataset", lambda: create_multi_dataset(
        [Path(d) for d in args.datasets],
        Path(args.out),
        config_path=Path(args.config) if args.config else None,
        run_id=args.run_id,
    ), args.json)


if __name__ == "__main__":
    main()
