#!/usr/bin/env python3
"""Generate a mixed-profile dataset for training a profile (component type) classifier.

Iterates over multiple component profiles from the config, generates samples for each,
and writes v2 labels with profile_id set. Logic: :mod:`simple_sim.generation.profile_dataset`.

Usage:
    .venv/bin/python scripts/generate_profile_dataset.py --config configs/run_profile_cls.yaml --out outputs/sim_data/runs/run_profile_cls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.cli_utils import add_json_flag, add_out_arg, run_cli
from simple_sim.generation.profile_dataset import generate_profile_dataset

COMMAND = "generate_profile_dataset"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Generate mixed-profile dataset for profile classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    add_out_arg(parser, 'Output directory for dataset')
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )
    add_json_flag(parser)
    args = parser.parse_args(argv)

    run_cli(COMMAND, lambda: generate_profile_dataset(
        Path(args.config),
        Path(args.out),
        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
    ), args.json)


if __name__ == '__main__':
    main()
