"""Generate a mixed-profile dataset for training a profile (component type) classifier.

Iterates over multiple component profiles from the config, generates samples for each,
and writes v2 labels with profile_id set. Logic: :mod:`simple_sim.generation.profile_dataset`.

Usage:
    python -m simple_sim generate-profile-dataset --config configs/run_profile_cls.yaml --out outputs/sim_data/runs/run_profile_cls
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import add_json_flag, add_out_arg, run_cli

COMMAND = "generate_profile_dataset"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Generate mixed-profile dataset for profile classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    add_out_arg(parser, 'Output directory for dataset')
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.generation.profile_dataset import generate_profile_dataset

    run_cli(COMMAND, lambda: generate_profile_dataset(
        Path(args.config),
        Path(args.out),
        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
    ), args.json)
    return 0
