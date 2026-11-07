"""Train ResNet18 profile (component type) classifier on mixed-profile dataset.

Usage:
    python -m simple_sim train-profile --data outputs/sim_data/runs/run_profile_cls --out outputs/models/profile_classifier_v1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import add_amp_flags, add_data_arg, add_json_flag, add_out_arg, run_cli
from simple_sim.devices import default_device_name

COMMAND = "train_profile"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Train profile classifier')
    add_data_arg(parser, 'Path to mixed-profile dataset directory')
    add_out_arg(parser, 'Output path for trained model (.pt)')
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    add_amp_flags(parser)
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.training import train_profile_model

    run_cli(COMMAND, lambda: train_profile_model(
        Path(args.data), Path(args.out), device=args.device, amp=args.amp,
    ), args.json)
    return 0
