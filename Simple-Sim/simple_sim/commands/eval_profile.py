"""Evaluate trained profile classifier on test set.

Usage:
    python -m simple_sim eval-profile --data outputs/sim_data/runs/run_profile_cls --model outputs/models/profile_classifier_v1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import add_data_arg, add_json_flag, add_model_arg, run_cli
from simple_sim.devices import default_device_name

COMMAND = "eval_profile"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Evaluate profile classifier on test set')
    add_data_arg(parser, 'Path to mixed-profile dataset directory')
    add_model_arg(parser, 'Path to trained profile classifier checkpoint')
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.evaluation import run_eval_profile

    run_cli(COMMAND, lambda: run_eval_profile(Path(args.data), Path(args.model), args.device), args.json)
    return 0
