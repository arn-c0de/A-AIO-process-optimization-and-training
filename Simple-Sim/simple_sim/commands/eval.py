"""Evaluate trained model on frozen test set."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import (
    PROFILE_MISMATCH_HINT,
    add_data_arg,
    add_json_flag,
    add_model_arg,
    add_profile_mismatch_flag,
    run_cli,
    with_hint,
)
from simple_sim.devices import default_device_name
from simple_sim.manifest import ProfileMismatchError

COMMAND = "eval"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Evaluate trained model on test set')
    add_data_arg(parser, 'Path to dataset directory')
    add_model_arg(parser, 'Path to trained model checkpoint')
    add_profile_mismatch_flag(parser, (
        "DANGEROUS: allow evaluating a checkpoint on a dataset with a different component profile. "
        "This disables the hard profile_id mismatch guard and continues with a warning."
    ))
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: average the softmax over horizontal/vertical flips and '
                             '90-degree rotations (8 views; for cardinal-rotation-augmented datasets)')
    parser.add_argument('--calibration', action='store_true',
                        help='Also report calibration: expected calibration error (15 bins), negative '
                             'log-likelihood and Brier score (metrics.calibration in the report)')
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.evaluation import run_eval

    run_cli(COMMAND, with_hint(lambda: run_eval(
        Path(args.data), Path(args.model), args.device, allow_profile_mismatch=args.allow_profile_mismatch,
        tta=args.tta, calibration=args.calibration,
    ), ProfileMismatchError, PROFILE_MISMATCH_HINT), args.json)
    return 0
