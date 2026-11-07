"""Train ResNet18 classifier on synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import (
    PROFILE_MISMATCH_HINT,
    add_amp_flags,
    add_data_arg,
    add_json_flag,
    add_out_arg,
    add_profile_mismatch_flag,
    run_cli,
    with_hint,
)
from simple_sim.devices import default_device_name
from simple_sim.manifest import ProfileMismatchError

COMMAND = "train"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Train ResNet18 classifier')
    add_data_arg(parser, 'Path to dataset directory')
    add_out_arg(parser, 'Output path for trained model')
    parser.add_argument('--resume', type=str, default=None, help='Optional: checkpoint to resume from (.pt)')
    add_profile_mismatch_flag(parser, (
        "DANGEROUS: allow resuming from a checkpoint trained on a different component profile. "
        "This disables the hard profile_id mismatch guard and continues with a warning."
    ))
    parser.add_argument('--extra-epochs', type=int, default=None,
                        help='Optional: train N additional epochs beyond the resumed epoch (default: use dataset config epochs)')
    parser.add_argument('--out-mode', choices=['best', 'last'], default='best',
                        help="Whether --out points to the best checkpoint ('best') or the latest epoch ('last'). Default: best")
    parser.add_argument('--device', type=str, default=default_device_name(), help='Device (cuda/cpu)')
    add_amp_flags(parser)
    parser.add_argument('--threads', type=int, default=None,
                        help='Torch intra-op threads for CPU training (default: torch picks; on hybrid CPUs the '
                             'number of performance cores is often faster than the default)')
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.training import train_model

    run_cli(COMMAND, with_hint(lambda: train_model(
        Path(args.data), Path(args.out),
        resume=Path(args.resume) if args.resume else None,
        allow_profile_mismatch=args.allow_profile_mismatch,
        extra_epochs=args.extra_epochs,
        out_mode=args.out_mode,
        device=args.device,
        amp=args.amp,
        threads=args.threads,
    ), ProfileMismatchError, PROFILE_MISMATCH_HINT), args.json)
    return 0
