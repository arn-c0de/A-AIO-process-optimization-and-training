"""Generate synthetic PCB defect dataset.

The logic lives in :mod:`simple_sim.generation.dataset`; this is the CLI wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_sim.cli_utils import RENDERER_MISMATCH_HINT, add_json_flag, add_out_arg, run_cli, with_hint
from simple_sim.manifest import RendererMismatchError

COMMAND = "generate"
SUPPORTS_JSON = True


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description='Generate synthetic PCB defect dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    add_out_arg(parser, 'Output directory for dataset')
    parser.add_argument('--extend', action='store_true', help='Append new samples to an existing dataset directory instead of overwriting')
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )
    parser.add_argument(
        '--allow-renderer-mismatch',
        action='store_true',
        help='With --extend: append even if the dataset was rendered by another renderer version (mixes two looks)',
    )
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.generation.dataset import generate_dataset

    run_cli(COMMAND, with_hint(lambda: generate_dataset(
        Path(args.config),
        Path(args.out),
        extend=bool(args.extend),
        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
        allow_renderer_mismatch=bool(args.allow_renderer_mismatch),
    ), RendererMismatchError, RENDERER_MISMATCH_HINT), args.json)
    return 0
