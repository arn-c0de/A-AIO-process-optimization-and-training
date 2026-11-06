#!/usr/bin/env python3
"""Generate synthetic PCB defect dataset.

The logic lives in :mod:`simple_sim.generation.dataset`; this is a thin CLI wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simple_sim.cli_utils import add_json_flag, emit_result
from simple_sim.generation.blender import postprocess_blender_images as _postprocess_blender_images  # noqa: F401 (back-compat)
from simple_sim.generation.dataset import generate_dataset  # noqa: F401 (re-exported for tests/tools)

COMMAND = "generate"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic PCB defect dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for dataset')
    parser.add_argument('--extend', action='store_true', help='Append new samples to an existing dataset directory instead of overwriting')
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )
    add_json_flag(parser)
    args = parser.parse_args(argv)

    try:
        result = generate_dataset(
            Path(args.config),
            Path(args.out),
            extend=bool(args.extend),
            enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
        )
    except Exception as exc:
        if not args.json:
            raise
        emit_result(COMMAND, ok=False, error=str(exc))
        sys.exit(1)
    if args.json:
        emit_result(COMMAND, result)


if __name__ == '__main__':
    main()
