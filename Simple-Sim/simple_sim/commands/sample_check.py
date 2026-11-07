"""Sample a few images per class from a dataset split and verify predictions vs GT.

Example:
  python -m simple_sim sample-check \\
    --model outputs/models/run_0001.pt \\
    --data outputs/sim_data/runs/run_0001 \\
    --split test \\
    --per-class 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from simple_sim.cli_utils import add_data_arg, add_json_flag, add_model_arg, run_cli
from simple_sim.devices import default_device_name

COMMAND = "sample_check"
SUPPORTS_JSON = True


def print_sample_check(result: dict[str, Any]) -> None:
    """Human-readable table + summary of :func:`simple_sim.inference.sample_check`."""
    print(f"{'id':<34} {'img':<12} {'gt':<10} {'pred':<10} {'ok'}")
    for r in result["rows"]:
        print(f"{r['id']:<34} {r['img']:<12} {r['gt']:<10} {r['pred']:<10} {str(r['ok'])}")
    print("")
    print(f"Summary: {result['correct']}/{result['seen']} correct  (acc={result['accuracy']:.4f})")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Sample-check predictions vs GT")
    add_model_arg(ap, "Path to checkpoint (.pt)")
    add_data_arg(ap, "Dataset dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--per-class", type=int, default=5, help="How many samples per class")
    ap.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=default_device_name())
    add_json_flag(ap)
    return ap


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.inference import sample_check

    def run() -> dict[str, Any]:
        result = sample_check(
            Path(args.model), Path(args.data), split=args.split, per_class=args.per_class,
            seed=args.seed, batch_size=args.batch_size, device=args.device,
        )
        print_sample_check(result)
        return result

    run_cli(COMMAND, run, args.json)
    return 0
