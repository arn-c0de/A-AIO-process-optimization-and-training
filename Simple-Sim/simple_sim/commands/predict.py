"""Predict defect class for a single image using a trained checkpoint.

Usage examples:
  python -m simple_sim predict --model outputs/models/run_0001.pt --image /path/to/img.png

  # Predict a dataset sample by ID (loads image_path from meta.jsonl):
  python -m simple_sim predict --model outputs/models/run_0001.pt --data outputs/sim_data/runs/run_0001 --id run_0001/domain_A/test/000042
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from simple_sim.cli_utils import add_data_arg, add_json_flag, add_model_arg, run_cli
from simple_sim.devices import default_device_name

COMMAND = "predict"
SUPPORTS_JSON = True


def print_prediction(result: dict[str, Any]) -> None:
    """Human-readable output of :func:`simple_sim.inference.predict_image`."""
    print("Prediction")
    print(f"  Model:  {result['model_path']}")
    print(f"  Image:  {result['image_path']}")
    if result["id"]:
        print(f"  ID:     {result['id']}")
    if result["gt"] is not None:
        print(f"  GT:     {result['gt']}")
    print(f"  Pred:   {result['pred']}")
    if result["gt"] is not None:
        print(f"  Match:  {result['match']}")
    print("\nTop-k:")
    for name, prob in result["topk"]:
        print(f"  {name:<12} {prob:.4f}")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Predict defect class for an image")
    add_model_arg(parser, "Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", help="Path to image file (png/jpg)")
    add_data_arg(parser, "Optional: dataset dir to lookup ground truth/ID (has meta.jsonl/labels.jsonl)", required=False)
    parser.add_argument("--id", help="Optional: sample ID (to load image from dataset + show ground truth label)")
    parser.add_argument("--device", default=default_device_name(), help="cuda/cpu")
    parser.add_argument("--topk", type=int, default=4, help="How many classes to show")
    add_json_flag(parser)
    return parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    from simple_sim.inference import predict_image

    def run() -> dict[str, Any]:
        result = predict_image(
            Path(args.model),
            image_path=Path(args.image) if args.image else None,
            data_dir=Path(args.data) if args.data else None,
            sample_id=args.id,
            device=args.device,
            topk=args.topk,
        )
        print_prediction(result)
        return result

    run_cli(COMMAND, run, args.json)
    return 0
