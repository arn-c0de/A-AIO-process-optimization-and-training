#!/usr/bin/env python3
"""Two-stage prediction: profile identification + defect classification.

Usage:
    .venv/bin/python scripts/predict_two_stage.py \
        --profile-model outputs/models/profile_classifier_v1.pt \
        --defect-bundle outputs/models/my_multi.bundle \
        --image /path/to/img.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.cli_utils import add_json_flag
from simple_sim.eval_utils import run_cli
from simple_sim.two_stage import TwoStageClassifier, TwoStageResult


def predict_two_stage(
    profile_model: Path,
    defect_bundle: Path,
    image: Path,
    *,
    device: str = "cpu",
    min_confidence: float = 0.5,
) -> TwoStageResult:
    """Run profile + defect prediction on one image file."""
    classifier = TwoStageClassifier(
        profile_model_path=Path(profile_model),
        defect_bundle_path=Path(defect_bundle),
        device=device,
        min_profile_confidence=min_confidence,
    )
    return classifier.predict(Path(image))


def print_two_stage(result: TwoStageResult, image: Path, min_confidence: float) -> None:
    """Human-readable output of a :class:`TwoStageResult`."""
    print("Two-Stage Prediction")
    print("=" * 50)
    print(f"Image: {image}")
    print()

    print("Stage 1: Profile Identification")
    print(f"  Predicted profile: {result.profile_id}")
    print(f"  Confidence: {result.profile_confidence:.4f}")
    if result.review_flag:
        print(f"  ** REVIEW FLAG: confidence below {min_confidence} **")
    print("  All probabilities:")
    for name, prob in sorted(result.profile_probabilities.items(), key=lambda x: -x[1]):
        print(f"    {name:<30} {prob:.4f}")

    print()
    print("Stage 2: Defect Classification")
    print(f"  Predicted defect: {result.defect_class}")
    print(f"  Confidence: {result.defect_confidence:.4f}")
    print("  All probabilities:")
    for name, prob in sorted(result.defect_probabilities.items(), key=lambda x: -x[1]):
        print(f"    {name:<15} {prob:.4f}")

    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description='Two-stage predict: profile + defect')
    parser.add_argument('--profile-model', required=True, help='Path to profile classifier checkpoint (.pt)')
    parser.add_argument('--defect-bundle', required=True, help='Path to defect model bundle directory')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                        help='Minimum profile confidence threshold (default: 0.5)')
    add_json_flag(parser)
    args = parser.parse_args()

    def run() -> dict[str, Any]:
        result = predict_two_stage(
            Path(args.profile_model), Path(args.defect_bundle), Path(args.image),
            device=args.device, min_confidence=args.min_confidence,
        )
        print_two_stage(result, args.image, args.min_confidence)
        return {"image_path": args.image, **asdict(result)}

    run_cli("predict_two_stage", run, args.json)


if __name__ == '__main__':
    main()
