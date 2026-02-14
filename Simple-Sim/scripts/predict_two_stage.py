#!/usr/bin/env python3
"""Two-stage prediction: profile identification + defect classification.

Usage:
    .venv/bin/python scripts/predict_two_stage.py \
        --profile-model outputs/models/profile_classifier_v1.pt \
        --defect-bundle outputs/models/my_multi.bundle \
        --image /path/to/img.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.two_stage import TwoStageClassifier


def main():
    parser = argparse.ArgumentParser(description='Two-stage predict: profile + defect')
    parser.add_argument('--profile-model', required=True, help='Path to profile classifier checkpoint (.pt)')
    parser.add_argument('--defect-bundle', required=True, help='Path to defect model bundle directory')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum profile confidence threshold (default: 0.5)')
    args = parser.parse_args()

    classifier = TwoStageClassifier(
        profile_model_path=Path(args.profile_model),
        defect_bundle_path=Path(args.defect_bundle),
        device=args.device,
        min_profile_confidence=args.min_confidence,
    )

    result = classifier.predict(Path(args.image))

    print("Two-Stage Prediction")
    print("=" * 50)
    print(f"Image: {args.image}")
    print()

    print("Stage 1: Profile Identification")
    print(f"  Predicted profile: {result.profile_id}")
    print(f"  Confidence: {result.profile_confidence:.4f}")
    if result.review_flag:
        print(f"  ** REVIEW FLAG: confidence below {args.min_confidence} **")
    print(f"  All probabilities:")
    for name, prob in sorted(result.profile_probabilities.items(), key=lambda x: -x[1]):
        print(f"    {name:<30} {prob:.4f}")

    print()
    print("Stage 2: Defect Classification")
    print(f"  Predicted defect: {result.defect_class}")
    print(f"  Confidence: {result.defect_confidence:.4f}")
    print(f"  All probabilities:")
    for name, prob in sorted(result.defect_probabilities.items(), key=lambda x: -x[1]):
        print(f"    {name:<15} {prob:.4f}")

    print("=" * 50)


if __name__ == '__main__':
    main()
