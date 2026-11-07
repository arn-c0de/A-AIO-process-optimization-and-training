"""Two-stage classifier: profile identification then defect classification.

Stage 1: Profile classifier identifies which component type (profile) is present.
Stage 2: Per-profile defect model classifies the defect state (OK/MISSING/MISALIGNED/TOMBSTONE).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from simple_sim.data_loader import load_image_bgr, preprocess_bgr
from simple_sim.devices import resolve_device
from simple_sim.eval_utils import predict_probs, top_prediction
from simple_sim.inference import Classifier
from simple_sim.model_bundle import bundle_checkpoint_path


@dataclass(slots=True)
class TwoStageResult:
    """Result from two-stage classification."""
    profile_id: str
    profile_confidence: float
    profile_probabilities: dict[str, float]
    defect_class: str
    defect_confidence: float
    defect_probabilities: dict[str, float]
    review_flag: bool  # True if profile_confidence < threshold


class TwoStageClassifier:
    """Two-stage inference: profile classifier -> per-profile defect model.

    Args:
        profile_model_path: Path to trained profile classifier checkpoint (.pt)
        defect_bundle_path: Path to defect model bundle directory (.bundle or dir)
        device: Torch device or device string (``cpu``, ``cuda``, ``cuda:1``)
        min_profile_confidence: Threshold below which review_flag is set
    """

    def __init__(
        self,
        profile_model_path: Path,
        defect_bundle_path: Path,
        device: str | torch.device = 'cpu',
        min_profile_confidence: float = 0.5,
    ) -> None:
        self.device = resolve_device(device)
        self.min_profile_confidence = min_profile_confidence
        self.defect_bundle_path = Path(defect_bundle_path)

        self._profile = Classifier(Path(profile_model_path), self.device)
        self.profile_model, self.profile_names = self._profile.model, self._profile.class_names

        # Lazy-loaded per-profile defect classifiers: profile_id -> Classifier
        self._defect_models: dict[str, Classifier] = {}

    def _get_defect_model(self, profile_id: str) -> Classifier:
        """Lazy-load the defect classifier for the given profile."""
        if profile_id not in self._defect_models:
            ckpt_path = bundle_checkpoint_path(self.defect_bundle_path, profile_id, kind="best")
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"No defect model for profile '{profile_id}' in bundle: {self.defect_bundle_path}\n"
                    f"Expected: {ckpt_path}"
                )
            self._defect_models[profile_id] = Classifier(ckpt_path, self.device)
        return self._defect_models[profile_id]

    def predict_array(self, img_bgr: np.ndarray) -> TwoStageResult:
        """Run two-stage prediction on a BGR numpy array."""
        x = preprocess_bgr(img_bgr).unsqueeze(0).to(self.device)

        # Stage 1: Profile classification
        profile_id, profile_conf, profile_probs = top_prediction(
            predict_probs(self.profile_model, x)[0], self.profile_names
        )

        # Stage 2: Defect classification using predicted profile's model
        defect = self._get_defect_model(profile_id)
        defect_class, defect_conf, defect_probs = top_prediction(
            predict_probs(defect.model, x)[0], defect.class_names
        )

        return TwoStageResult(
            profile_id=profile_id,
            profile_confidence=profile_conf,
            profile_probabilities=profile_probs,
            defect_class=defect_class,
            defect_confidence=defect_conf,
            defect_probabilities=defect_probs,
            review_flag=profile_conf < self.min_profile_confidence,
        )

    def predict(self, image_path: Path) -> TwoStageResult:
        """Run two-stage prediction on an image file."""
        return self.predict_array(load_image_bgr(image_path))
