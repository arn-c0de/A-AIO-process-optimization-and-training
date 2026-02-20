"""Two-stage classifier: profile identification then defect classification.

Stage 1: Profile classifier identifies which component type (profile) is present.
Stage 2: Per-profile defect model classifies the defect state (OK/MISSING/MISALIGNED/TOMBSTONE).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from simple_sim.data_loader import IMAGENET_MEAN, IMAGENET_STD
from simple_sim.model_bundle import bundle_checkpoint_path

_PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@dataclass
class TwoStageResult:
    """Result from two-stage classification."""
    profile_id: str
    profile_confidence: float
    profile_probabilities: Dict[str, float]
    defect_class: str
    defect_confidence: float
    defect_probabilities: Dict[str, float]
    review_flag: bool  # True if profile_confidence < threshold


def _load_model(checkpoint_path: Path, device: torch.device):
    """Load a ResNet18 checkpoint and return (model, class_names, checkpoint)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, class_names, checkpoint


def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Preprocess a BGR image to a normalized tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return _PREPROCESS_TRANSFORM(img_rgb)


class TwoStageClassifier:
    """Two-stage inference: profile classifier -> per-profile defect model.

    Args:
        profile_model_path: Path to trained profile classifier checkpoint (.pt)
        defect_bundle_path: Path to defect model bundle directory (.bundle or dir)
        device: Torch device string
        min_profile_confidence: Threshold below which review_flag is set
    """

    def __init__(
        self,
        profile_model_path: Path,
        defect_bundle_path: Path,
        device: str = 'cpu',
        min_profile_confidence: float = 0.5,
    ):
        self.device = torch.device(device)
        self.min_profile_confidence = min_profile_confidence
        self.defect_bundle_path = Path(defect_bundle_path)

        # Load profile classifier
        self.profile_model, self.profile_names, _ = _load_model(
            Path(profile_model_path), self.device
        )

        # Lazy-loaded per-profile defect models
        self._defect_models: Dict[str, tuple] = {}  # profile_id -> (model, class_names)

    def _get_defect_model(self, profile_id: str):
        """Lazy-load defect model for the given profile."""
        if profile_id not in self._defect_models:
            ckpt_path = bundle_checkpoint_path(self.defect_bundle_path, profile_id, kind="best")
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"No defect model for profile '{profile_id}' in bundle: {self.defect_bundle_path}\n"
                    f"Expected: {ckpt_path}"
                )
            model, class_names, _ = _load_model(ckpt_path, self.device)
            self._defect_models[profile_id] = (model, class_names)
        return self._defect_models[profile_id]

    def predict_array(self, img_bgr: np.ndarray) -> TwoStageResult:
        """Run two-stage prediction on a BGR numpy array.

        Args:
            img_bgr: Input image in BGR format (numpy array)

        Returns:
            TwoStageResult with profile and defect predictions
        """
        x = _preprocess(img_bgr).unsqueeze(0).to(self.device)

        # Stage 1: Profile classification
        with torch.no_grad():
            logits = self.profile_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        profile_id_pred = self.profile_names[pred_idx]
        profile_confidence = float(probs[pred_idx])
        profile_probabilities = {name: float(probs[i]) for i, name in enumerate(self.profile_names)}

        review_flag = profile_confidence < self.min_profile_confidence

        # Stage 2: Defect classification using predicted profile's model
        defect_model, defect_class_names = self._get_defect_model(profile_id_pred)

        with torch.no_grad():
            defect_logits = defect_model(x)
            defect_probs = torch.softmax(defect_logits, dim=1).squeeze(0).cpu().numpy()

        defect_pred_idx = int(np.argmax(defect_probs))
        defect_class = defect_class_names[defect_pred_idx]
        defect_confidence = float(defect_probs[defect_pred_idx])
        defect_probabilities = {name: float(defect_probs[i]) for i, name in enumerate(defect_class_names)}

        return TwoStageResult(
            profile_id=profile_id_pred,
            profile_confidence=profile_confidence,
            profile_probabilities=profile_probabilities,
            defect_class=defect_class,
            defect_confidence=defect_confidence,
            defect_probabilities=defect_probabilities,
            review_flag=review_flag,
        )

    def predict(self, image_path: Path) -> TwoStageResult:
        """Run two-stage prediction on an image file.

        Args:
            image_path: Path to image file

        Returns:
            TwoStageResult with profile and defect predictions
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise IOError(f"Failed to load image: {image_path}")
        return self.predict_array(img_bgr)
