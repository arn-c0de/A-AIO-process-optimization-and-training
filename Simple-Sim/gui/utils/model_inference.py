"""Model loading and inference wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

from simple_sim.data_loader import IMAGENET_MEAN, IMAGENET_STD
from simple_sim.models import load_checkpoint_model

_LOG = logging.getLogger(__name__)

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ModelWrapper:
    """Wrapper for trained model with inference utilities."""

    def __init__(self, model_path: Path, device: str = 'cpu') -> None:
        """Initialize model wrapper.

        Args:
            model_path: Path to model checkpoint
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self.class_names: list[str] = []
        self.transform = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Shared loader: builds the ResNet18 head (or an ensemble) and puts it in eval mode on device.
        self.model, self.class_names, _checkpoint = load_checkpoint_model(self.model_path, self.device)

        self.transform = _INFERENCE_TRANSFORM

    def predict(self, image_path: Path) -> tuple[str, float, np.ndarray]:
        """Run inference on single image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform and add batch dimension
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_value = confidence.item()
        probs_array = probabilities.cpu().numpy()[0]

        return predicted_class, confidence_value, probs_array


def load_model(model_path: Path, device: str = 'cpu') -> ModelWrapper | None:
    """Load model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        ModelWrapper or None if loading fails
    """
    try:
        return ModelWrapper(model_path, device)
    except Exception as e:
        _LOG.warning("Failed to load model: %s", e)
        return None


def predict_image(model: ModelWrapper, image_path: Path) -> dict[str, Any] | None:
    """Predict single image and return results as dictionary.

    Args:
        model: ModelWrapper instance
        image_path: Path to image file

    Returns:
        Dictionary with prediction results or None if failed
    """
    try:
        predicted_class, confidence, probabilities = model.predict(image_path)

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {name: float(prob)
                            for name, prob in zip(model.class_names, probabilities)}
        }
    except Exception as e:
        _LOG.warning("Failed to predict image %s: %s", image_path, e)
        return None
