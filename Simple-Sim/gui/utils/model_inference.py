"""Model loading and inference wrapper."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


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
        self.model: Optional[nn.Module] = None
        self.class_names: list[str] = []
        self.transform = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        num_classes = len(self.class_names)

        # Create model
        self.model = models.resnet18()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup transform (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: Path) -> Tuple[str, float, np.ndarray]:
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


def load_model(model_path: Path, device: str = 'cpu') -> Optional[ModelWrapper]:
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
        print(f"Failed to load model: {e}")
        return None


def predict_image(model: ModelWrapper, image_path: Path) -> Optional[Dict[str, Any]]:
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
        print(f"Failed to predict image {image_path}: {e}")
        return None
