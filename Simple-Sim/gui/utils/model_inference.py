"""GUI adapter over :class:`simple_sim.inference.Classifier` (plan L7).

The GUI used to run its own preprocessing with an extra ``Resize((256, 256))`` (plan X1), so its
predictions differed from ``predict`` / ``batch-predict`` / training for ROIs that are not
256 px. It now uses the library classifier, i.e. exactly the training/eval transform.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.inference import Classifier

_LOG = logging.getLogger(__name__)


class ModelWrapper:
    """A loaded checkpoint for the GUI (``model``, ``class_names``, :meth:`predict`)."""

    def __init__(self, model_path: Path, device: str = 'cpu') -> None:
        self.model_path = Path(model_path)
        self._classifier = Classifier(self.model_path, device)
        self.device = self._classifier.device
        self.model = self._classifier.model
        self.class_names: list[str] = list(self._classifier.class_names)

    def predict(self, image_path: Path) -> tuple[str, float, np.ndarray]:
        """Classify one image file; return ``(predicted_class, confidence, class_probabilities)``.

        Raises ``ValueError`` if the image cannot be read.
        """
        try:
            pred = self._classifier.predict_path(image_path)
        except OSError as e:
            raise ValueError(str(e)) from e
        assert pred.probs_array is not None
        return pred.label, pred.confidence, pred.probs_array


def load_model(model_path: Path, device: str = 'cpu') -> ModelWrapper | None:
    """Load *model_path*; ``None`` (and a logged warning) if that fails."""
    try:
        return ModelWrapper(model_path, device)
    except Exception as e:  # noqa: BLE001 - the GUI shows "no model" instead of crashing
        _LOG.warning("Failed to load model: %s", e)
        return None


def predict_image(model: ModelWrapper, image_path: Path) -> dict[str, Any] | None:
    """``{predicted_class, confidence, probabilities}`` for one image, or ``None`` on failure."""
    try:
        predicted_class, confidence, probabilities = model.predict(image_path)
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {name: float(prob) for name, prob in zip(model.class_names, probabilities)},
        }
    except Exception as e:  # noqa: BLE001
        _LOG.warning("Failed to predict image %s: %s", image_path, e)
        return None
