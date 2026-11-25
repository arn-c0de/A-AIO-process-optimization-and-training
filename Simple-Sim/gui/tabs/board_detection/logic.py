"""Logic for the Board Detection Tab."""

from __future__ import annotations
from pathlib import Path

from simple_sim.two_stage import TwoStageClassifier, TwoStageResult


class BoardDetectionLogic:
    def __init__(self) -> None:
        self._classifier: TwoStageClassifier | None = None
        self._profile_model_path: Path | None = None
        self._defect_bundle_path: Path | None = None
        self._device: str | None = None

    def predict(
        self,
        profile_model_path: Path,
        defect_bundle_path: Path,
        image_path: Path,
        device: str,
        min_profile_confidence: float,
    ) -> TwoStageResult:
        """Two-stage prediction of *image_path* (blocking: call it from a worker, e.g. ``BaseTab._run_background``).

        The classifier is reused while the model paths and the device stay the same.
        """
        if (
            self._classifier is None
            or self._profile_model_path != profile_model_path
            or self._defect_bundle_path != defect_bundle_path
            or self._device != device
        ):
            self._classifier = TwoStageClassifier(
                profile_model_path=profile_model_path,
                defect_bundle_path=defect_bundle_path,
                device=device,
                min_profile_confidence=min_profile_confidence,
            )
            self._profile_model_path = profile_model_path
            self._defect_bundle_path = defect_bundle_path
            self._device = device
        else:
            self._classifier.min_profile_confidence = min_profile_confidence
        return self._classifier.predict(image_path)
