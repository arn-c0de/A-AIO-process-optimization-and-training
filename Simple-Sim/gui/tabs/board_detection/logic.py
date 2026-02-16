"""Logic for the Board Detection Tab."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import threading

from simple_sim.two_stage import TwoStageClassifier


class BoardDetectionLogic:
    def __init__(self) -> None:
        self._classifier: Optional[TwoStageClassifier] = None
        self._profile_model_path: Optional[Path] = None
        self._defect_bundle_path: Optional[Path] = None

    def predict(
        self,
        profile_model_path: Path,
        defect_bundle_path: Path,
        image_path: Path,
        device: str,
        min_profile_confidence: float,
        on_complete: Callable[[object], None],
        on_error: Callable[[str], None],
    ) -> None:
        def _run():
            try:
                if (
                    self._classifier is None
                    or self._profile_model_path != profile_model_path
                    or self._defect_bundle_path != defect_bundle_path
                ):
                    self._classifier = TwoStageClassifier(
                        profile_model_path=profile_model_path,
                        defect_bundle_path=defect_bundle_path,
                        device=device,
                        min_profile_confidence=min_profile_confidence,
                    )
                    self._profile_model_path = profile_model_path
                    self._defect_bundle_path = defect_bundle_path
                else:
                    self._classifier.min_profile_confidence = min_profile_confidence

                result = self._classifier.predict(image_path)
                on_complete(result)
            except Exception as e:
                on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()
