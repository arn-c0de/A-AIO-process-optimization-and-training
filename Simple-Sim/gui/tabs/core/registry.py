"""Registry for monitor tab specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Type

from gui.tabs.analysis.tab import AnalysisTab
from gui.tabs.board_detection.tab import BoardDetectionTab
from gui.tabs.core.base import BaseTab
from gui.tabs.datasets.tab import DatasetsTab
from gui.tabs.merge.tab import MergeTab
from gui.tabs.pipeline.tab import PipelineControlTab
from gui.tabs.predictions.tab import PredictionsTab
from gui.tabs.validation.tab import ValidationTab
from gui.tabs.weights.tab import WeightsTab


@dataclass(frozen=True)
class TabSpec:
    key: str
    title: str
    cls: Type[BaseTab]


class TabRegistry:
    """Central tab registration used by the monitor app."""

    _TAB_SPECS: tuple[TabSpec, ...] = (
        TabSpec("pipeline", "Pipeline Control", PipelineControlTab),
        TabSpec("datasets", "Datasets", DatasetsTab),
        TabSpec("analysis", "Analysis", AnalysisTab),
        TabSpec("validation", "Validation", ValidationTab),
        TabSpec("predictions", "Predictions", PredictionsTab),
        TabSpec("weights", "Weights", WeightsTab),
        TabSpec("merge", "Merge", MergeTab),
        TabSpec("board_detection", "Board Detection", BoardDetectionTab),
    )

    @classmethod
    def specs(cls) -> Iterable[TabSpec]:
        return cls._TAB_SPECS

    @classmethod
    def keys(cls) -> list[str]:
        return [s.key for s in cls._TAB_SPECS]

    @classmethod
    def iter_specs(cls) -> Iterator[TabSpec]:
        yield from cls._TAB_SPECS
