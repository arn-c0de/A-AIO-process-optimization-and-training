"""Registry for monitor tab specifications."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator

from gui.tabs.analysis.tab import AnalysisTab
from gui.tabs.board_detection.tab import BoardDetectionTab
from gui.tabs.core.base import BaseTab
from gui.tabs.datasets.tab import DatasetsTab
from gui.tabs.merge.tab import MergeTab
from gui.tabs.pipeline.tab import PipelineControlTab
from gui.tabs.predictions.tab import PredictionsTab
from gui.tabs.profile_preview.tab import ProfilePreviewTab
from gui.tabs.validation.tab import ValidationTab
from gui.tabs.weights.tab import WeightsTab


@dataclass(frozen=True)
class TabSpec:
    key: str
    title: str
    cls: type[BaseTab]
    group: str = "General"
    description: str = ""


# Sidebar groups in display order.
GROUPS: tuple[str, ...] = ("Workflow", "Data", "Evaluate", "Models", "Inference")


class TabRegistry:
    """Central tab registration used by the monitor app."""

    _TAB_SPECS: tuple[TabSpec, ...] = (
        TabSpec("pipeline", "Pipeline Control", PipelineControlTab, "Workflow",
                "Generate, train and evaluate in one run."),
        TabSpec("datasets", "Datasets", DatasetsTab, "Data", "Browse, categorise and archive datasets."),
        TabSpec("profile_preview", "Profile Preview", ProfilePreviewTab, "Data",
                "Preview how a component profile renders."),
        TabSpec("analysis", "Analysis", AnalysisTab, "Evaluate", "Browse samples with overlays and give feedback."),
        TabSpec("validation", "Validation", ValidationTab, "Data", "Quality checks: schema, outliers, duplicates."),
        TabSpec("predictions", "Predictions", PredictionsTab, "Evaluate",
                "Batch-predict a split: metrics and confusion matrix."),
        TabSpec("weights", "Weights", WeightsTab, "Models", "Checkpoints: evaluate, compare, back up, export."),
        TabSpec("merge", "Merge", MergeTab, "Models", "Combine checkpoints into bundles or ensembles."),
        TabSpec("board_detection", "Board Detection", BoardDetectionTab, "Inference",
                "Detect and classify components on a full board image."),
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

    @classmethod
    def grouped(cls) -> list[tuple[str, list[TabSpec]]]:
        """Specs grouped for the sidebar (``GROUPS`` order, unknown groups last)."""
        order = list(GROUPS) + sorted({s.group for s in cls._TAB_SPECS} - set(GROUPS))
        return [(g, [s for s in cls._TAB_SPECS if s.group == g]) for g in order
                if any(s.group == g for s in cls._TAB_SPECS)]
