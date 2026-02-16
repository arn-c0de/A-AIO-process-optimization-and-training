"""GUI tabs for the multi-tab monitor interface."""

from .core.base import BaseTab
from .predictions.tab import PredictionsTab
from .validation.tab import ValidationTab
from .analysis.tab import AnalysisTab
from .board_detection.tab import BoardDetectionTab
from .weights.tab import WeightsTab
from .merge.tab import MergeTab
from .pipeline.tab import PipelineControlTab

__all__ = [
    'BaseTab',
    'PipelineControlTab',
    'AnalysisTab',
    'PredictionsTab',
    'WeightsTab',
    'ValidationTab',
    'MergeTab',
    'BoardDetectionTab',
]