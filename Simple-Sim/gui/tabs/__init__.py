"""GUI tabs for the multi-tab monitor interface."""

from .base_tab import BaseTab
from .pipeline_tab import PipelineControlTab
from .analysis_tab import AnalysisTab
from .weights_tab import WeightsTab
from .validation_tab import ValidationTab
from .predictions_tab import PredictionsTab

__all__ = ['BaseTab', 'PipelineControlTab', 'AnalysisTab', 'PredictionsTab', 'WeightsTab', 'ValidationTab']
