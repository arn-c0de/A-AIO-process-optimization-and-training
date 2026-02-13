"""GUI tabs for the multi-tab monitor interface."""

from .base_tab import BaseTab
from .pipeline_tab import PipelineControlTab
from .analysis_tab import AnalysisTab
from .validation_tab import ValidationTab

__all__ = ['BaseTab', 'PipelineControlTab', 'AnalysisTab', 'ValidationTab']
