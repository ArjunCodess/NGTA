"""NGTA package."""

from .attention_hook import apply_confidence_gate, attention_to_nars
from .data_loader import (
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_ID_COLUMN,
    DEFAULT_NUMERIC_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    load_data_bundle,
    load_merged_tcga_frame,
)
from .nars_interface import nars_revision, neural_to_nars
from .pipeline import PipelineConfig, run_pipeline

__all__ = [
    "DEFAULT_CATEGORICAL_COLUMNS",
    "DEFAULT_ID_COLUMN",
    "DEFAULT_NUMERIC_COLUMNS",
    "DEFAULT_TARGET_COLUMN",
    "PipelineConfig",
    "apply_confidence_gate",
    "attention_to_nars",
    "load_data_bundle",
    "load_merged_tcga_frame",
    "nars_revision",
    "neural_to_nars",
    "run_pipeline",
]
