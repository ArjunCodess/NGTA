"""NGTA package."""

from .attention_hook import apply_confidence_gate, attention_to_nars, revise_attention_truths
from .data_loader import (
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_ID_COLUMN,
    DEFAULT_NUMERIC_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    load_data_bundle,
    load_merged_tcga_frame,
)
from .knowledge_base import SYMBOLIC_RULES, build_symbolic_truth_matrices
from .nars_interface import (
    confidence_to_evidence,
    deduce_truth_values,
    evidence_to_confidence,
    nars_deduction,
    nars_revision,
    neural_to_nars,
    revise_truth_values,
    truth_to_expectation,
)
from .pipeline import PipelineConfig, run_pipeline

__all__ = [
    "DEFAULT_CATEGORICAL_COLUMNS",
    "DEFAULT_ID_COLUMN",
    "DEFAULT_NUMERIC_COLUMNS",
    "DEFAULT_TARGET_COLUMN",
    "PipelineConfig",
    "SYMBOLIC_RULES",
    "apply_confidence_gate",
    "attention_to_nars",
    "build_symbolic_truth_matrices",
    "confidence_to_evidence",
    "deduce_truth_values",
    "evidence_to_confidence",
    "load_data_bundle",
    "load_merged_tcga_frame",
    "nars_deduction",
    "nars_revision",
    "neural_to_nars",
    "revise_attention_truths",
    "revise_truth_values",
    "run_pipeline",
    "truth_to_expectation",
]
