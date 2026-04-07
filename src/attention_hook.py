from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nars_interface import neural_to_nars, revise_truth_values


@dataclass
class RevisedAttentionTruths:
    neural_frequency: np.ndarray
    neural_confidence: np.ndarray
    revised_frequency: np.ndarray
    revised_confidence: np.ndarray


def attention_to_nars(
    attention_means: np.ndarray,
    attention_variances: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MC attention summaries into per-feature NARS truth values."""
    frequency, confidence = neural_to_nars(
        np.asarray(attention_means, dtype=np.float64),
        np.asarray(attention_variances, dtype=np.float64),
        epsilon=epsilon,
    )
    return np.asarray(frequency), np.asarray(confidence)


def apply_confidence_gate(
    attention_weights: np.ndarray,
    confidences: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reweight attention by confidence and renormalize over features."""
    attention = np.asarray(attention_weights, dtype=np.float64)
    confidence = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)

    normalized_attention = attention / np.clip(
        attention.sum(axis=-1, keepdims=True),
        1e-8,
        None,
    )
    gated = normalized_attention * np.power(confidence, gamma)
    denominator = gated.sum(axis=-1, keepdims=True)
    safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
    gated_normalized = gated / safe_denominator

    if np.any(denominator <= 0.0):
        gated_normalized = np.where(denominator > 0.0, gated_normalized, normalized_attention)

    return gated_normalized


def revise_attention_truths(
    attention_means: np.ndarray,
    attention_variances: np.ndarray,
    symbolic_frequency: np.ndarray,
    symbolic_confidence: np.ndarray,
    symbolic_trigger_mask: np.ndarray,
    epsilon: float = 1e-5,
) -> RevisedAttentionTruths:
    """Fuse neural attention truths with symbolic truths for triggered features only."""
    neural_frequency, neural_confidence = attention_to_nars(
        attention_means,
        attention_variances,
        epsilon=epsilon,
    )
    symbolic_frequency_array = np.asarray(symbolic_frequency, dtype=np.float64)
    symbolic_confidence_array = np.asarray(symbolic_confidence, dtype=np.float64)
    symbolic_trigger_mask_array = np.asarray(symbolic_trigger_mask, dtype=bool)

    if symbolic_frequency_array.shape != neural_frequency.shape:
        raise ValueError("Symbolic frequency shape must match attention truth shape.")
    if symbolic_confidence_array.shape != neural_confidence.shape:
        raise ValueError("Symbolic confidence shape must match attention truth shape.")
    if symbolic_trigger_mask_array.shape != neural_confidence.shape:
        raise ValueError("Symbolic trigger mask shape must match attention truth shape.")

    revised_frequency, revised_confidence = revise_truth_values(
        np.asarray(neural_frequency, dtype=np.float64),
        np.asarray(neural_confidence, dtype=np.float64),
        symbolic_frequency_array,
        symbolic_confidence_array,
    )
    revised_frequency_array = np.where(
        symbolic_trigger_mask_array,
        np.asarray(revised_frequency, dtype=np.float64),
        np.asarray(neural_frequency, dtype=np.float64),
    )
    revised_confidence_array = np.where(
        symbolic_trigger_mask_array,
        np.asarray(revised_confidence, dtype=np.float64),
        np.asarray(neural_confidence, dtype=np.float64),
    )

    return RevisedAttentionTruths(
        neural_frequency=np.asarray(neural_frequency, dtype=np.float64),
        neural_confidence=np.asarray(neural_confidence, dtype=np.float64),
        revised_frequency=revised_frequency_array,
        revised_confidence=revised_confidence_array,
    )
