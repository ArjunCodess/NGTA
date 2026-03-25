from __future__ import annotations

import numpy as np

from .nars_interface import neural_to_nars


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
