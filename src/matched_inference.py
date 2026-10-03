"""Gate each cached dropout pass, then average probabilities identically."""
from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def score_cached_passes(attention: np.ndarray, token_scores: np.ndarray,
                        cls_logits: np.ndarray, confidence: np.ndarray | None = None,
                        gamma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    attention = np.asarray(attention, dtype=np.float64)
    scores = np.asarray(token_scores, dtype=np.float64)
    cls = np.asarray(cls_logits, dtype=np.float64)
    if attention.ndim != 3 or scores.shape != attention.shape or cls.shape != attention.shape[:2]:
        raise ValueError("Expected aligned pass/case/feature scores and pass/case CLS logits")
    if not np.isfinite(attention).all() or not np.isfinite(scores).all() or not np.isfinite(cls).all():
        raise ValueError("Cached inference arrays must be finite")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")
    if confidence is None:
        gated = attention.copy()
    else:
        confidence = np.asarray(confidence, dtype=np.float64)
        if confidence.shape != attention.shape[1:] or not np.isfinite(confidence).all():
            raise ValueError("Confidence must be a finite case/feature array")
        gate = np.clip(confidence, 0, 1) ** gamma
        weighted = attention * gate[None, :, :]
        denominator = weighted.sum(axis=-1, keepdims=True)
        gated = np.divide(weighted, denominator, out=attention.copy(), where=denominator > 0)
    probabilities = sigmoid(cls + np.sum(gated * scores, axis=-1))
    return probabilities.mean(axis=0), gated.mean(axis=0)
