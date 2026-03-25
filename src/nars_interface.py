from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _to_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _return_like(original: ArrayLike, value: np.ndarray) -> ArrayLike:
    if np.isscalar(original):
        return float(np.asarray(value).item())
    return value


def neural_to_nars(
    p: ArrayLike,
    variance: ArrayLike,
    epsilon: float = 1e-5,
) -> tuple[ArrayLike, ArrayLike]:
    """Map a Bernoulli mean/variance estimate to NARS frequency and confidence."""
    p_array = np.clip(_to_array(p), 0.0, 1.0)
    variance_array = np.maximum(_to_array(variance), 0.0)
    confidence = (p_array * (1.0 - p_array) + epsilon) / (
        p_array * (1.0 - p_array) + epsilon + variance_array
    )
    return _return_like(p, p_array), _return_like(p, confidence)


def nars_revision(
    f1: ArrayLike,
    c1: ArrayLike,
    f2: ArrayLike,
    c2: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Combine two truth values for the same proposition by NARS revision."""
    f1_array = np.clip(_to_array(f1), 0.0, 1.0)
    f2_array = np.clip(_to_array(f2), 0.0, 1.0)
    c1_array = np.clip(_to_array(c1), 1e-8, 1.0 - 1e-8)
    c2_array = np.clip(_to_array(c2), 1e-8, 1.0 - 1e-8)

    w1 = c1_array / (1.0 - c1_array)
    w2 = c2_array / (1.0 - c2_array)
    revised_frequency = (w1 * f1_array + w2 * f2_array) / (w1 + w2)
    revised_confidence = (w1 + w2) / (w1 + w2 + 1.0)

    return _return_like(f1, revised_frequency), _return_like(f1, revised_confidence)
