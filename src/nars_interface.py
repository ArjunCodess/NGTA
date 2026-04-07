from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[float, np.ndarray]
TensorLike = Union[float, np.ndarray, torch.Tensor]


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


def _to_tensor(value: TensorLike, reference: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        if reference is not None:
            return value.to(dtype=reference.dtype, device=reference.device)
        return value.to(dtype=torch.float32)
    if reference is not None:
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value, dtype=torch.float32)


def revise_truth_values(
    f1: TensorLike,
    c1: TensorLike,
    f2: TensorLike,
    c2: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Combine two truth values for the same proposition by NARS revision."""
    if any(torch.is_tensor(value) for value in (f1, c1, f2, c2)):
        reference = next(value for value in (f1, c1, f2, c2) if torch.is_tensor(value))
        f1_tensor = torch.clamp(_to_tensor(f1, reference=reference), min=0.0, max=1.0)
        f2_tensor = torch.clamp(_to_tensor(f2, reference=reference), min=0.0, max=1.0)
        c1_tensor = torch.clamp(_to_tensor(c1, reference=reference), min=0.001, max=0.999)
        c2_tensor = torch.clamp(_to_tensor(c2, reference=reference), min=0.001, max=0.999)

        w1 = c1_tensor / (1.0 - c1_tensor)
        w2 = c2_tensor / (1.0 - c2_tensor)
        revised_frequency = (w1 * f1_tensor + w2 * f2_tensor) / (w1 + w2)
        revised_confidence = (w1 + w2) / (w1 + w2 + 1.0)
        return revised_frequency, revised_confidence

    f1_array = np.clip(_to_array(f1), 0.0, 1.0)
    f2_array = np.clip(_to_array(f2), 0.0, 1.0)
    c1_array = np.clip(_to_array(c1), 0.001, 0.999)
    c2_array = np.clip(_to_array(c2), 0.001, 0.999)

    w1 = c1_array / (1.0 - c1_array)
    w2 = c2_array / (1.0 - c2_array)
    revised_frequency = (w1 * f1_array + w2 * f2_array) / (w1 + w2)
    revised_confidence = (w1 + w2) / (w1 + w2 + 1.0)

    return _return_like(f1, revised_frequency), _return_like(f1, revised_confidence)


def nars_revision(
    f1: TensorLike,
    c1: TensorLike,
    f2: TensorLike,
    c2: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Backward-compatible alias for NARS truth-value revision."""
    return revise_truth_values(f1, c1, f2, c2)
