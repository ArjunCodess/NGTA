from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[float, np.ndarray]
TensorLike = Union[float, np.ndarray, torch.Tensor]
DEFAULT_EVIDENTIAL_HORIZON = 1.0
REVISION_EPSILON = 1e-6


def _to_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _return_like(original: ArrayLike, value: np.ndarray) -> ArrayLike:
    if np.isscalar(original):
        return float(np.asarray(value).item())
    return value


def evidence_to_confidence(
    evidence_weight: ArrayLike,
    k: float = DEFAULT_EVIDENTIAL_HORIZON,
) -> ArrayLike:
    """Convert evidential mass into NARS confidence."""
    evidence_array = np.maximum(_to_array(evidence_weight), 0.0)
    confidence = evidence_array / (evidence_array + float(k))
    return _return_like(evidence_weight, confidence)


def confidence_to_evidence(
    confidence: ArrayLike,
    k: float = DEFAULT_EVIDENTIAL_HORIZON,
    epsilon: float = 1e-9,
) -> ArrayLike:
    """Convert NARS confidence into evidential mass."""
    confidence_array = np.clip(_to_array(confidence), 0.0, 1.0 - epsilon)
    evidence = (confidence_array * float(k)) / np.clip(1.0 - confidence_array, epsilon, None)
    return _return_like(confidence, evidence)


def truth_to_expectation(
    frequency: ArrayLike,
    confidence: ArrayLike,
) -> ArrayLike:
    """Compute the NAL expectation value e = c(f - 0.5) + 0.5."""
    frequency_array = np.clip(_to_array(frequency), 0.0, 1.0)
    confidence_array = np.clip(_to_array(confidence), 0.0, 1.0)
    expectation = confidence_array * (frequency_array - 0.5) + 0.5
    return _return_like(frequency, expectation)


def neural_to_nars(
    p: ArrayLike,
    variance: ArrayLike,
    epsilon: float = 1e-5,
) -> tuple[ArrayLike, ArrayLike]:
    """Heuristically map a Bernoulli mean/variance estimate to an initial NARS-style truth value."""
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


def deduce_truth_values(
    f1: TensorLike,
    c1: TensorLike,
    f2: TensorLike,
    c2: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Apply the standard NAL strong-deduction truth-value function."""
    if any(torch.is_tensor(value) for value in (f1, c1, f2, c2)):
        reference = next(value for value in (f1, c1, f2, c2) if torch.is_tensor(value))
        f1_tensor = torch.clamp(_to_tensor(f1, reference=reference), min=0.0, max=1.0)
        f2_tensor = torch.clamp(_to_tensor(f2, reference=reference), min=0.0, max=1.0)
        c1_tensor = torch.clamp(_to_tensor(c1, reference=reference), min=0.0, max=1.0)
        c2_tensor = torch.clamp(_to_tensor(c2, reference=reference), min=0.0, max=1.0)
        deduced_frequency = f1_tensor * f2_tensor
        deduced_confidence = deduced_frequency * c1_tensor * c2_tensor
        return deduced_frequency, deduced_confidence

    f1_array = np.clip(_to_array(f1), 0.0, 1.0)
    f2_array = np.clip(_to_array(f2), 0.0, 1.0)
    c1_array = np.clip(_to_array(c1), 0.0, 1.0)
    c2_array = np.clip(_to_array(c2), 0.0, 1.0)
    deduced_frequency = f1_array * f2_array
    deduced_confidence = deduced_frequency * c1_array * c2_array
    return _return_like(f1, deduced_frequency), _return_like(f1, deduced_confidence)


def revise_truth_values(
    f1: TensorLike,
    c1: TensorLike,
    f2: TensorLike,
    c2: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Combine two truth values for the same proposition by NARS revision.

    This operator assumes the two inputs come from distinct evidential sources.
    """
    if any(torch.is_tensor(value) for value in (f1, c1, f2, c2)):
        reference = next(value for value in (f1, c1, f2, c2) if torch.is_tensor(value))
        f1_tensor = torch.clamp(_to_tensor(f1, reference=reference), min=0.0, max=1.0)
        f2_tensor = torch.clamp(_to_tensor(f2, reference=reference), min=0.0, max=1.0)
        c1_tensor = torch.clamp(_to_tensor(c1, reference=reference), min=REVISION_EPSILON, max=1.0 - REVISION_EPSILON)
        c2_tensor = torch.clamp(_to_tensor(c2, reference=reference), min=REVISION_EPSILON, max=1.0 - REVISION_EPSILON)

        w1 = c1_tensor / (1.0 - c1_tensor)
        w2 = c2_tensor / (1.0 - c2_tensor)
        revised_frequency = (w1 * f1_tensor + w2 * f2_tensor) / (w1 + w2)
        revised_confidence = (w1 + w2) / (w1 + w2 + 1.0)
        return revised_frequency, revised_confidence

    f1_array = np.clip(_to_array(f1), 0.0, 1.0)
    f2_array = np.clip(_to_array(f2), 0.0, 1.0)
    c1_array = np.clip(_to_array(c1), REVISION_EPSILON, 1.0 - REVISION_EPSILON)
    c2_array = np.clip(_to_array(c2), REVISION_EPSILON, 1.0 - REVISION_EPSILON)

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


def nars_deduction(
    f1: TensorLike,
    c1: TensorLike,
    f2: TensorLike,
    c2: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Backward-compatible alias for NARS strong deduction."""
    return deduce_truth_values(f1, c1, f2, c2)
