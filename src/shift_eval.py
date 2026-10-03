"""Distribution-shift scoring with preprocessing, rules, and the model frozen."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SHIFT_RATES: tuple[float, ...] = (0.0, 0.10, 0.30, 0.50, 0.70)


def apply_random_feature_mask(
    features: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a fraction of already encoded cells with the scaled mean, 0.

    The replacement is an evaluation mask. It does not refit imputation or scaling.
    """
    shifted = np.array(features, dtype=np.float64, copy=True)
    fraction = float(rate)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("Shift mask rate must be between 0 and 1.")
    if fraction == 0.0 or shifted.size == 0:
        return shifted
    mask = rng.random(shifted.shape) < fraction
    shifted[mask] = 0.0
    return shifted


def freeze_fingerprint(values: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Hash frozen parameters so a shift run can prove it did not refit them."""
    return {
        name: np.ascontiguousarray(np.asarray(value)).tobytes()
        for name, value in sorted(values.items())
    }


def _metric_row(rate: float, y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(y_true, dtype=np.int64)
    row: dict[str, Any] = {
        "shift_rate": float(rate),
        "brier": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "retuned": False,
    }
    try:
        row["auc"] = float(roc_auc_score(labels, clipped))
    except ValueError:
        row["auc"] = float("nan")
    return row


def evaluate_frozen_shift(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    y_true: np.ndarray,
    fingerprint_fn: Callable[[], dict[str, np.ndarray]],
    rates: Sequence[float] = SHIFT_RATES,
    seed: int = 0,
) -> pd.DataFrame:
    """Score masks without refitting the model, rules, preprocessing, or threshold."""
    before = freeze_fingerprint(fingerprint_fn())
    rng = np.random.default_rng(seed)
    rows = [
        _metric_row(rate, y_true, predict_fn(apply_random_feature_mask(features, rate, rng)))
        for rate in rates
    ]
    after = freeze_fingerprint(fingerprint_fn())
    if before != after:
        raise RuntimeError("Distribution-shift evaluation changed a frozen parameter.")
    return pd.DataFrame(rows)
