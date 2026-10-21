"""Compare MC dropout with ensemble and entropy summaries.

The clinical tables remain single-seed MC dropout until an ensemble run is
saved. This module only summarizes estimators that were actually computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def deep_ensemble_statistics(member_probabilities: np.ndarray) -> dict[str, Any]:
    """Mean and variance across independently seeded ensemble members."""
    members = np.asarray(member_probabilities, dtype=np.float64)
    if members.ndim != 2 or members.shape[0] < 2:
        raise ValueError("A deep ensemble comparison needs at least two seeded members.")
    return {
        "members": int(members.shape[0]),
        "mean": members.mean(axis=0),
        "variance": members.var(axis=0),
    }


def mc_predictive_entropy(pass_probabilities: np.ndarray) -> np.ndarray:
    """Bernoulli entropy of the MC predictive mean. This is not a NARS confidence."""
    passes = np.asarray(pass_probabilities, dtype=np.float64)
    if passes.ndim == 1:
        mean_probability = passes
    elif passes.ndim == 2:
        mean_probability = passes.mean(axis=0)
    else:
        raise ValueError("MC probabilities must be a vector or a pass-by-case matrix.")
    probability = np.clip(mean_probability, 1e-6, 1.0 - 1e-6)
    return -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.size < 2:
        return float("nan")

    left_rank = rankdata(left_values, method="average")
    right_rank = rankdata(right_values, method="average")
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _error_detection_auroc(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    error_labels = np.asarray(errors, dtype=np.int64)
    if np.unique(error_labels).size < 2:
        return float("nan")
    return float(roc_auc_score(error_labels, np.asarray(uncertainty, dtype=np.float64)))


def compare_uncertainty_estimators(
    estimators: dict[str, np.ndarray],
    errors: np.ndarray | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Compare named per-case uncertainty scores, optionally against errors."""
    if len(estimators) < 2:
        raise ValueError("Uncertainty comparison needs at least two estimators.")
    prepared = {name: np.asarray(values, dtype=np.float64) for name, values in estimators.items()}
    shapes = {values.shape for values in prepared.values()}
    if len(shapes) != 1:
        raise ValueError("Uncertainty estimators must share one case axis.")

    names = list(prepared)
    pairwise: dict[str, float] = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            pairwise[f"{left_name}_vs_{right_name}_spearman"] = _spearman(
                prepared[left_name],
                prepared[right_name],
            )

    report: dict[str, Any] = {
        "estimators": names,
        "seeds": list(seeds or []),
        "seed_count": int(len(seeds or [])),
        "mean_score": {name: float(np.mean(values)) for name, values in prepared.items()},
        "pairwise_spearman": pairwise,
    }
    if errors is not None:
        report["error_detection_auroc"] = {
            name: _error_detection_auroc(values, errors) for name, values in prepared.items()
        }
    return report
