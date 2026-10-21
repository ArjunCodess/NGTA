"""Prespecified calibration and decision metrics for binary prediction."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_error(labels, probabilities, bins=10, quantile=False):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1))) if quantile else np.linspace(0, 1, bins + 1)
    if len(edges) == 1:
        return float(abs(p.mean() - y.mean()))
    indices = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges)-2)
    total_p = np.bincount(indices, weights=p, minlength=len(edges)-1)
    total_y = np.bincount(indices, weights=y, minlength=len(edges)-1)
    return float(np.abs(total_p-total_y).sum() / len(y))


def binary_metrics(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or not y.size or y.shape != p.shape or not np.isfinite(p).all()
            or not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Metrics require nonempty aligned binary labels and finite probabilities in [0, 1]")
    y = y.astype(int)
    p = np.clip(p, 1e-6, 1-1e-6)
    two_classes = np.unique(y).size == 2
    result = {"auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
              "pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
              "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
              "accuracy": float(np.mean((p >= .5) == y)),
              "ece": calibration_error(y, p), "ece_15": calibration_error(y, p, 15),
              "ece_quantile_10": calibration_error(y, p, quantile=True)}
    if two_classes and np.ptp(p) > 0:
        logits = np.log(p / (1-p))[:, None]
        # This is an evaluation diagnostic, never applied to test predictions.
        calibration = LogisticRegression(C=1e6, max_iter=2000).fit(logits, y)
        result.update(calibration_intercept=float(calibration.intercept_[0]), calibration_slope=float(calibration.coef_[0, 0]))
    else:
        result.update(calibration_intercept=float("nan"), calibration_slope=float("nan"))
    for threshold in (.1, .2, .5):
        predicted = p >= threshold
        tp = np.sum(predicted & (y == 1)); fp = np.sum(predicted & (y == 0))
        tn = np.sum(~predicted & (y == 0)); fn = np.sum(~predicted & (y == 1))
        suffix = f"_{threshold:.1f}"
        for name, numerator, denominator in (("sensitivity", tp, tp+fn), ("specificity", tn, tn+fp),
                                               ("ppv", tp, tp+fp), ("npv", tn, tn+fn)):
            result[name + suffix] = float(numerator / denominator) if denominator else float("nan")
    return result


def paired_bootstrap_indices(labels, iterations, rng, groups=None):
    y = np.asarray(labels)
    if np.unique(y).size != 2:
        raise ValueError("Bootstrap requires both outcome classes")
    if groups is None:
        members = None
    else:
        values = np.asarray(groups)
        if values.shape != y.shape or np.any(np.asarray([v is None or v != v for v in values])):
            raise ValueError("Cluster IDs must be present and aligned with labels")
        unique = np.unique(values)
        if len(unique) < 2:
            raise ValueError("Clustered bootstrap needs at least two clusters")
        members = [np.flatnonzero(values == group) for group in unique]
    samples = []
    for _ in range(iterations * 100):
        index = rng.integers(0, len(y), len(y)) if members is None else np.concatenate(
            [members[i] for i in rng.integers(0, len(members), len(members))])
        if np.unique(y[index]).size == 2:
            samples.append(index)
            if len(samples) == iterations:
                return samples
    raise RuntimeError("Unable to sample enough two-class bootstrap replicates")
