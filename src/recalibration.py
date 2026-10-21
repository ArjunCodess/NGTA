"""Logit calibration fitted exclusively on a designated validation partition."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def probability_logits(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Calibration requires a finite probability vector")
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p))[:, None]


def validation_recalibrator(labels, probabilities):
    y = np.asarray(labels)
    logits = probability_logits(probabilities)
    if y.shape != (len(logits),) or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Calibration requires aligned validation labels from both classes")
    return LogisticRegression(C=1.0, max_iter=2000).fit(logits, y)
