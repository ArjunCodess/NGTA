"""APACHE-only comparators fitted on training outcomes, with explicit availability."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .evaluation import binary_metrics


APACHE_COLUMN = "apache_4a_hospital_death_prob"


def apache_baselines(bundle):
    target = bundle.preprocessor.target_column
    scores = []
    labels = []
    for name in ("train", "val", "test"):
        frame = getattr(bundle, f"{name}_frame")
        score = pd.to_numeric(frame[APACHE_COLUMN], errors="coerce")
        score = score.where(score.between(0, 1)).to_numpy(dtype=float)
        scores.append(score)
        labels.append(frame[target].to_numpy(dtype=int))
    fallback = float(labels[0].mean())
    result = {}
    raw_val = np.where(np.isfinite(scores[1]), scores[1], fallback)
    raw_test = np.where(np.isfinite(scores[2]), scores[2], fallback)
    val_metrics = binary_metrics(labels[1], raw_val)
    result["apache_raw"] = {"model": None, "best_config": {"missing_probability": fallback},
                            "val_brier": val_metrics["brier"], "val_auc": val_metrics["auc"],
                            "test_probabilities": raw_test, "test_labels": labels[2]}
    for name, transform in (("apache_only_logistic", lambda x: x),
                             ("apache_recalibrated", lambda x: np.log(np.clip(x, 1e-6, 1-1e-6) / (1-np.clip(x, 1e-6, 1-1e-6))))):
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              LogisticRegression(C=1.0, max_iter=2000))
        inputs = [transform(score)[:, None] for score in scores]
        model.fit(inputs[0], labels[0])
        val_probability = model.predict_proba(inputs[1])[:, 1]
        val_metrics = binary_metrics(labels[1], val_probability)
        result[name] = {"model": model, "best_config": {"fit_split": "train", "score_transform": "logit" if name.endswith("recalibrated") else "identity"},
                        "val_brier": val_metrics["brier"], "val_auc": val_metrics["auc"],
                        "test_probabilities": model.predict_proba(inputs[2])[:, 1], "test_labels": labels[2]}
    return result
