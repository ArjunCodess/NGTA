"""Controls that recompute NAL revision after the symbolic input changes.

These ablations ask whether rule content changes the gated prediction beyond
MC-confidence gating. They do not select thresholds or truth values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .attention_hook import apply_confidence_gate, revise_attention_truths

FIXED_SYMBOLIC_PRIORS: tuple[tuple[float, float], ...] = (
    (0.50, 0.20),
    (0.50, 0.80),
    (0.90, 0.30),
    (0.90, 0.90),
)
CONFIDENCE_SCALES: tuple[float, ...] = (0.5, 0.75, 1.25, 1.5)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def _copy_triple(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(frequency, dtype=np.float64, copy=True),
        np.array(confidence, dtype=np.float64, copy=True),
        np.array(trigger_mask, dtype=bool, copy=True),
    )


def remove_all_rules(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop every symbolic revision so gating uses neural confidence only."""
    frequency_out, confidence_out, mask_out = _copy_triple(frequency, confidence, trigger_mask)
    frequency_out[:] = 0.0
    confidence_out[:] = 0.0
    mask_out[:] = False
    return frequency_out, confidence_out, mask_out


def shuffle_rules_across_patients(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute rule assignments across patients without changing trigger counts."""
    frequency_out, confidence_out, mask_out = _copy_triple(frequency, confidence, trigger_mask)
    if frequency_out.shape[0] == 0:
        return frequency_out, confidence_out, mask_out
    permutation = rng.permutation(frequency_out.shape[0])
    return frequency_out[permutation], confidence_out[permutation], mask_out[permutation]


def replace_with_random_truths(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the fired cells and replace their symbolic truth values at random."""
    frequency_out, confidence_out, mask_out = _copy_triple(frequency, confidence, trigger_mask)
    triggered = int(mask_out.sum())
    if triggered == 0:
        return frequency_out, confidence_out, mask_out
    frequency_out[mask_out] = rng.random(triggered)
    confidence_out[mask_out] = rng.uniform(0.05, 0.95, size=triggered)
    return frequency_out, confidence_out, mask_out


def replace_with_fixed_prior(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
    prior_frequency: float,
    prior_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign one symbolic prior to every fired cell, then revision must be rerun."""
    frequency_out, confidence_out, mask_out = _copy_triple(frequency, confidence, trigger_mask)
    frequency_out[mask_out] = float(prior_frequency)
    confidence_out[mask_out] = float(prior_confidence)
    return frequency_out, confidence_out, mask_out


def scale_symbolic_confidence(
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale symbolic confidence on fired cells. Revision is applied later."""
    scaled = np.array(confidence, dtype=np.float64, copy=True)
    mask = np.asarray(trigger_mask, dtype=bool)
    scaled[mask] = np.clip(scaled[mask] * float(scale), 0.0, 1.0)
    return scaled


def score_revised_gate(
    attention_mean: np.ndarray,
    attention_var: np.ndarray,
    symbolic_frequency: np.ndarray,
    symbolic_confidence: np.ndarray,
    symbolic_trigger_mask: np.ndarray,
    cls_logit_mean: np.ndarray,
    token_score_mean: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerun revision and confidence gating. Returns probabilities and revised confidence."""
    truths = revise_attention_truths(
        attention_mean,
        attention_var,
        symbolic_frequency=symbolic_frequency,
        symbolic_confidence=symbolic_confidence,
        symbolic_trigger_mask=symbolic_trigger_mask,
    )
    gated_attention = apply_confidence_gate(attention_mean, truths.revised_confidence, gamma=gamma)
    logits = np.asarray(cls_logit_mean, dtype=np.float64) + np.sum(gated_attention * token_score_mean, axis=1)
    return _sigmoid(logits), np.asarray(truths.revised_confidence, dtype=np.float64)


def _metric_row(name: str, y_true: np.ndarray, probabilities: np.ndarray, **extra: Any) -> dict[str, Any]:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    labels = np.asarray(y_true, dtype=np.int64)
    row: dict[str, Any] = {
        "variant": name,
        "brier": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }
    try:
        row["auc"] = float(roc_auc_score(labels, clipped))
    except ValueError:
        row["auc"] = float("nan")
    row.update(extra)
    return row


def build_symbolic_isolation_frame(
    y_true: np.ndarray,
    attention_mean: np.ndarray,
    attention_var: np.ndarray,
    symbolic_frequency: np.ndarray,
    symbolic_confidence: np.ndarray,
    symbolic_trigger_mask: np.ndarray,
    cls_logit_mean: np.ndarray,
    token_score_mean: np.ndarray,
    gamma: float,
    seed: int,
    rule_ids: Sequence[str] | None = None,
    rebuild_without_rules: Callable[[set[str]], Any] | None = None,
) -> pd.DataFrame:
    """Score NARS gating against controls that recompute revision."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(y_true)

    def _score(
        name: str,
        frequency: np.ndarray,
        confidence: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, Any]:
        probabilities, revised_confidence = score_revised_gate(
            attention_mean,
            attention_var,
            frequency,
            confidence,
            mask,
            cls_logit_mean,
            token_score_mean,
            gamma,
        )
        triggered = np.asarray(mask, dtype=bool)
        if int(triggered.sum()) == 0:
            revision_gap = 0.0
        else:
            revision_gap = float(np.max(np.abs(revised_confidence[triggered] - np.asarray(confidence)[triggered])))
        return _metric_row(
            name,
            labels,
            probabilities,
            gamma=float(gamma),
            revision_rerun=True,
            max_abs_revised_minus_symbolic_confidence=revision_gap,
        )

    neural_only_frequency, neural_only_confidence, neural_only_mask = remove_all_rules(
        symbolic_frequency,
        symbolic_confidence,
        symbolic_trigger_mask,
    )
    shuffled_frequency, shuffled_confidence, shuffled_mask = shuffle_rules_across_patients(
        symbolic_frequency,
        symbolic_confidence,
        symbolic_trigger_mask,
        rng,
    )
    random_frequency, random_confidence, random_mask = replace_with_random_truths(
        symbolic_frequency,
        symbolic_confidence,
        symbolic_trigger_mask,
        rng,
    )

    rows = [
        _score("mc_confidence_only", neural_only_frequency, neural_only_confidence, neural_only_mask),
        _score("nars_gated", symbolic_frequency, symbolic_confidence, symbolic_trigger_mask),
        _score("rules_removed", neural_only_frequency, neural_only_confidence, neural_only_mask),
        _score("rules_shuffled", shuffled_frequency, shuffled_confidence, shuffled_mask),
        _score("random_truth_values", random_frequency, random_confidence, random_mask),
    ]
    for prior_frequency, prior_confidence in FIXED_SYMBOLIC_PRIORS:
        prior_f, prior_c, prior_mask = replace_with_fixed_prior(
            symbolic_frequency,
            symbolic_confidence,
            symbolic_trigger_mask,
            prior_frequency,
            prior_confidence,
        )
        rows.append(
            _score(
                f"fixed_prior_f{prior_frequency:.2f}_c{prior_confidence:.2f}",
                prior_f,
                prior_c,
                prior_mask,
            )
        )
    for scale in CONFIDENCE_SCALES:
        scaled_confidence = scale_symbolic_confidence(symbolic_confidence, symbolic_trigger_mask, scale)
        rows.append(
            _score(
                "symbolic_confidence_scale",
                symbolic_frequency,
                scaled_confidence,
                symbolic_trigger_mask,
            )
        )
        rows[-1]["symbolic_confidence_scale"] = float(scale)

    if rebuild_without_rules is not None:
        for rule_id in rule_ids or ():
            rebuilt = rebuild_without_rules({rule_id})
            rows.append(
                _score(
                    f"rule_removed:{rule_id}",
                    rebuilt.symbolic_frequency,
                    rebuilt.symbolic_confidence,
                    rebuilt.symbolic_trigger_mask,
                )
            )

    frame = pd.DataFrame(rows)
    reference = frame.loc[frame["variant"] == "mc_confidence_only", "brier"]
    if not reference.empty:
        frame["brier_minus_mc_confidence_only"] = frame["brier"] - float(reference.iloc[0])
    return frame
