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

from .matched_inference import score_cached_passes
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
    cached_passes: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerun revision and confidence gating. Returns probabilities and revised confidence."""
    truths = revise_attention_truths(
        attention_mean,
        attention_var,
        symbolic_frequency=symbolic_frequency,
        symbolic_confidence=symbolic_confidence,
        symbolic_trigger_mask=symbolic_trigger_mask,
    )
    if cached_passes is not None:
        probabilities, _ = score_cached_passes(*cached_passes, truths.revised_confidence, gamma)
        return probabilities, np.asarray(truths.revised_confidence, dtype=np.float64)
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
    cached_passes: tuple | None = None,
    permutations: int = 100,
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
            cached_passes=cached_passes,
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
    # Shuffle only confidence values among fired cells, preserving predicates,
    # frequencies and the multiset of confidence assignments.
    confidence_permuted = np.array(symbolic_confidence, copy=True)
    confidence_permuted[symbolic_trigger_mask] = rng.permutation(confidence_permuted[symbolic_trigger_mask])
    rows.append(_score("confidence_assignment_permutation", symbolic_frequency, confidence_permuted, symbolic_trigger_mask))
    # Explicit arbitrary predicate controls preserve per-feature prevalence and
    # confidence/frequency distributions without clinical meaning.
    irrelevant_f = np.zeros_like(symbolic_frequency)
    irrelevant_c = np.zeros_like(symbolic_confidence)
    irrelevant_mask = np.zeros_like(symbolic_trigger_mask)
    for feature in range(symbolic_trigger_mask.shape[1]):
        active = np.flatnonzero(symbolic_trigger_mask[:, feature])
        target = rng.choice(len(labels), size=len(active), replace=False)
        irrelevant_mask[target, feature] = True
        irrelevant_f[target, feature] = symbolic_frequency[active, feature]
        irrelevant_c[target, feature] = symbolic_confidence[active, feature]
    rows.append(_score("matched_irrelevant_predicates", irrelevant_f, irrelevant_c, irrelevant_mask))
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
    for frequency_value in (0.1, 0.5, 0.9):
        changed_frequency = np.where(symbolic_trigger_mask, frequency_value, symbolic_frequency)
        rows.append(_score("symbolic_frequency", changed_frequency, symbolic_confidence, symbolic_trigger_mask))
        rows[-1]["symbolic_frequency"] = frequency_value
    for index in range(permutations):
        shuffled = shuffle_rules_across_patients(symbolic_frequency, symbolic_confidence, symbolic_trigger_mask, rng)
        rows.append(_score("predicate_permutation", *shuffled))
        rows[-1]["permutation"] = index
    # Closed-form confidence accumulation is a simpler non-frequency control.
    from .attention_hook import attention_to_nars
    _, neural_c = attention_to_nars(attention_mean, attention_var)
    a = np.clip(neural_c, 1e-6, 1 - 1e-6)
    b = np.clip(symbolic_confidence, 1e-6, 1 - 1e-6)
    boosted = np.where(symbolic_trigger_mask, (a + b - 2*a*b) / (1 - a*b), neural_c)
    if cached_passes is not None:
        probabilities, _ = score_cached_passes(*cached_passes, boosted, gamma)
    else:
        attention = apply_confidence_gate(attention_mean, boosted, gamma)
        probabilities = _sigmoid(cls_logit_mean + np.sum(attention * token_score_mean, axis=1))
    rows.append(_metric_row("confidence_boost", labels, probabilities, gamma=float(gamma), revision_rerun=False))
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


def symbolic_hypothesis_tests(frame: pd.DataFrame) -> dict:
    """Empirical permutation comparisons with Holm adjustment, retaining losses."""
    actual = frame.loc[frame.variant.eq("nars_gated")].iloc[0]
    mc = frame.loc[frame.variant.eq("mc_confidence_only")].iloc[0]
    null = frame.loc[frame.variant.eq("predicate_permutation")]
    if null.empty:
        return {"permutations": 0, "tests": {}}
    pvalues = {metric: float((1 + (null[metric] <= actual[metric]).sum()) / (len(null) + 1))
               for metric in ("brier", "log_loss")}
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for rank, metric in enumerate(ordered):
        previous = max(previous, min(1.0, pvalues[metric] * (len(ordered) - rank)))
        adjusted[metric] = previous
    return {"permutations": len(null), "brier_gain_over_mc": float(mc.brier - actual.brier),
            "minimum_prespecified_brier_gain": 1e-4,
            "tests": {metric: {"empirical_p": pvalues[metric], "holm_p": adjusted[metric],
                               "direction": "correct predicates have lower loss than permuted predicates"}
                      for metric in ordered},
            "interpretation": "development diagnostics; confirm on locked hospitals across five seeds"}
