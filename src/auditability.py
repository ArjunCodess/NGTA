from __future__ import annotations

from typing import Any

import numpy as np

from .nars_interface import revise_truth_values


def compute_operational_audit(
    *,
    dataset: str,
    patient_any_rule_triggered: np.ndarray,
    total_rule_trigger_count: int,
    mapped_feature_trigger_count: int,
    symbolic_trigger_mask: np.ndarray,
    neural_frequency: np.ndarray,
    neural_confidence: np.ndarray,
    symbolic_frequency: np.ndarray,
    symbolic_confidence: np.ndarray,
    revised_frequency: np.ndarray,
    revised_confidence: np.ndarray,
    attention_before: np.ndarray,
    attention_after: np.ndarray,
    baseline_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    gamma: float,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Quantify trace completeness and arithmetic fidelity, not clinical usability."""
    trigger_mask = np.asarray(symbolic_trigger_mask, dtype=bool)
    any_trigger = np.asarray(patient_any_rule_triggered, dtype=bool)
    attention = np.asarray(attention_before, dtype=np.float64)
    gated_attention = np.asarray(attention_after, dtype=np.float64)
    neural_f = np.asarray(neural_frequency, dtype=np.float64)
    neural_c = np.asarray(neural_confidence, dtype=np.float64)
    symbolic_f = np.asarray(symbolic_frequency, dtype=np.float64)
    symbolic_c = np.asarray(symbolic_confidence, dtype=np.float64)
    revised_f = np.asarray(revised_frequency, dtype=np.float64)
    revised_c = np.asarray(revised_confidence, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    gated = np.asarray(gated_probabilities, dtype=np.float64)

    expected_shape = attention.shape
    feature_arrays = (
        trigger_mask,
        neural_f,
        neural_c,
        symbolic_f,
        symbolic_c,
        revised_f,
        revised_c,
        gated_attention,
    )
    if any(array.shape != expected_shape for array in feature_arrays):
        raise ValueError("All feature-level audit arrays must share the attention shape.")
    if attention.ndim != 2 or any_trigger.shape != (attention.shape[0],):
        raise ValueError("Audit inputs must contain a case-by-feature matrix and one trigger flag per case.")
    if baseline.shape != any_trigger.shape or gated.shape != any_trigger.shape:
        raise ValueError("Probability arrays must contain one value per audited case.")

    event_count = int(trigger_mask.sum())
    case_count = int(attention.shape[0])
    cases_with_trigger = int(any_trigger.sum())

    finite_fields = (
        np.column_stack(
            [
                array[trigger_mask]
                for array in (
                    neural_f,
                    neural_c,
                    symbolic_f,
                    symbolic_c,
                    revised_f,
                    revised_c,
                    attention,
                    gated_attention,
                )
            ]
        )
        if event_count
        else np.empty((0, 8), dtype=np.float64)
    )
    finite_event_count = int(np.isfinite(finite_fields).all(axis=1).sum()) if event_count else 0

    expected_f, expected_c = revise_truth_values(neural_f, neural_c, symbolic_f, symbolic_c)
    revision_residual = np.maximum(
        np.abs(np.asarray(expected_f) - revised_f),
        np.abs(np.asarray(expected_c) - revised_c),
    )
    max_revision_residual = float(np.max(revision_residual[trigger_mask])) if event_count else 0.0

    normalized_attention = attention / np.clip(attention.sum(axis=-1, keepdims=True), 1e-8, None)
    expected_gate = normalized_attention * np.power(np.clip(revised_c, 0.0, 1.0), float(gamma))
    denominator = expected_gate.sum(axis=-1, keepdims=True)
    expected_gate = np.divide(expected_gate, np.where(denominator > 0.0, denominator, 1.0))
    expected_gate = np.where(denominator > 0.0, expected_gate, normalized_attention)
    max_gate_residual = float(np.max(np.abs(expected_gate - gated_attention))) if case_count else 0.0

    confidence_changed = np.abs(revised_c - neural_c) > tolerance
    attention_changed = np.abs(gated_attention - attention) > tolerance
    probability_delta = np.abs(gated - baseline)
    triggered_probability_delta = probability_delta[any_trigger]
    baseline_labels = baseline >= 0.5
    gated_labels = gated >= 0.5

    return {
        "dataset": dataset,
        "held_out_cases": case_count,
        "cases_with_any_trigger": cases_with_trigger,
        "case_coverage": cases_with_trigger / case_count if case_count else 0.0,
        "total_rule_trigger_count": int(total_rule_trigger_count),
        "mapped_feature_trigger_count": int(mapped_feature_trigger_count),
        "mapping_coverage": mapped_feature_trigger_count / total_rule_trigger_count if total_rule_trigger_count else 1.0,
        "finite_trace_event_count": finite_event_count,
        "finite_trace_rate": finite_event_count / event_count if event_count else 1.0,
        "confidence_changed_event_count": int((confidence_changed & trigger_mask).sum()),
        "confidence_changed_event_rate": float((confidence_changed & trigger_mask).sum() / event_count) if event_count else 0.0,
        "attention_changed_event_count": int((attention_changed & trigger_mask).sum()),
        "attention_changed_event_rate": float((attention_changed & trigger_mask).sum() / event_count) if event_count else 0.0,
        "mean_abs_probability_delta_triggered": float(np.mean(triggered_probability_delta)) if cases_with_trigger else 0.0,
        "median_abs_probability_delta_triggered": float(np.median(triggered_probability_delta)) if cases_with_trigger else 0.0,
        "max_abs_probability_delta_triggered": float(np.max(triggered_probability_delta)) if cases_with_trigger else 0.0,
        "threshold_flip_count_triggered": int(((baseline_labels != gated_labels) & any_trigger).sum()),
        "max_revision_residual": max_revision_residual,
        "max_gate_residual": max_gate_residual,
        "audit_tolerance": float(tolerance),
        "revision_fidelity_pass": bool(max_revision_residual <= tolerance),
        "gate_fidelity_pass": bool(max_gate_residual <= tolerance),
    }
