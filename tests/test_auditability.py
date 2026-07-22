from __future__ import annotations

import json

import numpy as np

from src.attention_hook import apply_confidence_gate, revise_attention_truths
from src.auditability import compute_operational_audit
from src.pipeline import _bootstrap_metric_intervals, _json_feature_trace


def _audit_fixture() -> tuple[dict[str, object], object, np.ndarray, np.ndarray]:
    attention = np.asarray([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    variance = np.asarray([[0.02, 0.03], [0.01, 0.02]], dtype=np.float64)
    symbolic_f = np.asarray([[0.8, 0.0], [0.0, 0.7]], dtype=np.float64)
    symbolic_c = np.asarray([[0.7, 0.0], [0.0, 0.6]], dtype=np.float64)
    trigger_mask = np.asarray([[True, False], [False, True]])
    truths = revise_attention_truths(attention, variance, symbolic_f, symbolic_c, trigger_mask)
    gated_attention = apply_confidence_gate(attention, truths.revised_confidence, gamma=2.0)
    audit = compute_operational_audit(
        dataset="fixture",
        patient_any_rule_triggered=np.asarray([True, True]),
        total_rule_trigger_count=2,
        mapped_feature_trigger_count=2,
        symbolic_trigger_mask=trigger_mask,
        neural_frequency=truths.neural_frequency,
        neural_confidence=truths.neural_confidence,
        symbolic_frequency=symbolic_f,
        symbolic_confidence=symbolic_c,
        revised_frequency=truths.revised_frequency,
        revised_confidence=truths.revised_confidence,
        attention_before=attention,
        attention_after=gated_attention,
        baseline_probabilities=np.asarray([0.49, 0.8]),
        gated_probabilities=np.asarray([0.51, 0.79]),
        gamma=2.0,
    )
    return audit, truths, attention, gated_attention


def test_operational_audit_reports_complete_faithful_traces() -> None:
    audit, _, _, _ = _audit_fixture()

    assert audit["case_coverage"] == 1.0
    assert audit["mapping_coverage"] == 1.0
    assert audit["finite_trace_rate"] == 1.0
    assert audit["threshold_flip_count_triggered"] == 1
    assert audit["revision_fidelity_pass"] is True
    assert audit["gate_fidelity_pass"] is True
    assert audit["max_revision_residual"] <= 1e-9
    assert audit["max_gate_residual"] <= 1e-9


def test_case_trace_contains_rule_identity_and_condition() -> None:
    _, truths, attention, gated_attention = _audit_fixture()
    trace = _json_feature_trace(
        feature_names=["feature_a", "feature_b"],
        patient_index=0,
        neural_frequency=truths.neural_frequency,
        neural_confidence=truths.neural_confidence,
        symbolic_frequency=np.asarray([[0.8, 0.0], [0.0, 0.7]]),
        symbolic_confidence=np.asarray([[0.7, 0.0], [0.0, 0.6]]),
        revised_frequency=truths.revised_frequency,
        revised_confidence=truths.revised_confidence,
        attention_mean=attention,
        gated_attention=gated_attention,
        symbolic_trigger_mask=np.asarray([[True, False], [False, True]]),
        rule_definitions={
            "rule_a": {
                "source_column": "feature_a",
                "condition": "feature_a >= threshold",
                "description": "Prototype evidence for feature A.",
            }
        },
    )
    record = json.loads(trace)[0]

    assert record["rule_id"] == "rule_a"
    assert record["rule_condition"] == "feature_a >= threshold"


def test_bootstrap_includes_mc_confidence_paired_deltas() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = {
        "baseline": np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
        "flat_confidence": np.asarray([0.11, 0.19, 0.31, 0.69, 0.81, 0.89]),
        "mc_confidence_only": np.asarray([0.12, 0.18, 0.29, 0.71, 0.79, 0.88]),
        "nars_gated": np.asarray([0.1, 0.17, 0.28, 0.72, 0.82, 0.9]),
    }

    result = _bootstrap_metric_intervals(labels, probabilities, iterations=50, seed=7)

    comparisons = result["comparison"]
    assert "mc_confidence_only_vs_nars_gated_brier_delta_left_minus_right" in comparisons
    assert "mc_confidence_only_vs_nars_gated_ece_delta_ci_95_upper" in comparisons
