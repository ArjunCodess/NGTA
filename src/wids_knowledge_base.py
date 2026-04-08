from __future__ import annotations

from typing import Sequence

import numpy as np

from .knowledge_base import SymbolicKnowledgeResult

WIDS_SYMBOLIC_RULES: dict[str, tuple[float, float]] = {
    "rule_lactate": (0.85, 0.80),
    "rule_hypotension": (0.75, 0.70),
    "rule_age": (0.65, 0.60),
    "rule_creatinine": (0.70, 0.65),
}

RULE_TO_FEATURE_NAME: dict[str, str] = {
    "rule_lactate": "d1_lactate_max",
    "rule_hypotension": "d1_sysbp_min",
    "rule_age": "age",
    "rule_creatinine": "d1_creatinine_max",
}

RULE_ORDER: tuple[str, ...] = tuple(WIDS_SYMBOLIC_RULES.keys())


def build_wids_symbolic_truth_matrices(
    rule_triggers: np.ndarray,
    feature_names: Sequence[str],
    rule_names: Sequence[str] | None = None,
) -> SymbolicKnowledgeResult:
    trigger_array = np.asarray(rule_triggers, dtype=bool)
    active_rule_names = tuple(rule_names or RULE_ORDER)
    feature_name_list = list(feature_names)
    feature_index = {name: index for index, name in enumerate(feature_name_list)}

    n_cases = int(trigger_array.shape[0])
    n_features = int(len(feature_name_list))
    symbolic_frequency = np.zeros((n_cases, n_features), dtype=np.float64)
    symbolic_confidence = np.zeros((n_cases, n_features), dtype=np.float64)
    symbolic_trigger_mask = np.zeros((n_cases, n_features), dtype=bool)
    patient_rule_counts = trigger_array.sum(axis=1, dtype=np.int64)
    patient_any_rule_triggered = patient_rule_counts > 0
    rule_trigger_counts: dict[str, int] = {}
    mapped_rule_trigger_counts: dict[str, int] = {}

    if trigger_array.ndim != 2 or trigger_array.shape[1] != len(active_rule_names):
        raise ValueError("Rule trigger array shape does not match the expected rule ordering.")

    for column_index, rule_name in enumerate(active_rule_names):
        triggered_patients = trigger_array[:, column_index]
        trigger_count = int(triggered_patients.sum())
        rule_trigger_counts[rule_name] = trigger_count

        feature_name = RULE_TO_FEATURE_NAME.get(rule_name)
        feature_position = feature_index.get(feature_name) if feature_name is not None else None
        if feature_position is None:
            mapped_rule_trigger_counts[rule_name] = 0
            continue

        frequency_value, confidence_value = WIDS_SYMBOLIC_RULES[rule_name]
        patient_indices = np.flatnonzero(triggered_patients)
        symbolic_frequency[patient_indices, feature_position] = frequency_value
        symbolic_confidence[patient_indices, feature_position] = confidence_value
        symbolic_trigger_mask[patient_indices, feature_position] = True
        mapped_rule_trigger_counts[rule_name] = int(len(patient_indices))

    return SymbolicKnowledgeResult(
        symbolic_frequency=symbolic_frequency,
        symbolic_confidence=symbolic_confidence,
        symbolic_trigger_mask=symbolic_trigger_mask,
        rule_trigger_counts=rule_trigger_counts,
        mapped_rule_trigger_counts=mapped_rule_trigger_counts,
        patient_rule_counts=patient_rule_counts,
        patient_any_rule_triggered=patient_any_rule_triggered,
        total_trigger_count=int(sum(rule_trigger_counts.values())),
        mapped_feature_trigger_count=int(symbolic_trigger_mask.sum()),
    )
