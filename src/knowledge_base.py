from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

SYMBOLIC_RULES: dict[str, dict[str, Any]] = {
    "braf_mutation": {
        "description": "BRAF mutation strongly predicts lymph node metastasis.",
        "source_column": "genomic_mutation__BRAF",
        "target_mode": "direct",
        "truth_value": {"frequency": 0.85, "confidence": 0.75},
    },
    "age_ge_55_years": {
        "description": "Age at diagnosis >= 55 years increases metastasis risk.",
        "source_column": "diagnoses.age_at_diagnosis",
        "target_mode": "direct",
        "truth_value": {"frequency": 0.70, "confidence": 0.60},
    },
    "pathologic_t_t3_t4": {
        "description": "Pathologic T3/T4 disease strongly predicts nodal spread.",
        "source_column": "diagnoses.ajcc_pathologic_t",
        "target_mode": "categorical_active",
        "truth_value": {"frequency": 0.90, "confidence": 0.85},
    },
    "extrathyroid_extension_present": {
        "description": "Any recorded extrathyroid extension increases metastasis risk.",
        "source_column": "pathology_details.extrathyroid_extension",
        "target_mode": "categorical_active",
        "truth_value": {"frequency": 0.85, "confidence": 0.80},
    },
}


@dataclass
class SymbolicKnowledgeResult:
    symbolic_frequency: np.ndarray
    symbolic_confidence: np.ndarray
    symbolic_trigger_mask: np.ndarray
    rule_trigger_counts: dict[str, int]
    mapped_rule_trigger_counts: dict[str, int]
    patient_rule_counts: np.ndarray
    patient_any_rule_triggered: np.ndarray
    total_trigger_count: int
    mapped_feature_trigger_count: int


def _categorical_feature_name(column: str, value: Any) -> str:
    return f"{column}_{value}"


def _assign_truth_value(
    frequency: np.ndarray,
    confidence: np.ndarray,
    trigger_mask: np.ndarray,
    patient_index: int,
    feature_index: int,
    frequency_value: float,
    confidence_value: float,
) -> None:
    frequency[patient_index, feature_index] = frequency_value
    confidence[patient_index, feature_index] = confidence_value
    trigger_mask[patient_index, feature_index] = True


def build_symbolic_truth_matrices(
    case_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> SymbolicKnowledgeResult:
    feature_name_list = list(feature_names)
    feature_index = {name: index for index, name in enumerate(feature_name_list)}
    n_cases = int(len(case_frame))
    n_features = int(len(feature_name_list))

    symbolic_frequency = np.zeros((n_cases, n_features), dtype=np.float64)
    symbolic_confidence = np.zeros((n_cases, n_features), dtype=np.float64)
    symbolic_trigger_mask = np.zeros((n_cases, n_features), dtype=bool)
    patient_rule_counts = np.zeros(n_cases, dtype=np.int64)
    rule_trigger_counts: dict[str, int] = {}
    mapped_rule_trigger_counts: dict[str, int] = {}

    if n_cases == 0 or n_features == 0:
        return SymbolicKnowledgeResult(
            symbolic_frequency=symbolic_frequency,
            symbolic_confidence=symbolic_confidence,
            symbolic_trigger_mask=symbolic_trigger_mask,
            rule_trigger_counts=rule_trigger_counts,
            mapped_rule_trigger_counts=mapped_rule_trigger_counts,
            patient_rule_counts=patient_rule_counts,
            patient_any_rule_triggered=np.zeros(n_cases, dtype=bool),
            total_trigger_count=0,
            mapped_feature_trigger_count=0,
        )

    for rule_id, rule in SYMBOLIC_RULES.items():
        frequency_value = float(rule["truth_value"]["frequency"])
        confidence_value = float(rule["truth_value"]["confidence"])
        source_column = str(rule["source_column"])
        triggered_patients = np.zeros(n_cases, dtype=bool)
        mapped_patients = np.zeros(n_cases, dtype=bool)

        if source_column not in case_frame.columns:
            rule_trigger_counts[rule_id] = 0
            mapped_rule_trigger_counts[rule_id] = 0
            continue

        if rule_id == "braf_mutation":
            series = pd.to_numeric(case_frame[source_column], errors="coerce").fillna(0.0)
            triggered_patients = series.eq(1.0).to_numpy(dtype=bool)
            feature_position = feature_index.get(source_column)
            if feature_position is not None:
                patient_indices = np.flatnonzero(triggered_patients)
                for patient_index in patient_indices:
                    _assign_truth_value(
                        symbolic_frequency,
                        symbolic_confidence,
                        symbolic_trigger_mask,
                        int(patient_index),
                        feature_position,
                        frequency_value,
                        confidence_value,
                    )
                mapped_patients = triggered_patients.copy()

        elif rule_id == "age_ge_55_years":
            age_years = pd.to_numeric(case_frame[source_column], errors="coerce") / 365.25
            triggered_patients = age_years.ge(55.0).fillna(False).to_numpy(dtype=bool)
            feature_position = feature_index.get(source_column)
            if feature_position is not None:
                patient_indices = np.flatnonzero(triggered_patients)
                for patient_index in patient_indices:
                    _assign_truth_value(
                        symbolic_frequency,
                        symbolic_confidence,
                        symbolic_trigger_mask,
                        int(patient_index),
                        feature_position,
                        frequency_value,
                        confidence_value,
                    )
                mapped_patients = triggered_patients.copy()

        elif rule_id == "pathologic_t_t3_t4":
            stage_series = case_frame[source_column].fillna("").astype(str)
            triggered_patients = stage_series.str.startswith(("T3", "T4")).to_numpy(dtype=bool)
            for patient_index, stage_value in enumerate(stage_series):
                if not triggered_patients[patient_index]:
                    continue
                feature_name = _categorical_feature_name(source_column, stage_value)
                feature_position = feature_index.get(feature_name)
                if feature_position is None:
                    continue
                _assign_truth_value(
                    symbolic_frequency,
                    symbolic_confidence,
                    symbolic_trigger_mask,
                    patient_index,
                    feature_position,
                    frequency_value,
                    confidence_value,
                )
                mapped_patients[patient_index] = True

        elif rule_id == "extrathyroid_extension_present":
            extension_series = case_frame[source_column]
            triggered_patients = extension_series.notna().to_numpy(dtype=bool)
            for patient_index, extension_value in enumerate(extension_series):
                if pd.isna(extension_value):
                    continue
                feature_name = _categorical_feature_name(source_column, extension_value)
                feature_position = feature_index.get(feature_name)
                if feature_position is None:
                    continue
                _assign_truth_value(
                    symbolic_frequency,
                    symbolic_confidence,
                    symbolic_trigger_mask,
                    patient_index,
                    feature_position,
                    frequency_value,
                    confidence_value,
                )
                mapped_patients[patient_index] = True

        rule_trigger_counts[rule_id] = int(triggered_patients.sum())
        mapped_rule_trigger_counts[rule_id] = int(mapped_patients.sum())
        patient_rule_counts += triggered_patients.astype(np.int64)

    patient_any_rule_triggered = patient_rule_counts > 0
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
