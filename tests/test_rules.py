from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.knowledge_base import (
    EXTRATHYROID_EXTENSION_VALUES,
    build_symbolic_truth_matrices,
    validate_unique_rule_targets,
)
from src.wids_loader import WIDS_CONTINUOUS_COLUMNS, WIDSPreprocessor


def test_extrathyroid_extension_rejects_missing_placeholder_and_unknown() -> None:
    source = "pathology_details.extrathyroid_extension"
    values = [
        EXTRATHYROID_EXTENSION_VALUES[0],
        "'--",
        "Unknown",
        None,
        EXTRATHYROID_EXTENSION_VALUES[2],
    ]
    frame = pd.DataFrame({source: values})
    feature_names = [f"{source}_{value}" for value in values if value is not None]

    result = build_symbolic_truth_matrices(frame, feature_names)

    assert result.rule_trigger_counts["extrathyroid_extension_present"] == 2
    assert result.patient_any_rule_triggered.tolist() == [True, False, False, False, True]


def test_duplicate_rule_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="Multiple symbolic rules target"):
        validate_unique_rule_targets({"rule_a": "feature", "rule_b": "feature"})


def test_wids_trigger_boundaries_are_inclusive() -> None:
    rows: list[dict[str, object]] = []
    for index in range(6):
        row: dict[str, object] = {
            "encounter_id": index,
            "hospital_death": index % 2,
            "elective_surgery": 0,
            "gender": "F" if index % 2 == 0 else "M",
        }
        row.update({column: 1.0 for column in WIDS_CONTINUOUS_COLUMNS})
        rows.append(row)

    rows[0].update({"d1_lactate_max": 4.0, "d1_sysbp_min": 90.0, "age": 75.0, "d1_creatinine_max": 2.0})
    rows[1].update({"d1_lactate_max": 3.999, "d1_sysbp_min": 90.001, "age": 74.999, "d1_creatinine_max": 1.999})
    frame = pd.DataFrame(rows)

    encoded = WIDSPreprocessor().fit(frame).transform_components(frame.iloc[:2].copy())

    np.testing.assert_array_equal(encoded.rule_triggers[0], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(encoded.rule_triggers[1], np.zeros(4, dtype=np.float32))
