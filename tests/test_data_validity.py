import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN, GENOMIC_FEATURE_PREFIX,
    TabularPreprocessor, _collapse_case_table, _load_genomic_binary_matrix,
)
from src.knowledge_base import build_symbolic_truth_matrices
from src.wids_loader import WIDS_CONTINUOUS_COLUMNS, WIDSPreprocessor, clean_numeric, grouped_split


def test_collapse_preserves_one_real_record():
    frame = pd.DataFrame({DEFAULT_ID_COLUMN: ["a", "a"], "x": [1, np.nan], "y": [np.nan, 2]})
    collapsed = _collapse_case_table(frame)
    assert collapsed.loc[0, "x"] == 1
    assert pd.isna(collapsed.loc[0, "y"])


def test_genomic_encoding_preserves_unknown_and_negative():
    gene = GENOMIC_FEATURE_PREFIX + "BRAF"
    train = pd.DataFrame({DEFAULT_ID_COLUMN: ["a", "b", "c"], DEFAULT_TARGET_COLUMN: [0, 1, 0], gene: [1, 0, np.nan]})
    preprocessor = TabularPreprocessor(numeric_columns=(), categorical_columns=(), binary_columns=(gene,)).fit(train)
    encoded = preprocessor.transform(train)
    np.testing.assert_array_equal(encoded.features, [[1, 0], [0, 0], [0, 1]])
    assert preprocessor.feature_names == [gene, "missing__" + gene]


def test_sparse_selection_uses_training_only():
    train = pd.DataFrame({DEFAULT_ID_COLUMN: ["a", "b", "c", "d"], DEFAULT_TARGET_COLUMN: [0, 1, 0, 1], "x": [np.nan, np.nan, np.nan, 4], "z": [1, 2, 3, 4]})
    processor = TabularPreprocessor(numeric_columns=("x", "z"), categorical_columns=()).fit(train)
    assert processor.dropped_missing_columns == ["x"]
    holdout = train.assign(x=1000)
    assert processor.transform(holdout).features.shape == (4, 1)


def test_extension_negative_and_unseen_categories_never_revise():
    column = "pathology_details.extrathyroid_extension"
    values = [None, "Unknown", "No", "-1", "Minimal (T3)", "Very Advanced (T4b)"]
    frame = pd.DataFrame({column: values})
    result = build_symbolic_truth_matrices(frame, [f"{column}_{v}" for v in values])
    np.testing.assert_array_equal(result.patient_any_rule_triggered, [0, 0, 0, 0, 1, 1])
    assert result.mapped_feature_trigger_count == 2


def test_apache_invalid_scores_become_missing():
    column = "apache_4a_hospital_death_prob"
    result = clean_numeric(pd.DataFrame({column: [-1, 0, 1, 2, np.inf]}))
    assert result[column].isna().tolist() == [True, False, False, True, True]


def test_missing_wids_values_do_not_trigger_observed_rules():
    frame = pd.DataFrame({c: [5.0, 6.0, 7.0, 8.0] for c in WIDS_CONTINUOUS_COLUMNS})
    frame = frame.assign(age=[75, 74, 76, 73], encounter_id=range(4), hospital_death=[0, 1, 0, 1], gender=["M", "F", "M", "F"], elective_surgery=[0, 1, 0, 1])
    processor = WIDSPreprocessor().fit(frame)
    test = frame.copy()
    test.loc[0, "d1_lactate_max"] = np.nan
    encoded = processor.transform_components(test)
    assert encoded.imputed_rule_triggers[0, 0] == 1
    assert encoded.rule_triggers[0, 0] == 0
    assert encoded.rule_triggers[1, 0] == 1
    assert encoded.numeric_missing[0, processor.numeric_columns.index("d1_lactate_max")]


def test_hospital_and_patient_partitions_are_disjoint():
    frame = pd.DataFrame({"hospital_id": np.repeat(range(20), 4), "patient_id": np.repeat(range(40), 2), "hospital_death": np.tile([0, 1], 40)})
    for column in ("hospital_id", "patient_id"):
        train, val, test = grouped_split(frame, column, 0)
        assert not set(train[column]) & set(val[column])
        assert not set(train[column]) & set(test[column])
        assert not set(val[column]) & set(test[column])
    with pytest.raises(ValueError, match="Missing split group"):
        grouped_split(frame.assign(hospital_id=np.nan), "hospital_id", 0)
