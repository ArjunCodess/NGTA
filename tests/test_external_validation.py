from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.external_validation import harmonize_cohort


def inputs():
    processor = SimpleNamespace(id_column="case", target_column="outcome", numeric_columns=["lactate"],
                                binary_columns=[], categorical_columns=[])
    frame = pd.DataFrame({"id": ["new1", "new2"], "pid": ["patient1", "patient2"], "death": [0, 1], "hospital": [1, 2], "lab": [10., np.nan]})
    policy = dict(cohort="independent", source_version="1", prediction_landmark="end of first day",
                  outcome_definition="hospital_mortality", independence_evidence="different institutions",
                  id_column="id", patient_id_column="pid", target_column="death", cluster_column="hospital",
                  features={"lactate": dict(column="lab", source_unit="mg/dl", target_unit="mmol/l",
                                           scale=.111, availability="observed", measurement_window="first day",
                                           evidence="verified source dictionary")})
    return frame, policy, processor


def test_external_mapping_converts_units_preserves_unknown_and_rejects_overlap():
    frame, policy, processor = inputs()
    mapped, groups = harmonize_cohort(frame, policy, processor, ["old"])
    assert mapped.lactate.iloc[0] == 1.11
    assert np.isnan(mapped.lactate.iloc[1])
    np.testing.assert_array_equal(groups, [1, 2])
    with pytest.raises(ValueError, match="overlap"):
        harmonize_cohort(frame, policy, processor, ["new2"])
    with pytest.raises(ValueError, match="patients overlap"):
        harmonize_cohort(frame, policy, processor, [], ["patient2"])


def test_external_mapping_rejects_leaked_outcomes_and_undocumented_inputs():
    frame, policy, processor = inputs()
    leaked = deepcopy(policy)
    leaked["features"]["lactate"]["column"] = "death"
    with pytest.raises(ValueError, match="predictors"):
        harmonize_cohort(frame, leaked, processor, [])
    del policy["features"]["lactate"]["measurement_window"]
    with pytest.raises(ValueError, match="evidence"):
        harmonize_cohort(frame, policy, processor, [])
