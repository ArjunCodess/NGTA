import numpy as np
import pandas as pd
import pytest

from src.knowledge_base import SYMBOLIC_RULES, build_symbolic_truth_matrices, validate_rule_registry, validate_unique_rule_targets
from src.trace_replay import reference_predicate
from src.wids_loader import WIDSPreprocessor, WIDS_CONTINUOUS_COLUMNS


@pytest.mark.parametrize("rule_id, values, expected", [
    ("braf_mutation", [None, "Unknown", -1, 0, .999999, 1, 1.000001, 2], [0,0,0,0,0,1,0,0]),
    ("age_ge_55_years", [None, "Unknown", -1, 55*365.25-1e-6, 55*365.25, 55*365.25+1e-6], [0,0,0,0,1,1]),
    ("pathologic_t_t3_t4", [None,"Unknown","T3unknown","T4-1","T2","T3","T3a","T3b","T4","T4a","T4b"], [0,0,0,0,0,1,1,1,1,1,1]),
    ("extrathyroid_extension_present", [None,"Unknown","No","-1","None","Minimal (T3)","Moderate/Advanced (T4a)","Very Advanced (T4b)","Minimal (T3)unknown"], [0,0,0,0,0,1,1,1,0]),
])
def test_every_tcga_rule_matches_independent_boundary_oracle(rule_id, values, expected):
    rule = SYMBOLIC_RULES[rule_id]
    column = rule['source_column']
    frame = pd.DataFrame({column: values})
    features = ([column] if rule['target_mode']=='direct' else [f'{column}_{v}' for v in values])
    result = build_symbolic_truth_matrices(frame, features)
    np.testing.assert_array_equal(result.rule_case_masks[rule_id], expected)
    np.testing.assert_array_equal(reference_predicate(rule_id, frame[column]), expected)
    assert result.rule_trigger_counts[rule_id] == sum(expected)


@pytest.mark.parametrize("rule_id,column,boundary,direction,index", [
    ('rule_lactate','d1_lactate_max',4,1,0), ('rule_hypotension','d1_sysbp_min',90,-1,1),
    ('rule_age','age',75,1,2), ('rule_creatinine','d1_creatinine_max',2,1,3),
])
def test_every_wids_rule_uses_exact_observed_boundaries(rule_id,column,boundary,direction,index):
    values = [None,'Unknown',-1,boundary-direction*1e-8,boundary,boundary+direction*1e-8]
    frame = pd.DataFrame({c: np.ones(6) for c in WIDS_CONTINUOUS_COLUMNS})
    frame = frame.assign(age=40,d1_sysbp_min=120,d1_lactate_max=1,d1_creatinine_max=1,
                         encounter_id=range(6),hospital_death=[0,1]*3,elective_surgery=[0,1]*3,gender=['M','F']*3)
    train = frame.copy()
    frame[column] = values
    encoded = WIDSPreprocessor(imputation='median').fit(train).transform_components(frame)
    expected = [0,0,0,0,1,1]
    np.testing.assert_array_equal(encoded.rule_triggers[:,index],expected)
    np.testing.assert_array_equal(reference_predicate(rule_id, frame[column]),expected)


def test_conflicting_rule_targets_and_invalid_metadata_are_rejected():
    import copy
    rules = copy.deepcopy(SYMBOLIC_RULES)
    rules['duplicate'] = copy.deepcopy(rules['braf_mutation'])
    with pytest.raises(ValueError):
        validate_unique_rule_targets({key: rule['target_key'] for key, rule in rules.items()})
    rules = copy.deepcopy(SYMBOLIC_RULES)
    rules['braf_mutation']['expert_review']='unknown'
    with pytest.raises(ValueError):
        validate_rule_registry(rules)
