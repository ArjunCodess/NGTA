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


def test_external_mapping_requires_rule_only_raw_inputs():
    frame, policy, processor=inputs()
    with pytest.raises(ValueError,match="every frozen input"):
        harmonize_cohort(frame,policy,processor,[],rule_source_columns=["stage"])
    frame["pathology"]=["T3","T2"]
    policy["features"]["stage"]=dict(column="pathology",source_unit="category",target_unit="category",
        availability="observed",measurement_window="post pathology",evidence="source dictionary",kind="categorical")
    mapped,_=harmonize_cohort(frame,policy,processor,[],rule_source_columns=["stage"])
    assert mapped.stage.tolist()==["T3","T2"]


def test_frozen_external_fixture_replays_and_rejects_partition_tampering(tmp_path,monkeypatch):
    import hashlib
    import json
    import joblib
    import torch
    from src.external_validation import evaluate_external
    from src.neural_encoder import TabularTransformerClassifier
    from src.trace_replay import replay_bundle
    from src.wids_loader import WIDSPreprocessor,WIDS_CONTINUOUS_COLUMNS
    from src.wids_knowledge_base import WIDS_RULE_DEFINITIONS

    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    n=12
    train=pd.DataFrame({column:np.linspace(1,10,n) for column in WIDS_CONTINUOUS_COLUMNS})
    train["encounter_id"]=[f"dev{i}" for i in range(n)]
    train["hospital_death"]=np.arange(n)%2
    train["gender"]="M"
    train["elective_surgery"]=np.arange(n)%2
    processor=WIDSPreprocessor(imputation="median").fit(train)
    checkpoint=tmp_path/"checkpoint"
    (checkpoint/"traces").mkdir(parents=True)
    manifest=train[["encounter_id"]].assign(split="train")
    manifest.to_csv(checkpoint/"traces/split_ids.csv",index=False)
    pd.DataFrame({"patient_id":[f"devpatient{i}" for i in range(n)]}).to_csv(checkpoint/"traces/development_groups.csv",index=False)
    model=TabularTransformerClassifier(processor.input_dim,8,2,1,.2)
    config=dict(dataset="wids",d_model=8,num_heads=2,num_layers=1,dropout=.2,batch_size=6,gamma=2.)
    torch.save(dict(state_dict=model.state_dict(),config=config,input_dim=processor.input_dim,
        feature_names=processor.feature_names,training_spec=dict(rules=WIDS_RULE_DEFINITIONS,
        split_ids_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest())),checkpoint/"model.pt")
    joblib.dump(processor,checkpoint/"preprocessor.joblib")
    external=train.copy()
    external["encounter_id"]=[f"external{i}" for i in range(n)]
    external["patient_id"]=[f"externalpatient{i}" for i in range(n)]
    external["hospital_id"]=np.arange(n)//4
    cohort=tmp_path/"external.csv"
    external.to_csv(cohort,index=False)
    columns=processor.numeric_columns+processor.binary_columns+processor.categorical_columns
    policy=dict(cohort="synthetic fixture only",source_version="1",prediction_landmark="first day",
        outcome_definition="hospital_mortality",development_dataset="wids",independence_evidence="generated fixture IDs",
        id_column="encounter_id",patient_id_column="patient_id",target_column="hospital_death",cluster_column="hospital_id",
        features={name:dict(column=name,source_unit="fixture units",target_unit="fixture units",availability="observed",
        measurement_window="fixture first day",evidence="synthetic values") for name in columns})
    mapping=tmp_path/"mapping.json"
    mapping.write_text(json.dumps(policy))
    output=tmp_path/"output"
    evaluate_external(checkpoint,cohort,mapping,output,mc_samples=2)
    assert replay_bundle(output/"traces")["passed"]
    manifest.loc[0,"split"]="test"
    manifest.to_csv(checkpoint/"traces/split_ids.csv",index=False)
    with pytest.raises(ValueError,match="partitions differ"):
        evaluate_external(checkpoint,cohort,mapping,output,mc_samples=2)
