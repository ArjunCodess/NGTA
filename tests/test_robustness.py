from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.apache_baselines import apache_baselines
from src.neural_encoder import TabularTransformerClassifier
from src.pipeline import PipelineConfig
from src.robustness import evaluate_raw_missingness, mask_observed_values, selective_risk
from src.uncertainty import _spearman
from src.wids_loader import WIDS_CONTINUOUS_COLUMNS, WIDSPreprocessor
from src.wids_knowledge_base import WIDS_RULE_DEFINITIONS


def test_raw_masks_never_change_missing_values_or_ids():
    frame = pd.DataFrame({"x": [np.nan, 1, 2, 3], "category": ["a", "b", "a", "b"], "id": range(4)})
    masked, mask = mask_observed_values(frame, ["x", "category"], 1, np.random.default_rng(0))
    assert not mask[0, 0]
    assert masked[["x", "category"]].isna().all().all()
    np.testing.assert_array_equal(masked.id, frame.id)
    _, low = mask_observed_values(frame, ["x", "category"], .1, np.random.default_rng(0))
    _, high = mask_observed_values(frame, ["x", "category"], .7, np.random.default_rng(0))
    assert np.all(~low | high)


def test_selective_risk_and_correlation_handle_ties():
    curve, area = selective_risk([0, 1, 1, 0], [.1, .9, .1, .9], [1, 1, 1, 1])
    assert len(curve) == 1
    assert area == .5
    assert np.isnan(_spearman(np.ones(4), np.arange(4)))
    assert _spearman(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1


def test_apache_fallback_and_calibration_are_train_only():
    column = "apache_4a_hospital_death_prob"
    frame = pd.DataFrame({column: [.1, .2, .8, .9, np.nan, -1], "hospital_death": [0, 0, 1, 1, 0, 1]})
    bundle = SimpleNamespace(preprocessor=SimpleNamespace(target_column="hospital_death"), train_frame=frame, val_frame=frame, test_frame=frame)
    scores = apache_baselines(bundle)
    assert scores["apache_raw"]["test_probabilities"][-1] == .5
    assert scores["apache_raw"]["test_probabilities"][-2] == .5
    before = scores["apache_recalibrated"]["model"].named_steps["logisticregression"].coef_.copy()
    bundle.test_frame = frame.assign(hospital_death=1-frame.hospital_death)
    after = apache_baselines(bundle)["apache_recalibrated"]["model"].named_steps["logisticregression"].coef_
    np.testing.assert_array_equal(before, after)


def test_raw_missingness_reprocesses_frozen_rules_and_preserves_model(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.uniform(1, 8, 12) for c in WIDS_CONTINUOUS_COLUMNS})
    frame["apache_4a_hospital_death_prob"] = rng.uniform(0, 1, 12)
    frame = frame.assign(encounter_id=range(12), hospital_death=np.tile([0, 1], 6), elective_surgery=np.tile([0, 1], 6), gender=np.tile(["M", "F"], 6))
    processor = WIDSPreprocessor().fit(frame)
    bundle = SimpleNamespace(preprocessor=processor, test_frame=frame)
    torch.manual_seed(0)
    model = TabularTransformerClassifier(processor.input_dim, 8, 2, 1, .2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    config = PipelineConfig(dataset="wids", mc_samples=2, batch_size=12)
    report = evaluate_raw_missingness(bundle, model, torch.device("cpu"), config, WIDS_RULE_DEFINITIONS, tmp_path)
    assert len(report) == 40
    assert report.retuned.eq(False).all()
    assert report.loc[report.mask_rate.eq(0), "brier_degradation"].eq(0).all()
    assert set(report.variant) == {"baseline", "mc_confidence_only", "nars_gated", "nars_imputed_rules"}
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
