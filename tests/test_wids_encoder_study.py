import numpy as np
import pytest

from scripts.evaluate_wids_encoder import brier_differences, evaluate, paired_intervals


def test_encoder_hierarchical_intervals_preserve_known_paired_effects():
    labels = np.array([0, 1, 0, 1])
    groups = np.array([10, 10, 20, 20])
    effects = np.array([-.002, .001, -.003])
    losses = np.broadcast_to(effects[None, :, None], (5, 3, 4))
    result = paired_intervals(losses, labels, groups, iterations=30)
    for row, effect in zip(result.values(), effects):
        assert row['brier'] == pytest.approx(effect)
        assert row['lower_95'] == pytest.approx(effect)
        assert row['family_upper_95'] == pytest.approx(effect)
    with pytest.raises(ValueError, match='finite'):
        paired_intervals(losses[:, :, :2], labels, groups)


def test_encoder_experiment_cannot_write_into_original_studies(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        evaluate(tmp_path, tmp_path / 'hospital_study/knn')


def test_tiny_encoder_effect_uses_double_precision_before_squaring():
    labels = np.zeros(2, dtype=np.float32)
    predictions = dict(encoder_nars=np.array([.99, .99], dtype=np.float32),
        encoder_mc=np.array([.9900001, .9900001], dtype=np.float32),
        readout_nars=np.array([.99, .99], dtype=np.float32),
        readout_mc=np.array([.99, .99], dtype=np.float32))
    result = brier_differences(predictions, labels)[0].mean()
    expected = float(predictions['encoder_nars'][0])**2-float(predictions['encoder_mc'][0])**2
    assert result == pytest.approx(expected, abs=1e-15)
