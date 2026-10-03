import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.matched_inference import score_cached_passes, sigmoid
from src.neural_encoder import TabularTransformerClassifier
from src.auditability import reference_revision


def test_uniform_gate_is_identity_and_keeps_covariance():
    attention = np.array([[[0.1, 0.9]], [[0.9, 0.1]]])
    scores = np.array([[[10, 0]], [[0, 10]]])
    cls = np.array([[1.0], [-1.0]])
    baseline, _ = score_cached_passes(attention, scores, cls)
    flat, _ = score_cached_passes(attention, scores, cls, np.full((1, 2), 0.5))
    np.testing.assert_allclose(baseline, flat, atol=1e-15)
    expected = sigmoid(np.array([2, 0])).mean()
    assert baseline[0] == pytest.approx(expected)
    assert abs(baseline[0] - sigmoid(cls.mean(0) + (attention.mean(0)*scores.mean(0)).sum(1))[0]) > 0.1


def test_zero_confidence_falls_back_to_ungated_pass():
    attention = np.array([[[0.2, 0.8]], [[0.7, 0.3]]])
    args = (attention, np.ones_like(attention), np.zeros((2, 1)))
    baseline, _ = score_cached_passes(*args)
    fallback, _ = score_cached_passes(*args, np.zeros((1, 2)))
    np.testing.assert_array_equal(baseline, fallback)


def test_independent_revision_known_evidence_values():
    # 9 and 3 evidence units produce frequency (9*.8 + 3*.2)/12.
    frequency, confidence = reference_revision(0.8, 0.9, 0.2, 0.75)
    assert frequency == pytest.approx(0.65)
    assert confidence == pytest.approx(12 / 13)
    frequency, confidence = reference_revision(0.2, 0, 0.8, 1)
    assert np.isfinite([frequency, confidence]).all()


def test_cached_passes_reconstruct_model_baseline():
    torch.manual_seed(4)
    model = TabularTransformerClassifier(3, d_model=8, nhead=2, num_layers=1)
    dataset = TensorDataset(torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=3)
    summary = model.predict_with_mc_dropout(loader, torch.device("cpu"), 4)
    cached = summary.attention_passes, summary.token_score_passes, summary.cls_logit_passes
    probabilities, _ = score_cached_passes(*cached)
    np.testing.assert_allclose(probabilities, summary.probabilities_mean, atol=1e-7)
    with pytest.raises(ValueError, match="sequential"):
        model.predict_with_mc_dropout(DataLoader(dataset, shuffle=True), torch.device("cpu"), 4)
    with pytest.raises(ValueError, match="at least two"):
        model.predict_with_mc_dropout(loader, torch.device("cpu"), 1)
