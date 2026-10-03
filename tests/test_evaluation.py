import numpy as np
import pytest

from src.evaluation import binary_metrics, calibration_error, paired_bootstrap_indices
from src.pipeline import _bootstrap_metric_intervals, _build_reliability_frame, _compute_ece


@pytest.mark.parametrize("labels, probabilities", [
    ([0, .5], [.2, .8]), ([0, 1], [.2, np.inf]),
    ([0, 1], [-.1, .8]), ([], []), ([[0, 1]], [[.2, .8]]),
])
def test_metrics_reject_invalid_inputs_before_conversion(labels, probabilities):
    with pytest.raises(ValueError):
        binary_metrics(labels, probabilities)


def test_calibration_and_threshold_metrics_are_defined():
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([.1, .8, .2, .9])
    metrics = binary_metrics(labels, probabilities)
    assert metrics["brier"] == pytest.approx(.025)
    assert metrics["auc"] == 1
    assert metrics["pr_auc"] == 1
    assert metrics["sensitivity_0.5"] == 1
    assert metrics["specificity_0.5"] == 1
    assert metrics["ece"] == pytest.approx(_compute_ece(_build_reliability_frame(labels, probabilities)))
    assert np.isfinite(metrics["calibration_slope"])
    assert np.isfinite(metrics["log_loss"])


def test_equal_risk_quantile_bins_do_not_artificially_separate_ties():
    assert calibration_error([0, 1, 0, 1], [.5]*4, quantile=True) == 0


def test_cluster_bootstrap_keeps_entire_hospitals():
    groups = np.repeat(range(5), 4)
    labels = np.tile([0, 1, 0, 1], 5)
    samples = paired_bootstrap_indices(labels, 20, np.random.default_rng(3), groups)
    for index in samples:
        counts = np.bincount(index, minlength=len(labels))
        for group in range(5):
            assert len(np.unique(counts[groups == group])) == 1


def test_bootstrap_reports_observed_delta_and_family_intervals():
    labels = np.tile([0, 1], 10)
    probabilities = {"baseline": np.tile([.1, .9], 10), "nars_gated": np.tile([.2, .8], 10)}
    report = _bootstrap_metric_intervals(labels, probabilities, 20, 0, np.repeat(range(5), 4))
    comparison = report["comparison"]
    key = "baseline_vs_nars_gated_brier"
    assert comparison[key + "_delta_left_minus_right"] == pytest.approx(-.03)
    assert comparison[key + "_delta_family_95_lower"] <= comparison[key + "_delta_ci_95_lower"]
    assert report["baseline"]["sampling_unit"] == "hospital"
