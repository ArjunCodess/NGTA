from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.knowledge_base import (
    RULE_GENERATION_PROCESS,
    SYMBOLIC_RULES,
    build_symbolic_truth_matrices,
    validate_rule_registry,
)
from src.neural_encoder import TabularTransformerClassifier
from src.shift_eval import evaluate_frozen_shift
from src.symbolic_ablations import (
    build_symbolic_isolation_frame,
    remove_all_rules,
    replace_with_fixed_prior,
    replace_with_random_truths,
    score_revised_gate,
    shuffle_rules_across_patients,
)
from src.uncertainty import (
    compare_uncertainty_estimators,
    deep_ensemble_statistics,
    mc_predictive_entropy,
)
from src.wids_knowledge_base import WIDS_RULE_DEFINITIONS, build_wids_symbolic_truth_matrices


def _inputs():
    attention_mean = np.array([[0.25, 0.75], [0.60, 0.40]], dtype=np.float64)
    attention_var = np.array([[0.20, 0.01], [0.01, 0.30]], dtype=np.float64)
    frequency = np.array([[0.90, 0.0], [0.0, 0.20]], dtype=np.float64)
    confidence = np.array([[0.80, 0.0], [0.0, 0.70]], dtype=np.float64)
    mask = np.array([[True, False], [False, True]])
    cls_logit = np.array([0.1, -0.2], dtype=np.float64)
    token_scores = np.array([[0.4, -0.2], [0.3, 0.5]], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    return attention_mean, attention_var, frequency, confidence, mask, cls_logit, token_scores, labels


def test_rule_removal_matches_mc_confidence_gate():
    attention_mean, attention_var, frequency, confidence, mask, cls_logit, token_scores, labels = _inputs()
    frame = build_symbolic_isolation_frame(
        labels,
        attention_mean,
        attention_var,
        frequency,
        confidence,
        mask,
        cls_logit,
        token_scores,
        gamma=2.0,
        seed=0,
    )
    removed = frame.loc[frame["variant"] == "rules_removed"].iloc[0]
    mc_only = frame.loc[frame["variant"] == "mc_confidence_only"].iloc[0]
    assert removed["brier"] == pytest.approx(mc_only["brier"])
    assert removed["log_loss"] == pytest.approx(mc_only["log_loss"])
    assert set(frame["variant"]).issuperset(
        {"rules_removed", "rules_shuffled", "random_truth_values", "nars_gated"}
    )
    assert frame["variant"].str.startswith("fixed_prior_").any()


def test_shuffle_preserves_trigger_prevalence_and_can_move_patients():
    frequency = np.arange(12, dtype=np.float64).reshape(4, 3)
    confidence = frequency / 20.0
    mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
        ]
    )
    moved = False
    for seed in range(30):
        shuffled_frequency, shuffled_confidence, shuffled_mask = shuffle_rules_across_patients(
            frequency,
            confidence,
            mask,
            np.random.default_rng(seed),
        )
        assert int(shuffled_mask.sum()) == int(mask.sum())
        assert sorted(map(tuple, shuffled_mask.tolist())) == sorted(map(tuple, mask.tolist()))
        assert shuffled_frequency.shape == frequency.shape
        assert shuffled_confidence.shape == confidence.shape
        if not np.array_equal(shuffled_mask, mask):
            moved = True
    assert moved


def test_random_and_fixed_priors_rerun_revision():
    attention_mean, attention_var, frequency, confidence, mask, cls_logit, token_scores, _labels = _inputs()
    random_frequency, random_confidence, random_mask = replace_with_random_truths(
        frequency,
        confidence,
        mask,
        np.random.default_rng(1),
    )
    assert np.array_equal(random_mask, mask)
    assert not np.allclose(random_confidence[mask], confidence[mask])

    prior_frequency, prior_confidence, prior_mask = replace_with_fixed_prior(
        frequency,
        confidence,
        mask,
        0.90,
        0.30,
    )
    assert np.allclose(prior_confidence[prior_mask], 0.30)
    _probabilities, revised_confidence = score_revised_gate(
        attention_mean,
        attention_var,
        prior_frequency,
        prior_confidence,
        prior_mask,
        cls_logit,
        token_scores,
        gamma=2.0,
    )
    assert float(np.max(np.abs(revised_confidence[prior_mask] - 0.30))) > 1e-3

    cleared_frequency, cleared_confidence, cleared_mask = remove_all_rules(frequency, confidence, mask)
    assert not cleared_mask.any()
    assert np.all(cleared_frequency == 0.0)
    assert np.all(cleared_confidence == 0.0)


def test_disabling_one_rule_removes_only_that_rule():
    frame = pd.DataFrame(
        {
            "genomic_mutation__BRAF": [1.0],
            "diagnoses.age_at_diagnosis": [40 * 365.25],
        }
    )
    features = ["genomic_mutation__BRAF", "diagnoses.age_at_diagnosis"]
    full = build_symbolic_truth_matrices(frame, features)
    disabled = build_symbolic_truth_matrices(frame, features, disabled_rule_ids={"braf_mutation"})
    assert full.rule_trigger_counts["braf_mutation"] == 1
    assert disabled.rule_trigger_counts["braf_mutation"] == 0
    assert int(disabled.symbolic_trigger_mask.sum()) == 0


def test_rule_registry_records_provenance_and_rejects_contradictions():
    assert "provenance" in RULE_GENERATION_PROCESS
    validate_rule_registry(SYMBOLIC_RULES)
    validate_rule_registry(WIDS_RULE_DEFINITIONS)
    assert all(rule["expert_review"] == "not_reviewed" for rule in SYMBOLIC_RULES.values())
    broken = {rule_id: dict(rule) for rule_id, rule in SYMBOLIC_RULES.items()}
    broken["age_ge_55_years"]["contradiction_group"] = broken["braf_mutation"]["contradiction_group"]
    broken["age_ge_55_years"]["target_key"] = broken["braf_mutation"]["target_key"]
    with pytest.raises(ValueError, match="Contradictory rules"):
        validate_rule_registry(broken)
    missing = {rule_id: dict(rule) for rule_id, rule in SYMBOLIC_RULES.items()}
    missing["braf_mutation"]["source"] = "  "
    with pytest.raises(ValueError, match="missing registry fields"):
        validate_rule_registry(missing)


def test_wids_disabled_rule_is_not_counted():
    triggers = np.array([[1, 1, 0, 0]], dtype=np.float64)
    features = ["d1_lactate_max", "d1_sysbp_min", "age", "d1_creatinine_max", "elective_surgery"]
    full = build_wids_symbolic_truth_matrices(triggers, features)
    disabled = build_wids_symbolic_truth_matrices(triggers, features, disabled_rule_ids={"rule_lactate"})
    assert full.patient_rule_counts[0] == 2
    assert disabled.patient_rule_counts[0] == 1
    assert disabled.rule_trigger_counts["rule_lactate"] == 0
    assert disabled.symbolic_trigger_mask[0, 0] == False


def test_uncertainty_estimators_compare_across_seeds():
    generator = np.random.default_rng(0)
    features = generator.normal(size=(48, 4)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.float32)
    dataset = TensorDataset(torch.as_tensor(features), torch.as_tensor(labels))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    member_probabilities = []
    mc_variance = None
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = TabularTransformerClassifier(input_dim=4, d_model=8, nhead=2, num_layers=1, dropout=0.4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _epoch in range(3):
            for batch_features, batch_labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_features).logits, batch_labels)
                loss.backward()
                optimizer.step()
        member_probabilities.append(model.predict_proba(features, torch.device("cpu"), batch_size=16))
        if seed == 0:
            summary = model.predict_with_mc_dropout(loader, torch.device("cpu"), mc_samples=4)
            mc_variance = summary.probabilities_var
    stacked = np.stack(member_probabilities, axis=0)
    assert not np.allclose(stacked[0], stacked[1])
    ensemble = deep_ensemble_statistics(stacked)
    entropy = mc_predictive_entropy(stacked)
    report = compare_uncertainty_estimators(
        {
            "mc_dropout_variance": mc_variance,
            "mc_predictive_entropy": entropy,
            "deep_ensemble_variance": ensemble["variance"],
        },
        errors=(stacked.mean(axis=0) >= 0.5).astype(int) != labels.astype(int),
        seeds=[0, 1, 2],
    )
    assert report["seed_count"] == 3
    assert ensemble["members"] == 3
    assert set(report["error_detection_auroc"]) == {
        "mc_dropout_variance",
        "mc_predictive_entropy",
        "deep_ensemble_variance",
    }
    assert np.isfinite(list(report["pairwise_spearman"].values())).all()


def test_shift_evaluation_does_not_retune_frozen_state():
    features = np.array([[0.2, -0.4], [1.5, 0.3], [-0.7, 0.8], [0.4, 0.1]], dtype=np.float64)
    labels = np.array([0, 1, 0, 1])
    frozen = {"scaler_mean": np.array([0.1, -0.2]), "rules": np.array([0.8, 0.2])}

    def predict(shifted: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-shifted.sum(axis=1)))

    frame = evaluate_frozen_shift(
        predict,
        features,
        labels,
        lambda: frozen,
        rates=(0.0, 0.5),
        seed=0,
    )
    assert frame["retuned"].eq(False).all()
    assert frame.loc[frame["shift_rate"] == 0.0, "brier"].iloc[0] == pytest.approx(
        frame["brier"].iloc[0]
    )
    unchanged = predict(features)
    zero_rate = predict(features)
    assert np.allclose(unchanged, zero_rate)

    def mutating_fingerprint() -> dict[str, np.ndarray]:
        frozen["scaler_mean"] = frozen["scaler_mean"] + 1.0
        return {"scaler_mean": frozen["scaler_mean"]}

    with pytest.raises(RuntimeError, match="frozen parameter"):
        evaluate_frozen_shift(predict, features, labels, mutating_fingerprint, rates=(0.0, 0.1), seed=0)
