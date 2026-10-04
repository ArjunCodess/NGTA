from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest
import hashlib
import json
from torch.utils.data import DataLoader, TensorDataset

from src.attention_hook import revise_attention_truths
from src.knowledge_base import SYMBOLIC_RULES, build_symbolic_truth_matrices
from src.matched_inference import score_cached_passes
from src.neural_encoder import TabularTransformerClassifier
from src.trace_replay import export_replay_bundle, replay_bundle, reference_predicate
from src.trace_replay import native_logit_check


def test_native_logit_check_bounds_float32_cancellation_but_rejects_corruption():
    attention = np.ones((1,1,3),dtype=np.float32)
    scores = np.array([[[1000,.0001,-1000]]],dtype=np.float32)
    cls = np.zeros((1,1),dtype=np.float32)
    native = (attention*scores).sum(-1)+cls
    assert native_logit_check(attention,scores,cls,native)[0]
    assert not native_logit_check(attention,scores,cls,native+1)[0]


def _export(tmp_path):
    features = ["diagnoses.age_at_diagnosis", "genomic_mutation__BRAF"]
    raw = pd.DataFrame({"case_submitter_id": ["a", "b", "c", "d"], features[0]: [70*365.25, 40*365.25, np.nan, 55*365.25], features[1]: [0, 1, np.nan, 0]})
    bundle = SimpleNamespace(test_frame=raw, preprocessor=SimpleNamespace(feature_names=features, id_column="case_submitter_id"))
    dataset = TensorDataset(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    torch.manual_seed(3)
    model = TabularTransformerClassifier(2, 8, 2, 1, .3)
    summary = model.predict_with_mc_dropout(DataLoader(dataset, batch_size=2), torch.device("cpu"), 3)
    knowledge = build_symbolic_truth_matrices(raw, features)
    truths = revise_attention_truths(summary.attention_mean, summary.attention_var, knowledge.symbolic_frequency, knowledge.symbolic_confidence, knowledge.symbolic_trigger_mask)
    cached = summary.attention_passes, summary.token_score_passes, summary.cls_logit_passes
    variants = {name: score_cached_passes(*cached, confidence, 2)[0] for name, confidence in (
        ("baseline", None), ("flat_confidence", np.full((4, 2), .5)),
        ("mc_confidence_only", truths.neural_confidence), ("nars_gated", truths.revised_confidence))}
    attention = score_cached_passes(*cached, truths.revised_confidence, 2)[1]
    return export_replay_bundle(tmp_path, bundle=bundle, summary=summary, knowledge=knowledge, truths=truths,
                                rules=SYMBOLIC_RULES, dataset="tcga", gamma=2, probabilities=variants, attention_after=attention)


def test_complete_bundle_replays_and_detects_missing_event(tmp_path):
    report = _export(tmp_path)
    assert report["passed"]
    assert report["events"] == 3
    events = pd.read_csv(tmp_path / "intervention_events.csv")
    assert events["rule_off_probability"].notna().all()
    events.iloc[1:].to_csv(tmp_path / "intervention_events.csv", index=False)
    replay = replay_bundle(tmp_path)
    assert not replay["passed"]
    assert not replay["event_completeness"]
    assert not replay["artifact_integrity"]


def test_replay_detects_corrupted_revision_cache(tmp_path):
    assert _export(tmp_path)["passed"]
    path = tmp_path / "inference_cache.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values["revised_confidence"][0, 0] = .1
    np.savez_compressed(path, **values)
    assert replay_bundle(tmp_path)["max_residuals"]["revised_confidence"] > .1


def test_independent_hypotension_predicate_rejects_negative_missing_and_boundary():
    observed = pd.Series([-1, None, "Unknown", 90, 91, 89])
    np.testing.assert_array_equal(reference_predicate("rule_hypotension", observed), [0, 0, 0, 1, 0, 1])


@pytest.mark.parametrize("field, value", [
    ("neural_frequency", np.nan), ("attention_before", .9),
    ("attention_after", .9), ("nars_probability", .99),
    ("imputed_value", -1), ("expert_review", "reviewed"),
])
def test_replay_rejects_invalid_event_fields_even_with_matching_hash(tmp_path, field, value):
    assert _export(tmp_path)["passed"]
    path = tmp_path / "intervention_events.csv"
    events = pd.read_csv(path)
    events.loc[0, field] = value
    events.to_csv(path, index=False)
    hashes_path = tmp_path / "replay_hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes))
    report = replay_bundle(tmp_path)
    assert report["artifact_integrity"]
    assert not report["passed"]
