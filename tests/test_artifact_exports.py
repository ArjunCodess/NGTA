import json

import pandas as pd
import pytest

from main import _write_submission_outputs
from src.paper_figures import generate_paper_figures


def test_mismatched_evaluation_lock_preserves_existing_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.pipeline as pipeline

    frame = lambda value: pd.DataFrame({"case_submitter_id": [value]})
    bundle = SimpleNamespace(preprocessor=SimpleNamespace(id_column="case_submitter_id"),
                             train_frame=frame("train"), val_frame=frame("val"), test_frame=frame("test"))
    monkeypatch.setitem(pipeline.DATASET_METADATA["tcga"], "loader", lambda **kwargs: bundle)
    monkeypatch.setattr(pipeline, "source_manifest", lambda *args: [])
    output = tmp_path / "output"
    traces = output / "tcga/traces"
    traces.mkdir(parents=True)
    sentinel = traces / "split_ids.csv"
    sentinel.write_text("existing audit")
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    with pytest.raises(ValueError, match="supplied lock"):
        pipeline.run_pipeline(pipeline.PipelineConfig(output_dir=str(output), evaluation_lock=str(lock)))
    assert sentinel.read_text() == "existing audit"
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*")) == [
        "tcga", "tcga/traces", "tcga/traces/split_ids.csv"]


def test_single_seed_does_not_claim_zero_variability_and_exports_apache_comparison(tmp_path):
    summary = {"task": {"dataset_key": "wids"}, "config": {"seed": 0},
               "metrics": [{"variant": "nars_gated", "auc": .7, "brier": .1, "accuracy": .8, "ece": .02}],
               "metric_bootstrap": {"comparison": {
                   "apache_raw_vs_nars_gated_brier_delta_left_minus_right": .01,
                   "apache_raw_vs_nars_gated_brier_bootstrap_mean_delta": .02,
                   "apache_raw_vs_nars_gated_brier_delta_family_95_lower": -.01,
                   "apache_raw_vs_nars_gated_brier_delta_family_95_upper": .03,
                   "multiplicity_family_size": 4}}}
    _write_submission_outputs([summary], tmp_path, write_paper_tables=False)
    metrics = pd.read_csv(tmp_path / "submission/multiseed_metrics.csv")
    assert pd.isna(metrics.loc[0, "brier_std"])
    comparison = pd.read_csv(tmp_path / "submission/paired_metric_deltas.csv").iloc[0]
    assert comparison.comparison == "apache_raw_vs_nars_gated"
    assert comparison.brier_delta_left_minus_right == .01
    assert comparison.brier_bootstrap_mean_delta == .02
    assert comparison.brier_delta_family_95_lower == -.01


def test_figures_reject_mixed_legacy_and_v2_bundles(tmp_path):
    for dataset, schema in (("tcga", 1), ("wids", 2)):
        root = tmp_path / dataset / "metrics"
        root.mkdir(parents=True)
        (root / "run_summary.json").write_text(json.dumps({"schema_version": schema, "persisted_replay": {"passed": True}}))
    with pytest.raises(ValueError, match="mix legacy and v2"):
        generate_paper_figures(tmp_path, figures_dir=tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
