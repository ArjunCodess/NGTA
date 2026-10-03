"""Evaluate an explicitly harmonized independent cohort with frozen artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .acceptance import external_compatibility
from .attention_hook import revise_attention_truths
from .evaluation import binary_metrics
from .knowledge_base import build_symbolic_truth_matrices
from .matched_inference import score_cached_passes
from .neural_encoder import TabularTransformerClassifier
from .trace_replay import export_replay_bundle
from .wids_knowledge_base import build_wids_symbolic_truth_matrices


def harmonize_cohort(frame, policy, processor, development_ids, development_patient_ids=None):
    required = ("cohort", "source_version", "prediction_landmark", "outcome_definition",
                "independence_evidence", "id_column", "patient_id_column", "target_column", "cluster_column", "features")
    if any(not policy.get(key) for key in required):
        raise ValueError("External mapping requires source, landmark, outcome, independence and cluster documentation")
    id_source, target_source = policy["id_column"], policy["target_column"]
    ids = frame[id_source].astype(str)
    if frame[id_source].isna().any() or ids.duplicated().any():
        raise ValueError("External case IDs must be observed and unique")
    if set(ids) & set(map(str, development_ids)):
        raise ValueError("External cases overlap the development cohort")
    patients = frame[policy["patient_id_column"]]
    if patients.isna().any():
        raise ValueError("External patient identity must be observed")
    if development_patient_ids is not None and set(patients.astype(str)) & set(map(str, development_patient_ids)):
        raise ValueError("External patients overlap the development cohort")
    if frame[policy["cluster_column"]].isna().any():
        raise ValueError("External institutional clusters must be observed")
    y = pd.to_numeric(frame[target_source], errors="coerce")
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("External outcomes must be complete binary labels with both classes")
    required_features = processor.numeric_columns + processor.binary_columns + processor.categorical_columns
    if set(policy["features"]) != set(required_features):
        raise ValueError("Provide an explicit mapping for every frozen input, including unavailable inputs")
    mapped = pd.DataFrame({processor.id_column: ids, processor.target_column: y.astype(int)})
    for name in required_features:
        rule = policy["features"][name]
        if not all(rule.get(field) for field in ("source_unit", "target_unit", "availability", "measurement_window", "evidence")):
            raise ValueError(f"Missing unit/window/availability evidence for {name}")
        source = rule.get("column")
        if source is None:
            if rule["availability"] != "unavailable":
                raise ValueError(f"Absent input {name} must be explicitly unavailable")
            mapped[name] = np.nan
            continue
        if source in {target_source, id_source, policy["cluster_column"], policy["patient_id_column"]}:
            raise ValueError("Outcome and identity fields cannot be mapped to predictors")
        values = frame[source]
        if name in processor.numeric_columns:
            scale, offset = float(rule.get("scale", 1)), float(rule.get("offset", 0))
            if not np.isfinite([scale, offset]).all() or scale <= 0:
                raise ValueError("Unit conversions require a positive finite scale and finite offset")
            values = pd.to_numeric(values, errors="coerce") * scale + offset
        elif "values" in rule:
            values = values.map(rule["values"])
        mapped[name] = values
    return mapped, frame[policy["cluster_column"]].to_numpy()


def evaluate_external(checkpoint_dir, cohort_csv, mapping_json, output_dir, mc_samples=50, seed=0):
    checkpoint_dir = Path(checkpoint_dir)
    saved = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    training_spec = saved.get("training_spec")
    if training_spec is None:
        raise ValueError("External evaluation requires a checkpoint with immutable training provenance")
    processor = joblib.load(checkpoint_dir / "preprocessor.joblib")
    manifest = pd.read_csv(checkpoint_dir / "traces" / "split_ids.csv")
    policy = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
    # Validate the frozen study definitions before reading any external outcomes.
    expected_outcome = "hospital_mortality" if config["dataset"] == "wids" else "lymph_node_metastasis"
    if policy.get("outcome_definition") != expected_outcome or policy.get("development_dataset") != config["dataset"]:
        raise ValueError("External outcome and frozen development dataset do not match")
    frame = pd.read_csv(cohort_csv)
    patient_ids = manifest[processor.id_column]
    if config["dataset"] == "wids":
        group_manifest = pd.read_csv(checkpoint_dir / "traces" / "development_groups.csv")
        patient_ids = group_manifest.patient_id
    mapped, groups = harmonize_cohort(frame, policy, processor, manifest[processor.id_column], patient_ids)
    fingerprint = joblib.hash(processor)
    encoded = processor.transform_components(mapped) if config["dataset"] == "wids" else processor.transform(mapped)
    knowledge = (build_wids_symbolic_truth_matrices(encoded.rule_triggers, processor.feature_names)
                 if config["dataset"] == "wids" else build_symbolic_truth_matrices(mapped, processor.feature_names))
    model = TabularTransformerClassifier(saved["input_dim"], config["d_model"], config["num_heads"],
                                        config["num_layers"], config["dropout"])
    model.load_state_dict(saved["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loader = DataLoader(TensorDataset(torch.tensor(encoded.features), torch.tensor(encoded.target)),
                        batch_size=config["batch_size"])
    summary = model.predict_with_mc_dropout(loader, device, mc_samples)
    truths = revise_attention_truths(summary.attention_mean, summary.attention_var,
        knowledge.symbolic_frequency, knowledge.symbolic_confidence, knowledge.symbolic_trigger_mask)
    cached = summary.attention_passes, summary.token_score_passes, summary.cls_logit_passes
    mc, _ = score_cached_passes(*cached, truths.neural_confidence, config["gamma"])
    nars, attention = score_cached_passes(*cached, truths.revised_confidence, config["gamma"])
    if fingerprint != joblib.hash(processor):
        raise RuntimeError("External evaluation changed frozen preprocessing")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    probabilities = {"baseline": summary.probabilities_mean, "mc_confidence_only": mc, "nars_gated": nars}
    report = external_compatibility(encoded.target, nars, mc, groups=groups, seed=seed)
    pd.DataFrame([dict(variant=name, **binary_metrics(encoded.target, p)) for name, p in probabilities.items()]).to_csv(root / "metrics.csv", index=False)
    bundle = SimpleNamespace(preprocessor=processor, test_frame=mapped, encoded_test=encoded)
    export_replay_bundle(root / "traces", bundle=bundle, summary=summary, knowledge=knowledge, truths=truths,
        rules=training_spec["rules"], dataset=config["dataset"], gamma=config["gamma"],
        probabilities=probabilities, attention_after=attention)
    report["mapping"] = policy
    report["checkpoint_sha256"] = hashlib.sha256((checkpoint_dir / "model.pt").read_bytes()).hexdigest()
    report["cohort_sha256"] = hashlib.sha256(Path(cohort_csv).read_bytes()).hexdigest()
    report["mc_samples"] = mc_samples
    report["independence_verification"] = "case and patient identifier overlap checked; independent source identity requires the documented cohort provenance"
    (root / "external_compatibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
