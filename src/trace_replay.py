"""Complete event export and independent replay from persisted numeric artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .auditability import reference_revision
from .matched_inference import score_cached_passes


def reference_predicate(rule_id: str, values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    predicates = {
        "rule_lactate": lambda: numeric.ge(4),
        "rule_hypotension": lambda: numeric.le(90) & numeric.ge(0),
        "rule_age": lambda: numeric.ge(75),
        "rule_creatinine": lambda: numeric.ge(2),
        "braf_mutation": lambda: numeric.eq(1),
        "age_ge_55_years": lambda: numeric.ge(55 * 365.25),
        "pathologic_t_t3_t4": lambda: values.fillna("").astype(str).str.startswith(("T3", "T4")),
        "extrathyroid_extension_present": lambda: values.isin(["Minimal (T3)", "Moderate/Advanced (T4a)", "Very Advanced (T4b)"]),
    }
    if rule_id not in predicates:
        raise ValueError(f"No independent predicate implementation for {rule_id}")
    return predicates[rule_id]().fillna(False).to_numpy(dtype=bool)


def _feature_position(rule: dict, raw_value, features: list[str]) -> int | None:
    source = str(rule["source_column"])
    feature = f"{source}_{raw_value}" if rule.get("target_mode") == "categorical_active" else source
    return features.index(feature) if feature in features else None


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EVENT_COLUMNS = ["case_index", "case_id", "rule_id", "rule_version", "source", "expert_review",
                 "source_column", "raw_value", "imputed_value", "was_imputed", "mapped", "feature_index",
                 "neural_frequency", "neural_confidence", "symbolic_frequency", "symbolic_confidence",
                 "revised_frequency", "revised_confidence", "attention_before", "attention_after",
                 "rule_off_probability", "nars_probability", "symbolic_probability_delta"]


def export_replay_bundle(directory, *, bundle, summary, knowledge, truths, rules, dataset,
                         gamma, probabilities, attention_after) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    features = bundle.preprocessor.feature_names
    raw = bundle.test_frame.reset_index(drop=True)
    raw.to_csv(root / "raw_test.csv", index=False)
    spec = {"schema_version": 2, "dataset": dataset, "gamma": gamma, "feature_names": features,
            "id_column": bundle.preprocessor.id_column, "rules": rules,
            "rule_input_policy": "observed_only", "aggregation": "mean_per_pass_probability",
            "evidential_independence": "not established; neural and symbolic paths share clinical inputs"}
    (root / "replay_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cached = summary.attention_passes, summary.token_score_passes, summary.cls_logit_passes
    np.savez_compressed(root / "inference_cache.npz",
                        attention_passes=summary.attention_passes, token_score_passes=summary.token_score_passes,
                        cls_logit_passes=summary.cls_logit_passes, probability_passes=summary.probability_passes,
                        logit_passes=summary.logit_passes,
                        labels=summary.labels, neural_frequency=truths.neural_frequency,
                        neural_confidence=truths.neural_confidence, revised_frequency=truths.revised_frequency,
                        revised_confidence=truths.revised_confidence, symbolic_frequency=knowledge.symbolic_frequency,
                        symbolic_confidence=knowledge.symbolic_confidence, trigger_mask=knowledge.symbolic_trigger_mask,
                        attention_after=attention_after, **{f"probability__{k}": v for k, v in probabilities.items()})
    rows = []
    encoded = getattr(bundle, "encoded_test", None)
    for rule_id, rule in rules.items():
        source = rule["source_column"]
        trigger = knowledge.rule_case_masks.get(rule_id, np.zeros(len(raw), dtype=bool))
        for i in np.flatnonzero(trigger):
            value = raw.at[i, source]
            j = _feature_position(rule, value, features)
            row = {"case_index": int(i), "case_id": raw.at[i, bundle.preprocessor.id_column],
                   "rule_id": rule_id, "rule_version": rule["provenance"], "source": rule["source"],
                   "expert_review": rule["expert_review"], "source_column": source,
                   "raw_value": value, "imputed_value": value, "was_imputed": False,
                   "mapped": j is not None, "feature_index": j,
                   "symbolic_frequency": rule["truth_value"]["frequency"],
                   "symbolic_confidence": rule["truth_value"]["confidence"]}
            if encoded is not None and source in bundle.preprocessor.numeric_columns:
                row["imputed_value"] = encoded.numeric_imputed[i, bundle.preprocessor.numeric_columns.index(source)]
            if j is not None:
                confidence_off = truths.revised_confidence[i:i+1].copy()
                confidence_off[0, j] = truths.neural_confidence[i, j]
                case_cache = tuple(array[:, i:i+1] for array in cached)
                rule_off, _ = score_cached_passes(*case_cache, confidence_off, gamma)
                row.update(neural_frequency=truths.neural_frequency[i, j], neural_confidence=truths.neural_confidence[i, j],
                           revised_frequency=truths.revised_frequency[i, j], revised_confidence=truths.revised_confidence[i, j],
                           attention_before=summary.attention_mean[i, j], attention_after=attention_after[i, j],
                           rule_off_probability=rule_off[0], nars_probability=probabilities["nars_gated"][i],
                           symbolic_probability_delta=probabilities["nars_gated"][i] - rule_off[0])
            rows.append(row)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if len(events) != knowledge.total_trigger_count:
        raise ValueError("Event export must include every rule trigger, including unmapped triggers")
    events.to_csv(root / "intervention_events.csv", index=False)
    artifacts = ("raw_test.csv", "replay_spec.json", "inference_cache.npz", "intervention_events.csv")
    (root / "replay_hashes.json").write_text(json.dumps({name: _hash(root / name) for name in artifacts}, indent=2), encoding="utf-8")
    report = replay_bundle(root)
    (root / "replay_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"Persisted replay failed: {report}")
    return report


def replay_bundle(directory: str | Path, tolerance: float = 1e-7) -> dict:
    root = Path(directory)
    hashes = json.loads((root / "replay_hashes.json").read_text())
    integrity = all(_hash(root / name) == digest for name, digest in hashes.items())
    spec = json.loads((root / "replay_spec.json").read_text())
    raw = pd.read_csv(root / "raw_test.csv", dtype={spec["id_column"]: str}, keep_default_na=True)
    events = pd.read_csv(root / "intervention_events.csv", dtype={"case_id": str})
    features = spec["feature_names"]
    with np.load(root / "inference_cache.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    mask = np.zeros((len(raw), len(features)), dtype=bool)
    symbolic_f = np.zeros_like(mask, dtype=float)
    symbolic_c = np.zeros_like(mask, dtype=float)
    expected_events = set()
    for rule_id, rule in spec["rules"].items():
        source = rule["source_column"]
        if source not in raw:
            continue
        triggered = reference_predicate(rule_id, raw[source])
        for i in np.flatnonzero(triggered):
            expected_events.add((int(i), str(raw.at[i, spec["id_column"]]), rule_id))
            j = _feature_position(rule, raw.at[i, source], features)
            if j is not None:
                mask[i, j] = True
                symbolic_f[i, j] = rule["truth_value"]["frequency"]
                symbolic_c[i, j] = rule["truth_value"]["confidence"]
    actual_events = set(zip(events.case_index, events.case_id, events.rule_id))
    complete = expected_events == actual_events and len(events) == len(expected_events)
    attention = data["attention_passes"]
    mean = attention.mean(axis=0).astype(float)
    variance = attention.var(axis=0).astype(float)
    freq = np.clip(mean, 0, 1)
    confidence = (freq * (1-freq) + 1e-5) / (freq * (1-freq) + 1e-5 + np.maximum(variance, 0))
    revised_f, revised_c = reference_revision(freq, confidence, symbolic_f, symbolic_c)
    revised_f = np.where(mask, revised_f, freq)
    revised_c = np.where(mask, revised_c, confidence)
    residuals = {}
    original_logits = data["logit_passes"].astype(float)
    residuals["original_probability_passes"] = float(np.max(np.abs(
        np.exp(-np.logaddexp(0, -original_logits)) - data["probability_passes"])))
    components = data["cls_logit_passes"].astype(float) + (attention.astype(float) * data["token_score_passes"]).sum(-1)
    native_logit_consistency = np.allclose(components, original_logits, atol=1e-7, rtol=1e-6)
    for name, expected in (("trigger_mask", mask), ("symbolic_frequency", symbolic_f), ("symbolic_confidence", symbolic_c),
                           ("neural_frequency", freq), ("neural_confidence", confidence),
                           ("revised_frequency", revised_f), ("revised_confidence", revised_c)):
        residuals[name] = float(np.max(np.abs(expected.astype(float) - data[name].astype(float))))
    # Implement probability reconstruction locally, independently of production gating.
    baseline = None
    for variant, gate_c in (("baseline", None), ("flat_confidence", np.full_like(confidence, .5)),
                            ("mc_confidence_only", confidence), ("nars_gated", revised_c)):
        weights = attention.astype(float)
        if gate_c is not None:
            weights = weights * np.clip(gate_c, 0, 1)[None] ** spec["gamma"]
            total = weights.sum(-1, keepdims=True)
            weights = np.divide(weights, total, out=attention.astype(float).copy(), where=total > 0)
        logits = data["cls_logit_passes"] + (weights * data["token_score_passes"]).sum(-1)
        predicted = np.exp(-np.logaddexp(0, -logits)).mean(0)
        residuals[variant] = float(np.max(np.abs(predicted - data["probability__" + variant])))
        if variant == "baseline":
            baseline = predicted
        if variant == "flat_confidence":
            residuals["uniform_identity"] = float(np.max(np.abs(predicted - baseline)))
        if variant == "nars_gated":
            residuals["attention_after"] = float(np.max(np.abs(weights.mean(0) - data["attention_after"])))
    # Verify persisted event fields as well as the numerical cache.
    for event in events.itertuples():
        rule = spec["rules"].get(event.rule_id, {})
        if event.rule_version != rule.get("provenance") or event.source != rule.get("source") or event.was_imputed:
            complete = False
        j = _feature_position(rule, raw.at[event.case_index, rule["source_column"]], features) if rule else None
        if bool(event.mapped) != (j is not None):
            complete = False
        if j is not None:
            i = int(event.case_index)
            if int(event.feature_index) != j or event.source_column != rule["source_column"] or event.expert_review != rule["expert_review"]:
                complete = False
            value = raw.at[i, rule["source_column"]]
            try:
                raw_matches = np.isclose(float(event.raw_value), float(value), atol=1e-10)
            except (ValueError, TypeError):
                raw_matches = str(event.raw_value) == str(value)
            complete = complete and bool(raw_matches)
            for name, array in (("neural_frequency", freq), ("neural_confidence", confidence),
                                ("revised_frequency", revised_f), ("revised_confidence", revised_c),
                                ("symbolic_frequency", symbolic_f), ("symbolic_confidence", symbolic_c)):
                residuals["event_" + name] = max(residuals.get("event_" + name, 0), abs(getattr(event, name) - array[i, j]))
            off_confidence = revised_c[i:i+1].copy()
            off_confidence[0, j] = confidence[i, j]
            off_weights = attention[:, i:i+1].astype(float) * off_confidence[None] ** spec["gamma"]
            denominator = off_weights.sum(-1, keepdims=True)
            off_weights = np.divide(off_weights, denominator, out=attention[:, i:i+1].astype(float).copy(), where=denominator > 0)
            off_logits = data["cls_logit_passes"][:, i:i+1] + (off_weights * data["token_score_passes"][:, i:i+1]).sum(-1)
            off_probability = np.exp(-np.logaddexp(0, -off_logits)).mean()
            residuals["event_rule_off_probability"] = max(residuals.get("event_rule_off_probability", 0), abs(event.rule_off_probability - off_probability))
            residuals["event_symbolic_delta"] = max(residuals.get("event_symbolic_delta", 0), abs(event.symbolic_probability_delta - (data["probability__nars_gated"][i] - off_probability)))
    finite = all(np.isfinite(value) for value in residuals.values())
    passed = integrity and complete and finite and native_logit_consistency and all(value <= tolerance for value in residuals.values())
    return {"schema_version": 2, "passed": bool(passed), "artifact_integrity": bool(integrity),
            "event_completeness": bool(complete), "events": len(events), "expected_events": len(expected_events),
            "native_float32_logit_consistency": bool(native_logit_consistency),
            "tolerance": tolerance, "max_residuals": residuals}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Independently replay an NGTA v2 trace bundle")
    parser.add_argument("directory")
    args = parser.parse_args()
    result = replay_bundle(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
