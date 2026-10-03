"""Raw-value masking with frozen preprocessing and matched gate comparisons."""
from __future__ import annotations

import hashlib
import json
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import uuid
from torch.utils.data import DataLoader, TensorDataset

from .attention_hook import revise_attention_truths
from .acceptance import robustness_acceptance
from .evaluation import binary_metrics
from .knowledge_base import build_symbolic_truth_matrices
from .matched_inference import score_cached_passes
from .wids_knowledge_base import build_wids_symbolic_truth_matrices


RATES = (0.0, .1, .3, .5, .7)


def mask_observed_values(frame, columns, rate, rng, scenario="random", labels=None):
    if not 0 <= rate <= 1:
        raise ValueError("Mask rate must lie in [0,1]")
    if scenario not in {"random", "feature_dependent", "outcome_dependent_simulation"}:
        raise ValueError("Unknown prespecified masking scenario")
    observed = frame[columns].notna().to_numpy()
    weights = np.ones(len(columns)) if scenario == "random" else np.linspace(.5, 1.5, len(columns))
    if scenario == "outcome_dependent_simulation":
        y = np.asarray(labels)
        if y.shape != (len(frame),) or not np.isin(y, [0,1]).all():
            raise ValueError("Outcome-dependent simulation requires aligned binary outcomes")
        weights = np.where(y[:, None] == 1, 1.5, .5)
    mask = (rng.random(observed.shape) < np.clip(rate * weights, 0, 1)) & observed
    shifted = frame.copy()
    shifted[columns] = shifted[columns].mask(mask)
    return shifted, mask


def selective_risk(labels, probabilities, uncertainty):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(uncertainty, kind="stable")
    errors = ((probabilities >= .5) != labels).astype(float)[order]
    # Retain complete tie groups; arbitrary input order must not improve AURC.
    sorted_uncertainty = np.asarray(uncertainty)[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_uncertainty) != 0) + 1, len(errors)]
    risk = np.cumsum(errors)[endpoints-1] / endpoints
    coverage = endpoints / len(errors)
    area = float(np.sum(risk * np.diff(np.r_[0, coverage])))
    return pd.DataFrame({"coverage": coverage, "error_risk": risk}), area


def _frozen_digest(model, preprocessor, rules):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(joblib.hash(preprocessor).encode())
    digest.update(json.dumps(rules, sort_keys=True).encode())
    return digest.hexdigest()


def evaluate_raw_missingness(bundle, model, device, config, rules, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    preprocessor = bundle.preprocessor
    columns = preprocessor.numeric_columns + preprocessor.binary_columns + preprocessor.categorical_columns
    labels = bundle.test_frame[preprocessor.target_column].to_numpy(dtype=int)
    before = _frozen_digest(model, preprocessor, rules)
    rows, curves, predictions, masks, acceptance = [], [], [], {}, {}
    groups = bundle.test_frame["hospital_id"].to_numpy() if "hospital_id" in bundle.test_frame else None
    transform_cache = Path(config.cache_dir) / "masked" if config.cache_dir else None
    if transform_cache:
        transform_cache.mkdir(parents=True, exist_ok=True)
    processor_hash = joblib.hash(preprocessor)
    transform_source_hash = hashlib.sha256(Path(inspect.getfile(type(preprocessor))).read_bytes()).hexdigest()
    scenarios = ("random", "feature_dependent", "outcome_dependent_simulation")
    for scenario in scenarios:
        paired_predictions = {"nars_gated": [], "mc_confidence_only": []}
        # Nested masks use the same draws, making higher-rate observations a subset.
        for rate in RATES:
            raw, mask = mask_observed_values(bundle.test_frame, columns, rate, np.random.default_rng(config.split_seed), scenario, labels)
            masks[f"{scenario}_{rate:.1f}"] = mask
            cache_path = (transform_cache / (joblib.hash((processor_hash, transform_source_hash, raw, scenario, rate)) + ".joblib")) if transform_cache else None
            if cache_path and cache_path.exists():
                encoded = joblib.load(cache_path)
            else:
                encoded = preprocessor.transform_components(raw) if config.dataset == "wids" else preprocessor.transform(raw)
                if cache_path:
                    temporary = cache_path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    joblib.dump(encoded, temporary, compress=3)
                    temporary.replace(cache_path)
            if config.dataset == "wids":
                knowledge = build_wids_symbolic_truth_matrices(encoded.rule_triggers, preprocessor.feature_names)
            else:
                knowledge = build_symbolic_truth_matrices(raw, preprocessor.feature_names)
            loader = DataLoader(TensorDataset(torch.tensor(encoded.features), torch.tensor(encoded.target)), batch_size=config.batch_size)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            summary = model.predict_with_mc_dropout(loader, device, config.mc_samples)
            truths = revise_attention_truths(summary.attention_mean, summary.attention_var,
                                             knowledge.symbolic_frequency, knowledge.symbolic_confidence, knowledge.symbolic_trigger_mask)
            cached = summary.attention_passes, summary.token_score_passes, summary.cls_logit_passes
            variants = {"baseline": None, "mc_confidence_only": truths.neural_confidence, "nars_gated": truths.revised_confidence}
            if config.dataset == "wids":
                imputed = build_wids_symbolic_truth_matrices(encoded.imputed_rule_triggers, preprocessor.feature_names)
                imputed_truths = revise_attention_truths(summary.attention_mean, summary.attention_var,
                                                       imputed.symbolic_frequency, imputed.symbolic_confidence, imputed.symbolic_trigger_mask)
                variants["nars_imputed_rules"] = imputed_truths.revised_confidence
            for name, confidence in variants.items():
                probabilities, _ = score_cached_passes(*cached, confidence, config.gamma)
                if name in paired_predictions:
                    paired_predictions[name].append(probabilities)
                curve, area = selective_risk(labels, probabilities, summary.probabilities_var)
                errors = (probabilities >= .5) != labels
                from sklearn.metrics import roc_auc_score
                error_auc = float(roc_auc_score(errors, summary.probabilities_var)) if np.unique(errors).size == 2 else float("nan")
                rows.append({"scenario": scenario, "mask_rate": rate, "variant": name,
                             "masked_observed_cells": int(mask.sum()), "retuned": False,
                             "error_detection_auroc": error_auc, "selective_risk_area": area,
                             **binary_metrics(labels, probabilities)})
                curves.append(curve.assign(scenario=scenario, mask_rate=rate, variant=name))
                predictions.append(pd.DataFrame({"case_id": raw[preprocessor.id_column].to_numpy(), "target": labels,
                                                  "probability": probabilities, "uncertainty": summary.probabilities_var})
                                   .assign(scenario=scenario, mask_rate=rate, variant=name))
        acceptance[scenario] = robustness_acceptance(labels, paired_predictions["nars_gated"],
            paired_predictions["mc_confidence_only"], RATES, groups=groups, seed=config.seed)
    after = _frozen_digest(model, preprocessor, rules)
    if before != after:
        raise RuntimeError("Missingness evaluation changed frozen model, preprocessing, or rules")
    metrics = pd.DataFrame(rows)
    zero = metrics.loc[metrics.mask_rate.eq(0), ["scenario", "variant", "brier"]].rename(columns={"brier": "unmasked_brier"})
    metrics = metrics.merge(zero, on=["scenario", "variant"])
    metrics["brier_degradation"] = metrics.brier - metrics.unmasked_brier
    metrics.to_csv(root / "raw_missingness.csv", index=False)
    pd.concat(curves).to_csv(root / "selective_risk.csv", index=False)
    pd.concat(predictions).to_csv(root / "missingness_predictions.csv", index=False)
    np.savez_compressed(root / "raw_masks.npz", **masks)
    (root / "robustness_acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    (root / "missingness_spec.json").write_text(json.dumps({"schema_version": 2, "mask_columns": columns,
        "rates": RATES, "seed": config.split_seed, "scenarios": scenarios,
        "outcome_dependent_policy": "simulation only: positives weight 1.5, negatives 0.5; never used for model selection",
        "frozen_digest": before, "rule_input_policy": "observed_only",
        "feature_dependent_weights": np.linspace(.5, 1.5, len(columns)).tolist()}, indent=2), encoding="utf-8")
    return metrics
