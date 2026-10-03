"""Replay completed seed bundles, compare their ensemble, and score frozen masks."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import joblib
import sklearn
import numpy as np
import pandas as pd
from scipy.stats import t
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bundle_cache import load_cached_bundle
from src.evaluation import binary_metrics, paired_bootstrap_indices
from src.neural_encoder import TabularTransformerClassifier
from src.pipeline import PipelineConfig, DATASET_METADATA
from src.robustness import evaluate_raw_missingness, selective_risk
from src.trace_replay import replay_bundle
from src.uncertainty import deep_ensemble_statistics, mc_predictive_entropy, compare_uncertainty_estimators


def analyze_study(root, seeds, mask_seeds=(), cache_dir=".cache/ngta", iterations=1000, dataset="wids"):
    root = Path(root)
    if mask_seeds and dataset!="wids":
        raise ValueError("The masking protocol is defined for WiDS")
    selected = [root / f"seed_{seed}" / dataset for seed in seeds]
    id_column="encounter_id" if dataset=="wids" else "case_submitter_id"
    if not all((path / "metrics" / "run_summary.json").exists() for path in selected):
        raise ValueError("Every selected seed must have completed evaluation")
    specs = [json.loads((path / "evaluation_spec.json").read_text()) for path in selected]
    if any(any(spec[key] != specs[0][key] for key in ("sources", "split_ids_sha256", "rules")) for spec in specs[1:]):
        raise ValueError("Study seeds have different data, splits or rules")
    comparable = lambda spec: {"feature_mode": "all", **{key:value for key,value in spec["config"].items() if key != "seed"}}
    if any(comparable(spec) != comparable(specs[0]) for spec in specs[1:]):
        raise ValueError("Study seeds have different training/evaluation settings")
    rows, deterministic, mc_variance, repeats, labels, case_ids = [], [], [], [], None, None
    for seed, path in zip(seeds, selected):
        if not replay_bundle(path / "traces")["passed"]:
            raise ValueError(f"Independent seed replay failed: {seed}")
        print(f"Replayed seed {seed}", flush=True)
        raw = pd.read_csv(path / "traces" / "raw_test.csv")
        with np.load(path / "traces" / "inference_cache.npz") as cache:
            if labels is None:
                labels = cache["labels"].copy()
                case_ids = raw[id_column].to_numpy()
                groups = raw.hospital_id.to_numpy() if dataset=="wids" else None
            elif not np.array_equal(labels, cache["labels"]) or not np.array_equal(case_ids, raw[id_column]):
                raise ValueError("Study seed predictions are not case aligned")
            deterministic.append(cache["probability__deterministic"].copy())
            mc_variance.append(cache["probability_passes"].var(axis=0))
            draws = [(cache["probability__nars_gated"]-labels)**2 - (cache["probability__mc_confidence_only"]-labels)**2]
        for repeat_path in sorted((path / "traces").glob("mc_repeat_*.npz")):
            if repeat_path.name == "mc_repeat_0.npz":
                continue  # The original cache already supplies this identical draw.
            with np.load(repeat_path) as repeat:
                draws.append((repeat["nars_gated"]-labels)**2 - (repeat["mc_confidence_only"]-labels)**2)
        repeats.append(np.array(draws))
        metrics = pd.read_csv(path / "metrics" / "metrics.csv")
        rows.append(metrics.assign(seed=seed))
    root.joinpath("analysis").mkdir(exist_ok=True)
    destination = root / "analysis"
    per_seed = pd.concat(rows, ignore_index=True)
    per_seed.to_csv(destination / "per_seed_metrics.csv", index=False)
    metric_names = ["auc", "pr_auc", "brier", "log_loss", "ece", "accuracy"]
    means = per_seed.groupby("variant")[metric_names].agg(["mean", "std"])
    means.columns = ["_".join(column) for column in means.columns]
    if len(seeds) > 1:
        for name in metric_names:
            half = t.ppf(.975, len(seeds)-1) * means[name+"_std"] / np.sqrt(len(seeds))
            means[name+"_seed_mean_lower_95"] = means[name+"_mean"]-half
            means[name+"_seed_mean_upper_95"] = means[name+"_mean"]+half
    means.to_csv(destination / "seed_variability.csv")
    ensemble = deep_ensemble_statistics(deterministic)
    estimators = {"deep_ensemble_variance": ensemble["variance"], "mc_dropout_variance_seed_mean": np.mean(mc_variance, axis=0),
                  "ensemble_predictive_entropy": mc_predictive_entropy(ensemble["mean"])}
    errors = (ensemble["mean"] >= .5) != labels
    uncertainty = compare_uncertainty_estimators(estimators, errors, seeds)
    uncertainty["error_reference"] = "same ensemble prediction errors for every estimator"
    uncertainty["ensemble_metrics"] = binary_metrics(labels, ensemble["mean"])
    curves = []
    for name, scores in estimators.items():
        curve, area = selective_risk(labels, ensemble["mean"], scores)
        curves.append(curve.assign(estimator=name))
        uncertainty.setdefault("selective_risk_area", {})[name] = area
    pd.concat(curves).to_csv(destination / "ensemble_selective_risk.csv", index=False)
    (destination / "ensemble_uncertainty.json").write_text(json.dumps(uncertainty, indent=2))
    np.savez_compressed(destination / "ensemble_predictions.npz", case_ids=case_ids, labels=labels,
                        deterministic_members=deterministic, ensemble_mean=ensemble["mean"], ensemble_variance=ensemble["variance"])
    rng = np.random.default_rng(0)
    indices = paired_bootstrap_indices(labels, iterations, rng, groups)
    bootstrap = []
    for index in indices:
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        delta = []
        for position in sampled_seeds:
            draws = repeats[position]
            sampled_draws = rng.integers(0, len(draws), len(draws))
            delta.append(draws[sampled_draws][:, index].mean())
        bootstrap.append(np.mean(delta))
    observed = np.mean([draws.mean() for draws in repeats])
    hierarchy = dict(brier_nars_minus_mc=float(observed), lower_95=float(np.percentile(bootstrap, 2.5)),
        upper_95=float(np.percentile(bootstrap, 97.5)), seeds=list(seeds), mc_repeats=[len(draws) for draws in repeats],
        bootstrap_iterations=iterations, sampling_units=["hospital" if groups is not None else "case", "training_seed", "dropout_repeat"],
        scope="one fixed development split; excludes variability across different source cohorts and split selections",
        minimum_symbolic_gain=.0001, brier_margin_met=bool(-observed >= .0001))
    (destination / "hierarchical_symbolic_comparison.json").write_text(json.dumps(hierarchy, indent=2))
    for seed in mask_seeds:
        path = root / f"seed_{seed}" / "wids"
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        config = PipelineConfig(**{k:v for k,v in saved["config"].items() if k in {f.name for f in fields(PipelineConfig)}})
        config.cache_dir = cache_dir
        bundle = load_cached_bundle(DATASET_METADATA["wids"]["loader"], dataset="wids", data_dir=config.data_dir,
            batch_size=config.batch_size, seed=config.split_seed, cache_dir=cache_dir,
            split_mode=config.split_mode, include_apache=config.include_apache, imputation=config.imputation)
        model = TabularTransformerClassifier(saved["input_dim"], config.d_model, config.num_heads, config.num_layers, config.dropout)
        model.load_state_dict(saved["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        output = path / "metrics" / "missingness"
        print(f"Scoring frozen masks for seed {seed}", flush=True)
        with sklearn.config_context(working_memory=64):
            evaluate_raw_missingness(bundle, model, device, config, specs[0]["rules"], output)
    return hierarchy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset",choices=("wids","tcga"),default="wids")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--mask-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--cache-dir", default=".cache/ngta")
    parser.add_argument("--iterations", type=int, default=1000)
    print(json.dumps(analyze_study(**vars(parser.parse_args())), indent=2))
