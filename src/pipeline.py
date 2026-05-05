from __future__ import annotations

import copy
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from torch import nn

from .attention_hook import apply_confidence_gate, revise_attention_truths
from .data_loader import DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN, load_data_bundle
from .knowledge_base import SYMBOLIC_RULES, build_symbolic_truth_matrices
from .nars_interface import neural_to_nars
from .neural_encoder import TabularTransformerClassifier
from .wids_knowledge_base import WIDS_SYMBOLIC_RULES, build_wids_symbolic_truth_matrices
from .wids_loader import WIDS_ID_COLUMN, WIDS_TARGET_COLUMN, load_wids_data_bundle

GAMMA_ABLATION_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
TREE_BASELINE_LABEL = "random_forest"
TRANSFORMER_VARIANTS = ("baseline", "flat_confidence", "mc_confidence_only", "nars_gated")
DATASET_METADATA: dict[str, dict[str, Any]] = {
    "tcga": {
        "display_name": "TCGA-THCA",
        "positive_class": "Lymph Node Metastasis",
        "id_column": DEFAULT_ID_COLUMN,
        "target_column": DEFAULT_TARGET_COLUMN,
        "loader": load_data_bundle,
        "symbolic_rules": SYMBOLIC_RULES,
        "symbolic_builder": "tcga",
        "batch_size": None,
        "source_description": "TCGA clinical TSV tables and somatic mutation MAF file(s).",
    },
    "wids": {
        "display_name": "WiDS ICU 2020",
        "positive_class": "Hospital Mortality",
        "id_column": WIDS_ID_COLUMN,
        "target_column": WIDS_TARGET_COLUMN,
        "loader": load_wids_data_bundle,
        "symbolic_rules": WIDS_SYMBOLIC_RULES,
        "symbolic_builder": "wids",
        "batch_size": 512,
        "source_description": "WiDS ICU CSV file.",
    },
}


def _get_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return completed.stdout.strip() or None


def _get_environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "git_commit": _get_git_commit(),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "results"
    dataset: str = "tcga"
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mc_samples: int = 50
    gamma: float = 2.0
    seed: int = 0
    patience: int = 12
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    baseline_set: str = "minimal"
    ablation_set: str = "quick"
    export_case_traces: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_dataset_metadata(dataset: str) -> dict[str, Any]:
    try:
        return DATASET_METADATA[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def _ensure_output_directories(base_dir: str | Path, dataset: str) -> dict[str, Path]:
    root = Path(base_dir) / dataset
    charts_dir = root / "charts"
    metrics_dir = root / "metrics"
    traces_dir = root / "traces"
    charts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "charts": charts_dir, "metrics": metrics_dir, "traces": traces_dir}


def _get_encoded_split(bundle, split_name: str):
    encoded = getattr(bundle, f"encoded_{split_name}", None)
    if encoded is not None:
        return encoded
    return bundle.preprocessor.transform(getattr(bundle, f"{split_name}_frame"))


def _evaluate_loss(
    model: TabularTransformerClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    probabilities = []
    labels = []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            output = model(features)
            loss = criterion(output.logits, target)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
            labels.append(target.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probabilities), np.concatenate(labels)


def _train_model(model: TabularTransformerClassifier, bundle, config: PipelineConfig, device: torch.device) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for features, target in bundle.train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss = criterion(output.logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_probabilities, val_labels = _evaluate_loss(model, bundle.val_loader, device, criterion)
        try:
            val_auc = float(roc_auc_score(val_labels, val_probabilities))
        except ValueError:
            val_auc = float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
        )
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _build_reliability_frame(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"probability": np.asarray(probabilities, dtype=np.float64), "label": np.asarray(y_true, dtype=np.float64)}
    ).sort_values("probability", kind="mergesort").reset_index(drop=True)
    frame["bin"] = pd.qcut(frame.index, q=n_bins, labels=False, duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("probability", "mean"),
            fraction_positives=("label", "mean"),
            count=("label", "size"),
        )
        .reset_index(drop=True)
    )


def _compute_ece(reliability_frame: pd.DataFrame) -> float:
    total = float(reliability_frame["count"].sum())
    weighted_error = (
        (reliability_frame["count"] / total)
        * np.abs(reliability_frame["fraction_positives"] - reliability_frame["mean_predicted_probability"])
    ).sum()
    return float(weighted_error)


def _compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped_probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    predictions = (clipped_probabilities >= 0.5).astype(int)
    reliability = _build_reliability_frame(y_true, clipped_probabilities, n_bins=10)
    return {
        "auc": float(roc_auc_score(y_true, clipped_probabilities)),
        "brier": float(brier_score_loss(y_true, clipped_probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "ece": _compute_ece(reliability),
    }


def _bootstrap_metric_intervals(
    y_true: np.ndarray,
    probability_map: dict[str, np.ndarray],
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    bootstrap_indices: list[np.ndarray] = []
    max_attempts = iterations * 100
    attempts = 0

    while len(bootstrap_indices) < iterations:
        if attempts >= max_attempts:
            raise RuntimeError("Unable to generate enough valid bootstrap samples for metric estimation.")
        sampled_indices = rng.integers(0, n_samples, size=n_samples)
        sampled_labels = y_true[sampled_indices]
        attempts += 1
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_indices.append(sampled_indices)

    intervals: dict[str, Any] = {}
    for variant, probabilities in probability_map.items():
        clipped_probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        auc_samples = np.asarray(
            [roc_auc_score(y_true[index_set], clipped_probabilities[index_set]) for index_set in bootstrap_indices],
            dtype=np.float64,
        )
        brier_samples = np.asarray(
            [brier_score_loss(y_true[index_set], clipped_probabilities[index_set]) for index_set in bootstrap_indices],
            dtype=np.float64,
        )
        ece_samples = np.asarray(
            [
                _compute_ece(_build_reliability_frame(y_true[index_set], clipped_probabilities[index_set], n_bins=10))
                for index_set in bootstrap_indices
            ],
            dtype=np.float64,
        )
        intervals[variant] = {
            "iterations": iterations,
            "auc_samples_mean": float(np.mean(auc_samples)),
            "auc_ci_95_lower": float(np.percentile(auc_samples, 2.5)),
            "auc_ci_95_upper": float(np.percentile(auc_samples, 97.5)),
            "brier_samples_mean": float(np.mean(brier_samples)),
            "brier_ci_95_lower": float(np.percentile(brier_samples, 2.5)),
            "brier_ci_95_upper": float(np.percentile(brier_samples, 97.5)),
            "ece_samples_mean": float(np.mean(ece_samples)),
            "ece_ci_95_lower": float(np.percentile(ece_samples, 2.5)),
            "ece_ci_95_upper": float(np.percentile(ece_samples, 97.5)),
        }

    def _interval_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
        return not (
            left["auc_ci_95_upper"] < right["auc_ci_95_lower"]
            or right["auc_ci_95_upper"] < left["auc_ci_95_lower"]
        )

    def _metric_samples(metric_name: str, variant: str) -> np.ndarray:
        probabilities = np.clip(probability_map[variant], 1e-6, 1.0 - 1e-6)
        if metric_name == "brier":
            return np.asarray(
                [brier_score_loss(y_true[index_set], probabilities[index_set]) for index_set in bootstrap_indices],
                dtype=np.float64,
            )
        if metric_name == "ece":
            return np.asarray(
                [
                    _compute_ece(_build_reliability_frame(y_true[index_set], probabilities[index_set], n_bins=10))
                    for index_set in bootstrap_indices
                ],
                dtype=np.float64,
            )
        raise ValueError(f"Unsupported bootstrap metric: {metric_name}")

    def _add_delta_intervals(comparisons: dict[str, Any], left_variant: str, right_variant: str) -> None:
        comparison_key = f"{left_variant}_vs_{right_variant}"
        for metric_name in ("brier", "ece"):
            left_samples = _metric_samples(metric_name, left_variant)
            right_samples = _metric_samples(metric_name, right_variant)
            delta_samples = left_samples - right_samples
            comparisons[f"{comparison_key}_{metric_name}_delta_left_minus_right"] = float(np.mean(delta_samples))
            comparisons[f"{comparison_key}_{metric_name}_delta_ci_95_lower"] = float(np.percentile(delta_samples, 2.5))
            comparisons[f"{comparison_key}_{metric_name}_delta_ci_95_upper"] = float(np.percentile(delta_samples, 97.5))
            comparisons[f"{comparison_key}_{metric_name}_interpretation"] = (
                f"The 95% paired bootstrap interval for the {metric_name} difference excludes zero."
                if (
                    comparisons[f"{comparison_key}_{metric_name}_delta_ci_95_lower"] > 0.0
                    or comparisons[f"{comparison_key}_{metric_name}_delta_ci_95_upper"] < 0.0
                )
                else f"The 95% paired bootstrap interval for the {metric_name} difference includes zero."
            )

    comparisons: dict[str, Any] = {}
    if "baseline" in intervals and "nars_gated" in intervals:
        overlap = _interval_overlap(intervals["baseline"], intervals["nars_gated"])
        comparisons["baseline_vs_nars_gated_ci_overlap"] = overlap
        comparisons["baseline_vs_nars_gated_interpretation"] = (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        )
        _add_delta_intervals(comparisons, "baseline", "nars_gated")
    if "flat_confidence" in intervals and "nars_gated" in intervals:
        overlap = _interval_overlap(intervals["flat_confidence"], intervals["nars_gated"])
        comparisons["flat_confidence_vs_nars_gated_ci_overlap"] = overlap
        comparisons["flat_confidence_vs_nars_gated_interpretation"] = (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        )
        _add_delta_intervals(comparisons, "flat_confidence", "nars_gated")
    if "random_forest" in intervals and "nars_gated" in intervals:
        overlap = _interval_overlap(intervals["random_forest"], intervals["nars_gated"])
        comparisons["random_forest_vs_nars_gated_ci_overlap"] = overlap
        comparisons["random_forest_vs_nars_gated_interpretation"] = (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        )
        _add_delta_intervals(comparisons, "random_forest", "nars_gated")
    if comparisons:
        intervals["comparison"] = comparisons
    return intervals


def _fit_best_baseline(
    label: str,
    candidates: list[tuple[Any, dict[str, Any]]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> dict[str, Any]:
    best_model = None
    best_config: dict[str, Any] | None = None
    best_val_brier = float("inf")
    best_val_auc = float("-inf")

    for model, candidate_config in candidates:
        model.fit(x_train, y_train)
        val_probabilities = model.predict_proba(x_val)[:, 1]
        val_metrics = _compute_metrics(y_val, val_probabilities)
        if (
            val_metrics["brier"] < best_val_brier - 1e-9
            or (
                abs(val_metrics["brier"] - best_val_brier) <= 1e-9
                and val_metrics["auc"] > best_val_auc
            )
        ):
            best_model = model
            best_config = candidate_config
            best_val_brier = val_metrics["brier"]
            best_val_auc = val_metrics["auc"]

    if best_model is None or best_config is None:
        raise RuntimeError(f"Failed to train baseline: {label}")

    return {
        "label": label,
        "best_config": best_config,
        "val_brier": best_val_brier,
        "val_auc": best_val_auc,
        "test_probabilities": best_model.predict_proba(x_test)[:, 1],
    }


def _train_classical_baselines(bundle, config: PipelineConfig) -> dict[str, dict[str, Any]]:
    encoded_train = _get_encoded_split(bundle, "train")
    encoded_val = _get_encoded_split(bundle, "val")
    encoded_test = _get_encoded_split(bundle, "test")

    x_train = encoded_train.features
    y_train = encoded_train.target.astype(int)
    x_val = encoded_val.features
    y_val = encoded_val.target.astype(int)
    x_test = encoded_test.features
    y_test = encoded_test.target.astype(int)

    random_forest_configs = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3},
    ]
    baseline_candidates: dict[str, list[tuple[Any, dict[str, Any]]]] = {
        TREE_BASELINE_LABEL: [
            (
                RandomForestClassifier(
                    **candidate,
                    random_state=config.seed,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                candidate,
            )
            for candidate in random_forest_configs
        ]
    }

    if config.baseline_set == "standard":
        extra_trees_configs = [
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2},
        ]
        baseline_candidates["extra_trees"] = [
            (
                ExtraTreesClassifier(
                    **candidate,
                    random_state=config.seed,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                candidate,
            )
            for candidate in extra_trees_configs
        ]
        hist_gradient_configs = [
            {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 31},
            {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 15},
        ]
        baseline_candidates["hist_gradient_boosting"] = [
            (
                HistGradientBoostingClassifier(
                    **candidate,
                    random_state=config.seed,
                    l2_regularization=1e-4,
                ),
                candidate,
            )
            for candidate in hist_gradient_configs
        ]
        logistic_configs = [
            {"C": 0.5},
            {"C": 1.0},
        ]
        baseline_candidates["calibrated_logistic_regression"] = [
            (
                CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        **candidate,
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                    method="sigmoid",
                    cv=3,
                ),
                candidate,
            )
            for candidate in logistic_configs
        ]

    baselines: dict[str, dict[str, Any]] = {}
    for label, candidates in baseline_candidates.items():
        baselines[label] = _fit_best_baseline(
            label=label,
            candidates=candidates,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
        )
        baselines[label]["test_labels"] = y_test
    return baselines


def _train_tree_baseline(bundle, config: PipelineConfig) -> dict[str, Any]:
    return _train_classical_baselines(bundle, config)[TREE_BASELINE_LABEL]


def _save_roc_plot(
    y_true: np.ndarray,
    tree_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    flat_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    output_path: Path,
    positive_class_label: str,
) -> None:
    tree_fpr, tree_tpr, _ = roc_curve(y_true, tree_probabilities)
    baseline_fpr, baseline_tpr, _ = roc_curve(y_true, baseline_probabilities)
    flat_fpr, flat_tpr, _ = roc_curve(y_true, flat_probabilities)
    gated_fpr, gated_tpr, _ = roc_curve(y_true, gated_probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(tree_fpr, tree_tpr, label="Random Forest")
    plt.plot(baseline_fpr, baseline_tpr, label="Baseline")
    plt.plot(flat_fpr, flat_tpr, label="Flat-confidence")
    plt.plot(gated_fpr, gated_tpr, label="NARS-gated")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel(f"False Positive Rate ({positive_class_label})")
    plt.ylabel(f"True Positive Rate ({positive_class_label})")
    plt.title(f"ROC Curve: {positive_class_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_calibration_plot(
    y_true: np.ndarray,
    tree_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    flat_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    output_path: Path,
    positive_class_label: str,
) -> None:
    plt.figure(figsize=(7, 5))
    for label, probabilities in (
        ("Random Forest", tree_probabilities),
        ("Baseline", baseline_probabilities),
        ("Flat-confidence", flat_probabilities),
        ("NARS-gated", gated_probabilities),
    ):
        reliability = _build_reliability_frame(y_true, probabilities, n_bins=10)
        plt.plot(
            reliability["mean_predicted_probability"],
            reliability["fraction_positives"],
            marker="o",
            label=label,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel(f"Mean Predicted Probability ({positive_class_label})")
    plt.ylabel(f"Observed Frequency ({positive_class_label})")
    plt.title(f"Calibration Reliability Diagram: {positive_class_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_training_plot(history: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_gamma_ablation_plot(ablation_frame: pd.DataFrame, output_path: Path, positive_class_label: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(ablation_frame["gamma"], ablation_frame["baseline_auc"], marker="o", label="Baseline AUC")
    plt.plot(ablation_frame["gamma"], ablation_frame["nars_gated_auc"], marker="o", label="NARS-gated AUC")
    plt.xscale("log", base=2)
    plt.xticks(ablation_frame["gamma"], [str(gamma) for gamma in ablation_frame["gamma"]])
    plt.xlabel("Gamma")
    plt.ylabel("AUC")
    plt.title(f"Gamma Ablation: {positive_class_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _compute_net_benefit(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    predictions = probabilities >= threshold
    n_samples = len(y_true)
    true_positives = float(np.sum((predictions == 1) & (y_true == 1)))
    false_positives = float(np.sum((predictions == 1) & (y_true == 0)))
    odds = threshold / (1.0 - threshold)
    return (true_positives / n_samples) - (false_positives / n_samples) * odds


def _build_decision_curve_frame(
    y_true: np.ndarray,
    tree_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    flat_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
) -> pd.DataFrame:
    thresholds = np.arange(0.05, 1.0, 0.05, dtype=np.float64)
    prevalence = float(np.mean(y_true))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        treat_all_net_benefit = prevalence - (1.0 - prevalence) * (threshold / (1.0 - threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "random_forest_net_benefit": _compute_net_benefit(y_true, tree_probabilities, float(threshold)),
                "baseline_net_benefit": _compute_net_benefit(y_true, baseline_probabilities, float(threshold)),
                "flat_confidence_net_benefit": _compute_net_benefit(y_true, flat_probabilities, float(threshold)),
                "nars_gated_net_benefit": _compute_net_benefit(y_true, gated_probabilities, float(threshold)),
                "treat_all_net_benefit": float(treat_all_net_benefit),
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _save_decision_curve_plot(decision_curve_frame: pd.DataFrame, output_path: Path, positive_class_label: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["random_forest_net_benefit"], marker="o", label="Random forest")
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["baseline_net_benefit"], marker="o", label="Baseline transformer")
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["flat_confidence_net_benefit"], marker="o", label="Flat-confidence transformer")
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["nars_gated_net_benefit"], marker="o", label="NARS-gated transformer")
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["treat_all_net_benefit"], linestyle="--", label="Treat all")
    plt.plot(decision_curve_frame["threshold"], decision_curve_frame["treat_none_net_benefit"], linestyle=":", label="Treat none")
    plt.xlabel(f"Threshold Probability ({positive_class_label})")
    plt.ylabel("Net Benefit")
    plt.title(f"Decision Curve Analysis: {positive_class_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _build_trace_frame(
    bundle,
    probabilities_mean: np.ndarray,
    probabilities_var: np.ndarray,
    neural_frequency: np.ndarray,
    neural_confidence: np.ndarray,
    attention_mean: np.ndarray,
    attention_var: np.ndarray,
    neural_feature_confidence: np.ndarray,
    revised_feature_confidence: np.ndarray,
    gated_attention: np.ndarray,
    token_score_mean: np.ndarray,
    gated_probabilities: np.ndarray,
    symbolic_rule_counts: np.ndarray,
    symbolic_any_rule_triggered: np.ndarray,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    trace_frame = bundle.test_frame[[id_column, target_column]].reset_index(drop=True).copy()
    trace_frame = trace_frame.rename(columns={target_column: "target"})
    base_columns = pd.DataFrame(
        {
            "baseline_probability": probabilities_mean,
            "baseline_variance": probabilities_var,
            "neural_frequency": neural_frequency,
            "neural_confidence": neural_confidence,
            "gated_probability": gated_probabilities,
            "symbolic_rule_count": symbolic_rule_counts,
            "symbolic_any_rule_triggered": symbolic_any_rule_triggered.astype(int),
            "neural_attention_reliability": (attention_mean * neural_feature_confidence).sum(axis=1),
            "revised_attention_reliability": (gated_attention * revised_feature_confidence).sum(axis=1),
            "attention_reliability": (gated_attention * revised_feature_confidence).sum(axis=1),
        }
    )
    feature_columns: dict[str, np.ndarray] = {}
    for index, feature_name in enumerate(bundle.preprocessor.feature_names):
        safe_feature_name = feature_name.replace(" ", "_")
        feature_columns[f"attention_mean__{safe_feature_name}"] = attention_mean[:, index]
        feature_columns[f"attention_var__{safe_feature_name}"] = attention_var[:, index]
        feature_columns[f"attention_confidence__{safe_feature_name}"] = neural_feature_confidence[:, index]
        feature_columns[f"revised_attention_confidence__{safe_feature_name}"] = revised_feature_confidence[:, index]
        feature_columns[f"gated_attention__{safe_feature_name}"] = gated_attention[:, index]
        feature_columns[f"token_score__{safe_feature_name}"] = token_score_mean[:, index]
    return pd.concat([trace_frame, base_columns, pd.DataFrame(feature_columns)], axis=1)


def _build_gamma_ablation_frame(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    attention_mean: np.ndarray,
    feature_confidence: np.ndarray,
    cls_logit_mean: np.ndarray,
    token_score_mean: np.ndarray,
) -> pd.DataFrame:
    baseline_metrics = _compute_metrics(y_true, baseline_probabilities)
    rows: list[dict[str, float]] = []
    for gamma in GAMMA_ABLATION_VALUES:
        gated_attention = apply_confidence_gate(attention_mean, feature_confidence, gamma=gamma)
        gated_logits = cls_logit_mean + np.sum(gated_attention * token_score_mean, axis=1)
        gated_probabilities = _sigmoid(gated_logits)
        gated_metrics = _compute_metrics(y_true, gated_probabilities)
        rows.append(
            {
                "gamma": float(gamma),
                "baseline_auc": baseline_metrics["auc"],
                "baseline_brier": baseline_metrics["brier"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "nars_gated_auc": gated_metrics["auc"],
                "nars_gated_brier": gated_metrics["brier"],
                "nars_gated_accuracy": gated_metrics["accuracy"],
                "auc_delta_gated_minus_baseline": gated_metrics["auc"] - baseline_metrics["auc"],
                "brier_delta_gated_minus_baseline": gated_metrics["brier"] - baseline_metrics["brier"],
                "accuracy_delta_gated_minus_baseline": gated_metrics["accuracy"] - baseline_metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def _build_submission_ablation_frame(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    attention_mean: np.ndarray,
    neural_confidence: np.ndarray,
    revised_confidence: np.ndarray,
    symbolic_confidence: np.ndarray,
    symbolic_trigger_mask: np.ndarray,
    cls_logit_mean: np.ndarray,
    token_score_mean: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    def _add_row(ablation: str, gamma: float, confidence: np.ndarray) -> None:
        ablated_attention = apply_confidence_gate(attention_mean, confidence, gamma=gamma)
        ablated_logits = cls_logit_mean + np.sum(ablated_attention * token_score_mean, axis=1)
        ablated_probabilities = _sigmoid(ablated_logits)
        metrics = _compute_metrics(y_true, ablated_probabilities)
        rows.append(
            {
                "ablation": ablation,
                "gamma": float(gamma),
                "symbolic_confidence_scale": 1.0,
                "baseline_auc": _compute_metrics(y_true, baseline_probabilities)["auc"],
                "baseline_brier": _compute_metrics(y_true, baseline_probabilities)["brier"],
                "auc": metrics["auc"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
            }
        )

    for gamma in GAMMA_ABLATION_VALUES:
        _add_row("nars_gated_gamma", float(gamma), revised_confidence)
        _add_row("symbolic_disabled_mc_confidence_only", float(gamma), neural_confidence)

    for scale in (0.5, 0.75, 1.25, 1.5):
        scaled_symbolic_confidence = np.where(
            symbolic_trigger_mask,
            np.clip(symbolic_confidence * scale, 0.0, 1.0),
            neural_confidence,
        )
        revised_scaled_confidence = np.where(symbolic_trigger_mask, scaled_symbolic_confidence, neural_confidence)
        gated_attention = apply_confidence_gate(attention_mean, revised_scaled_confidence, gamma=2.0)
        probabilities = _sigmoid(cls_logit_mean + np.sum(gated_attention * token_score_mean, axis=1))
        metrics = _compute_metrics(y_true, probabilities)
        rows.append(
            {
                "ablation": "rule_truth_confidence_scale",
                "gamma": 2.0,
                "symbolic_confidence_scale": float(scale),
                "baseline_auc": _compute_metrics(y_true, baseline_probabilities)["auc"],
                "baseline_brier": _compute_metrics(y_true, baseline_probabilities)["brier"],
                "auc": metrics["auc"],
                "brier": metrics["brier"],
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
            }
        )

    return pd.DataFrame(rows)


def _missingness_fraction(frame: pd.DataFrame) -> np.ndarray:
    feature_frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_id")], errors="ignore")
    return feature_frame.isna().mean(axis=1).to_numpy(dtype=np.float64)


def _json_feature_trace(
    feature_names: list[str],
    patient_index: int,
    neural_frequency: np.ndarray,
    neural_confidence: np.ndarray,
    symbolic_frequency: np.ndarray,
    symbolic_confidence: np.ndarray,
    revised_frequency: np.ndarray,
    revised_confidence: np.ndarray,
    attention_mean: np.ndarray,
    gated_attention: np.ndarray,
    symbolic_trigger_mask: np.ndarray,
) -> str:
    triggered_indices = np.flatnonzero(symbolic_trigger_mask[patient_index])
    if triggered_indices.size == 0:
        changed = np.abs(gated_attention[patient_index] - attention_mean[patient_index])
        triggered_indices = np.argsort(changed)[-3:][::-1]
    records = []
    for feature_index in triggered_indices[:6]:
        records.append(
            {
                "feature": feature_names[int(feature_index)],
                "neural_f": float(neural_frequency[patient_index, feature_index]),
                "neural_c": float(neural_confidence[patient_index, feature_index]),
                "symbolic_f": float(symbolic_frequency[patient_index, feature_index]),
                "symbolic_c": float(symbolic_confidence[patient_index, feature_index]),
                "revised_f": float(revised_frequency[patient_index, feature_index]),
                "revised_c": float(revised_confidence[patient_index, feature_index]),
                "attention_before": float(attention_mean[patient_index, feature_index]),
                "attention_after": float(gated_attention[patient_index, feature_index]),
                "attention_delta": float(gated_attention[patient_index, feature_index] - attention_mean[patient_index, feature_index]),
            }
        )
    return json.dumps(records, sort_keys=True)


def _build_case_trace_frame(
    bundle,
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    neural_frequency: np.ndarray,
    neural_confidence: np.ndarray,
    attention_truths,
    symbolic_knowledge,
    attention_mean: np.ndarray,
    gated_attention: np.ndarray,
    id_column: str,
) -> pd.DataFrame:
    predictions = (gated_probabilities >= 0.5).astype(int)
    missingness = _missingness_fraction(bundle.test_frame)
    categories = {
        "true_positive": np.flatnonzero((predictions == 1) & (y_true == 1)),
        "true_negative": np.flatnonzero((predictions == 0) & (y_true == 0)),
        "false_positive": np.flatnonzero((predictions == 1) & (y_true == 0)),
        "false_negative": np.flatnonzero((predictions == 0) & (y_true == 1)),
        "high_missingness": np.argsort(missingness)[-5:][::-1],
        "symbolic_active": np.flatnonzero(symbolic_knowledge.patient_any_rule_triggered),
    }

    selected: list[tuple[str, int]] = []
    seen: set[int] = set()
    for category, indices in categories.items():
        for index in indices[:5]:
            patient_index = int(index)
            if patient_index in seen:
                continue
            selected.append((category, patient_index))
            seen.add(patient_index)
            break
    if len(selected) < 20:
        uncertainty_order = np.argsort(np.abs(gated_probabilities - 0.5))
        for index in uncertainty_order:
            patient_index = int(index)
            if patient_index in seen:
                continue
            selected.append(("near_threshold", patient_index))
            seen.add(patient_index)
            if len(selected) >= 20:
                break

    rows: list[dict[str, Any]] = []
    feature_names = bundle.preprocessor.feature_names
    for category, patient_index in selected[:20]:
        row = {
            "case_category": category,
            "case_index": patient_index,
            "case_id": bundle.test_frame.iloc[patient_index][id_column],
            "target": int(y_true[patient_index]),
            "baseline_probability": float(baseline_probabilities[patient_index]),
            "nars_gated_probability": float(gated_probabilities[patient_index]),
            "prediction": int(predictions[patient_index]),
            "missingness_fraction": float(missingness[patient_index]),
            "symbolic_rule_count": int(symbolic_knowledge.patient_rule_counts[patient_index]),
            "symbolic_any_rule_triggered": int(symbolic_knowledge.patient_any_rule_triggered[patient_index]),
            "neural_prediction_f": float(neural_frequency[patient_index]),
            "neural_prediction_c": float(neural_confidence[patient_index]),
            "feature_trace_json": _json_feature_trace(
                feature_names=feature_names,
                patient_index=patient_index,
                neural_frequency=attention_truths.neural_frequency,
                neural_confidence=attention_truths.neural_confidence,
                symbolic_frequency=symbolic_knowledge.symbolic_frequency,
                symbolic_confidence=symbolic_knowledge.symbolic_confidence,
                revised_frequency=attention_truths.revised_frequency,
                revised_confidence=attention_truths.revised_confidence,
                attention_mean=attention_mean,
                gated_attention=gated_attention,
                symbolic_trigger_mask=symbolic_knowledge.symbolic_trigger_mask,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_symbolic_knowledge(bundle, config: PipelineConfig):
    dataset_metadata = _get_dataset_metadata(config.dataset)
    if dataset_metadata["symbolic_builder"] == "tcga":
        return build_symbolic_truth_matrices(bundle.test_frame, bundle.preprocessor.feature_names)
    encoded_test = _get_encoded_split(bundle, "test")
    print("WiDS ICU rule trigger counts on the test set:")
    for rule_name, count in zip(bundle.preprocessor.rule_names, encoded_test.rule_triggers.sum(axis=0).astype(int)):
        print(f"  {rule_name}: {int(count)}")
    return build_wids_symbolic_truth_matrices(
        encoded_test.rule_triggers,
        bundle.preprocessor.feature_names,
        rule_names=bundle.preprocessor.rule_names,
    )


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    dataset_metadata = _get_dataset_metadata(config.dataset)
    effective_batch_size = int(dataset_metadata["batch_size"] or config.batch_size)
    effective_config = PipelineConfig(**{**asdict(config), "batch_size": effective_batch_size})

    set_seed(effective_config.seed)
    output_dirs = _ensure_output_directories(effective_config.output_dir, effective_config.dataset)
    bundle = dataset_metadata["loader"](data_dir=effective_config.data_dir, batch_size=effective_config.batch_size, seed=effective_config.seed)
    bundle.preprocessor.save(output_dirs["traces"] / "preprocessing_metadata.json")
    (output_dirs["traces"] / "split_summary.json").write_text(json.dumps(bundle.split_summary, indent=2), encoding="utf-8")

    print(
        f"Loaded {dataset_metadata['display_name']} dataset: "
        f"{bundle.split_summary['labeled_case_rows']} labeled cases, "
        f"{bundle.split_summary['input_dim']} total input features."
    )
    print(
        "Split sizes: "
        f"train={bundle.split_summary['train_rows']}, "
        f"val={bundle.split_summary['val_rows']}, "
        f"test={bundle.split_summary['test_rows']}."
    )
    if effective_config.batch_size != config.batch_size:
        print(f"Using dataset-specific batch size override: {effective_config.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformerClassifier(
        input_dim=bundle.preprocessor.input_dim,
        d_model=effective_config.d_model,
        nhead=effective_config.num_heads,
        num_layers=effective_config.num_layers,
        dropout=effective_config.dropout,
    ).to(device)

    history = _train_model(model=model, bundle=bundle, config=effective_config, device=device)
    history.to_csv(output_dirs["metrics"] / "training_history.csv", index=False)
    _save_training_plot(history, output_dirs["charts"] / "training_history.png", dataset_metadata["display_name"])
    classical_baselines = _train_classical_baselines(bundle=bundle, config=effective_config)
    tree_baseline = classical_baselines[TREE_BASELINE_LABEL]

    summary = model.predict_with_mc_dropout(loader=bundle.test_loader, device=device, mc_samples=effective_config.mc_samples)
    symbolic_knowledge = _resolve_symbolic_knowledge(bundle, effective_config)
    neural_frequency, neural_confidence = neural_to_nars(summary.probabilities_mean, summary.probabilities_var)
    attention_truths = revise_attention_truths(
        summary.attention_mean,
        summary.attention_var,
        symbolic_frequency=symbolic_knowledge.symbolic_frequency,
        symbolic_confidence=symbolic_knowledge.symbolic_confidence,
        symbolic_trigger_mask=symbolic_knowledge.symbolic_trigger_mask,
    )
    gated_attention = apply_confidence_gate(summary.attention_mean, attention_truths.revised_confidence, gamma=effective_config.gamma)
    mc_attention = apply_confidence_gate(summary.attention_mean, attention_truths.neural_confidence, gamma=effective_config.gamma)
    flat_attention = apply_confidence_gate(
        summary.attention_mean,
        np.full_like(attention_truths.revised_confidence, 0.5, dtype=np.float64),
        gamma=effective_config.gamma,
    )
    flat_logits = summary.cls_logit_mean + np.sum(flat_attention * summary.token_score_mean, axis=1)
    flat_probabilities = _sigmoid(flat_logits)
    mc_logits = summary.cls_logit_mean + np.sum(mc_attention * summary.token_score_mean, axis=1)
    mc_probabilities = _sigmoid(mc_logits)
    gated_logits = summary.cls_logit_mean + np.sum(gated_attention * summary.token_score_mean, axis=1)
    gated_probabilities = _sigmoid(gated_logits)

    y_true = summary.labels.astype(int)
    tree_probabilities = tree_baseline["test_probabilities"]
    probability_map = {
        **{label: baseline["test_probabilities"] for label, baseline in classical_baselines.items()},
        "baseline": summary.probabilities_mean,
        "flat_confidence": flat_probabilities,
        "mc_confidence_only": mc_probabilities,
        "nars_gated": gated_probabilities,
    }
    reliability_frame = pd.concat(
        [
            _build_reliability_frame(y_true, probabilities, n_bins=10).assign(variant=variant)
            for variant, probabilities in probability_map.items()
        ],
        ignore_index=True,
    )
    metric_bootstrap = _bootstrap_metric_intervals(
        y_true=y_true,
        probability_map=probability_map,
        iterations=1000,
        seed=effective_config.seed,
    )
    metric_rows: list[dict[str, Any]] = []
    for variant, probabilities in probability_map.items():
        metric_rows.append(
            {
                "variant": variant,
                **_compute_metrics(y_true, probabilities),
                "auc_ci_95_lower": metric_bootstrap[variant]["auc_ci_95_lower"],
                "auc_ci_95_upper": metric_bootstrap[variant]["auc_ci_95_upper"],
                "brier_ci_95_lower": metric_bootstrap[variant]["brier_ci_95_lower"],
                "brier_ci_95_upper": metric_bootstrap[variant]["brier_ci_95_upper"],
                "ece_ci_95_lower": metric_bootstrap[variant]["ece_ci_95_lower"],
                "ece_ci_95_upper": metric_bootstrap[variant]["ece_ci_95_upper"],
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dirs["metrics"] / "metrics.csv", index=False)
    reliability_frame.to_csv(output_dirs["metrics"] / "calibration_reliability.csv", index=False)

    trace_frame = _build_trace_frame(
        bundle=bundle,
        probabilities_mean=summary.probabilities_mean,
        probabilities_var=summary.probabilities_var,
        neural_frequency=np.asarray(neural_frequency),
        neural_confidence=np.asarray(neural_confidence),
        attention_mean=summary.attention_mean,
        attention_var=summary.attention_var,
        neural_feature_confidence=attention_truths.neural_confidence,
        revised_feature_confidence=attention_truths.revised_confidence,
        gated_attention=gated_attention,
        token_score_mean=summary.token_score_mean,
        gated_probabilities=gated_probabilities,
        symbolic_rule_counts=symbolic_knowledge.patient_rule_counts,
        symbolic_any_rule_triggered=symbolic_knowledge.patient_any_rule_triggered,
        id_column=dataset_metadata["id_column"],
        target_column=dataset_metadata["target_column"],
    )
    trace_frame.to_csv(output_dirs["traces"] / "test_predictions.csv", index=False)
    case_trace_frame = pd.DataFrame()
    if effective_config.export_case_traces:
        case_trace_frame = _build_case_trace_frame(
            bundle=bundle,
            y_true=y_true,
            baseline_probabilities=summary.probabilities_mean,
            gated_probabilities=gated_probabilities,
            neural_frequency=np.asarray(neural_frequency),
            neural_confidence=np.asarray(neural_confidence),
            attention_truths=attention_truths,
            symbolic_knowledge=symbolic_knowledge,
            attention_mean=summary.attention_mean,
            gated_attention=gated_attention,
            id_column=dataset_metadata["id_column"],
        )
        case_trace_frame.to_csv(output_dirs["traces"] / "case_traces.csv", index=False)

    gamma_ablation_frame = _build_gamma_ablation_frame(
        y_true=y_true,
        baseline_probabilities=summary.probabilities_mean,
        attention_mean=summary.attention_mean,
        feature_confidence=attention_truths.revised_confidence,
        cls_logit_mean=summary.cls_logit_mean,
        token_score_mean=summary.token_score_mean,
    )
    gamma_ablation_frame.to_csv(output_dirs["metrics"] / "gamma_ablation.csv", index=False)
    _save_gamma_ablation_plot(gamma_ablation_frame, output_dirs["charts"] / "gamma_ablation_auc.png", dataset_metadata["positive_class"])
    submission_ablation_frame = pd.DataFrame()
    if effective_config.ablation_set == "submission":
        submission_ablation_frame = _build_submission_ablation_frame(
            y_true=y_true,
            baseline_probabilities=summary.probabilities_mean,
            attention_mean=summary.attention_mean,
            neural_confidence=attention_truths.neural_confidence,
            revised_confidence=attention_truths.revised_confidence,
            symbolic_confidence=symbolic_knowledge.symbolic_confidence,
            symbolic_trigger_mask=symbolic_knowledge.symbolic_trigger_mask,
            cls_logit_mean=summary.cls_logit_mean,
            token_score_mean=summary.token_score_mean,
        )
        submission_ablation_frame.to_csv(output_dirs["metrics"] / "submission_ablation.csv", index=False)

    decision_curve_frame = _build_decision_curve_frame(
        y_true=y_true,
        tree_probabilities=tree_probabilities,
        baseline_probabilities=summary.probabilities_mean,
        flat_probabilities=flat_probabilities,
        gated_probabilities=gated_probabilities,
    )
    decision_curve_frame.to_csv(output_dirs["metrics"] / "decision_curve.csv", index=False)
    _save_decision_curve_plot(decision_curve_frame, output_dirs["charts"] / "decision_curve.png", dataset_metadata["positive_class"])

    _save_roc_plot(
        y_true,
        tree_probabilities,
        summary.probabilities_mean,
        flat_probabilities,
        gated_probabilities,
        output_dirs["charts"] / "roc_curve.png",
        dataset_metadata["positive_class"],
    )
    _save_calibration_plot(
        y_true,
        tree_probabilities,
        summary.probabilities_mean,
        flat_probabilities,
        gated_probabilities,
        output_dirs["charts"] / "calibration_curve.png",
        dataset_metadata["positive_class"],
    )

    summary_frame = {
        "config": asdict(effective_config),
        "environment": _get_environment_info(),
        "task": {
            "dataset": dataset_metadata["display_name"],
            "dataset_key": effective_config.dataset,
            "positive_class": dataset_metadata["positive_class"],
            "id_column": dataset_metadata["id_column"],
            "target_column": dataset_metadata["target_column"],
            "source_description": dataset_metadata["source_description"],
        },
        "split_summary": bundle.split_summary,
        "metrics": metrics_frame.to_dict(orient="records"),
        "symbolic_rules": {
            "definitions": dataset_metadata["symbolic_rules"],
            "interface_notes": {
                "neural_truth_mapping": "variance-derived heuristic initializer for NARS-style confidence",
                "symbolic_rule_grounding": "explicit NAL deduction from direct observation before revision",
                "revision_assumption": "neural and symbolic truths are treated as distinct evidential sources",
            },
            "total_trigger_count": symbolic_knowledge.total_trigger_count,
            "mapped_feature_trigger_count": symbolic_knowledge.mapped_feature_trigger_count,
            "cases_with_any_trigger": int(symbolic_knowledge.patient_any_rule_triggered.sum()),
            "per_rule_trigger_counts": symbolic_knowledge.rule_trigger_counts,
            "per_rule_mapped_trigger_counts": symbolic_knowledge.mapped_rule_trigger_counts,
        },
        "classical_baseline": {
            "variant": TREE_BASELINE_LABEL,
            "selection_objective": "validation_brier",
            "best_config": tree_baseline["best_config"],
            "validation_auc": tree_baseline["val_auc"],
            "validation_brier": tree_baseline["val_brier"],
        },
        "classical_baselines": {
            label: {
                "variant": label,
                "selection_objective": "validation_brier",
                "best_config": baseline["best_config"],
                "validation_auc": baseline["val_auc"],
                "validation_brier": baseline["val_brier"],
            }
            for label, baseline in classical_baselines.items()
        },
        "auc_bootstrap": metric_bootstrap,
        "metric_bootstrap": metric_bootstrap,
        "gamma_ablation": gamma_ablation_frame.to_dict(orient="records"),
        "submission_ablation": submission_ablation_frame.to_dict(orient="records"),
        "case_traces": case_trace_frame.to_dict(orient="records"),
        "decision_curve": decision_curve_frame.to_dict(orient="records"),
        "calibration_reliability": reliability_frame.to_dict(orient="records"),
        "artifacts": {
            "root": str(output_dirs["root"]),
            "metrics_csv": str(output_dirs["metrics"] / "metrics.csv"),
            "training_history_csv": str(output_dirs["metrics"] / "training_history.csv"),
            "gamma_ablation_csv": str(output_dirs["metrics"] / "gamma_ablation.csv"),
            "submission_ablation_csv": str(output_dirs["metrics"] / "submission_ablation.csv") if effective_config.ablation_set == "submission" else None,
            "decision_curve_csv": str(output_dirs["metrics"] / "decision_curve.csv"),
            "calibration_reliability_csv": str(output_dirs["metrics"] / "calibration_reliability.csv"),
            "trace_csv": str(output_dirs["traces"] / "test_predictions.csv"),
            "case_traces_csv": str(output_dirs["traces"] / "case_traces.csv") if effective_config.export_case_traces else None,
            "roc_curve": str(output_dirs["charts"] / "roc_curve.png"),
            "calibration_curve": str(output_dirs["charts"] / "calibration_curve.png"),
            "training_history_plot": str(output_dirs["charts"] / "training_history.png"),
            "gamma_ablation_plot": str(output_dirs["charts"] / "gamma_ablation_auc.png"),
            "decision_curve_plot": str(output_dirs["charts"] / "decision_curve.png"),
            "preprocessing_metadata_json": str(output_dirs["traces"] / "preprocessing_metadata.json"),
            "split_summary_json": str(output_dirs["traces"] / "split_summary.json"),
        },
    }
    (output_dirs["metrics"] / "run_summary.json").write_text(
        json.dumps(summary_frame, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary_frame
