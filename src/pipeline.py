from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from torch import nn

from .attention_hook import apply_confidence_gate, attention_to_nars
from .data_loader import DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN, load_data_bundle
from .nars_interface import neural_to_nars
from .neural_encoder import TabularTransformerClassifier

GAMMA_ABLATION_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
POSITIVE_CLASS_LABEL = "Lymph Node Metastasis"
TREE_BASELINE_LABEL = "random_forest"


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "."
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_output_directories(base_dir: str | Path) -> dict[str, Path]:
    root = Path(base_dir)
    charts_dir = root / "charts"
    metrics_dir = root / "results" / "metrics"
    traces_dir = root / "results" / "traces"
    charts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "charts": charts_dir,
        "metrics": metrics_dir,
        "traces": traces_dir,
    }


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


def _train_model(
    model: TabularTransformerClassifier,
    bundle,
    config: PipelineConfig,
    device: torch.device,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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

        val_loss, val_probabilities, val_labels = _evaluate_loss(
            model,
            bundle.val_loader,
            device,
            criterion,
        )
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


def _build_reliability_frame(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "probability": np.asarray(probabilities, dtype=np.float64),
            "label": np.asarray(y_true, dtype=np.float64),
        }
    ).sort_values("probability", kind="mergesort").reset_index(drop=True)
    frame["bin"] = pd.qcut(frame.index, q=n_bins, labels=False, duplicates="drop")
    reliability = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted_probability=("probability", "mean"),
            fraction_positives=("label", "mean"),
            count=("label", "size"),
        )
        .reset_index(drop=True)
    )
    return reliability


def _compute_ece(reliability_frame: pd.DataFrame) -> float:
    total = float(reliability_frame["count"].sum())
    weighted_error = (
        (reliability_frame["count"] / total)
        * np.abs(reliability_frame["fraction_positives"] - reliability_frame["mean_predicted_probability"])
    ).sum()
    return float(weighted_error)


def _bootstrap_auc_intervals(
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
            raise RuntimeError("Unable to generate enough valid bootstrap samples for AUC estimation.")
        sampled_indices = rng.integers(0, n_samples, size=n_samples)
        sampled_labels = y_true[sampled_indices]
        attempts += 1
        if np.unique(sampled_labels).size < 2:
            continue
        bootstrap_indices.append(sampled_indices)

    intervals: dict[str, Any] = {}
    for variant, probabilities in probability_map.items():
        auc_samples = np.asarray(
            [roc_auc_score(y_true[index_set], probabilities[index_set]) for index_set in bootstrap_indices],
            dtype=np.float64,
        )
        intervals[variant] = {
            "iterations": iterations,
            "auc_samples_mean": float(np.mean(auc_samples)),
            "auc_ci_95_lower": float(np.percentile(auc_samples, 2.5)),
            "auc_ci_95_upper": float(np.percentile(auc_samples, 97.5)),
        }

    def _interval_overlap(left: dict[str, float], right: dict[str, float]) -> bool:
        return not (
            left["auc_ci_95_upper"] < right["auc_ci_95_lower"]
            or right["auc_ci_95_upper"] < left["auc_ci_95_lower"]
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
    if "flat_confidence" in intervals and "nars_gated" in intervals:
        overlap = _interval_overlap(intervals["flat_confidence"], intervals["nars_gated"])
        comparisons["flat_confidence_vs_nars_gated_ci_overlap"] = overlap
        comparisons["flat_confidence_vs_nars_gated_interpretation"] = (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        )
    if "random_forest" in intervals and "nars_gated" in intervals:
        overlap = _interval_overlap(intervals["random_forest"], intervals["nars_gated"])
        comparisons["random_forest_vs_nars_gated_ci_overlap"] = overlap
        comparisons["random_forest_vs_nars_gated_interpretation"] = (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        )
    if comparisons:
        intervals["comparison"] = comparisons
    return intervals


def _train_tree_baseline(bundle, config: PipelineConfig) -> dict[str, Any]:
    encoded_train = bundle.preprocessor.transform(bundle.train_frame)
    encoded_val = bundle.preprocessor.transform(bundle.val_frame)
    encoded_test = bundle.preprocessor.transform(bundle.test_frame)

    x_train = encoded_train.features
    y_train = encoded_train.target.astype(int)
    x_val = encoded_val.features
    y_val = encoded_val.target.astype(int)
    x_test = encoded_test.features
    y_test = encoded_test.target.astype(int)

    candidate_configs = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3},
    ]
    best_model: RandomForestClassifier | None = None
    best_config: dict[str, Any] | None = None
    best_val_brier = float("inf")
    best_val_auc = float("-inf")

    for candidate in candidate_configs:
        model = RandomForestClassifier(
            **candidate,
            random_state=config.seed,
            n_jobs=-1,
            class_weight="balanced",
        )
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
            best_config = candidate
            best_val_brier = val_metrics["brier"]
            best_val_auc = val_metrics["auc"]

    if best_model is None or best_config is None:
        raise RuntimeError("Failed to train a Random Forest baseline.")

    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    return {
        "label": TREE_BASELINE_LABEL,
        "best_config": best_config,
        "val_brier": best_val_brier,
        "val_auc": best_val_auc,
        "test_labels": y_test,
        "test_probabilities": test_probabilities,
    }


def _save_roc_plot(
    y_true: np.ndarray,
    tree_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    flat_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    output_path: Path,
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
    plt.xlabel(f"False Positive Rate ({POSITIVE_CLASS_LABEL})")
    plt.ylabel(f"True Positive Rate ({POSITIVE_CLASS_LABEL})")
    plt.title(f"ROC Curve: {POSITIVE_CLASS_LABEL}")
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
    plt.xlabel(f"Mean Predicted Probability ({POSITIVE_CLASS_LABEL})")
    plt.ylabel(f"Observed Frequency ({POSITIVE_CLASS_LABEL})")
    plt.title(f"Calibration Reliability Diagram: {POSITIVE_CLASS_LABEL}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_training_plot(history: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_gamma_ablation_plot(ablation_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(ablation_frame["gamma"], ablation_frame["baseline_auc"], marker="o", label="Baseline AUC")
    plt.plot(ablation_frame["gamma"], ablation_frame["nars_gated_auc"], marker="o", label="NARS-gated AUC")
    plt.xscale("log", base=2)
    plt.xticks(ablation_frame["gamma"], [str(gamma) for gamma in ablation_frame["gamma"]])
    plt.xlabel("Gamma")
    plt.ylabel("AUC")
    plt.title(f"Gamma Ablation: {POSITIVE_CLASS_LABEL}")
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


def _save_decision_curve_plot(decision_curve_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["random_forest_net_benefit"],
        marker="o",
        label="Random forest",
    )
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["baseline_net_benefit"],
        marker="o",
        label="Baseline transformer",
    )
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["flat_confidence_net_benefit"],
        marker="o",
        label="Flat-confidence transformer",
    )
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["nars_gated_net_benefit"],
        marker="o",
        label="NARS-gated transformer",
    )
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["treat_all_net_benefit"],
        linestyle="--",
        label="Treat all",
    )
    plt.plot(
        decision_curve_frame["threshold"],
        decision_curve_frame["treat_none_net_benefit"],
        linestyle=":",
        label="Treat none",
    )
    plt.xlabel(f"Threshold Probability ({POSITIVE_CLASS_LABEL})")
    plt.ylabel("Net Benefit")
    plt.title(f"Decision Curve Analysis: {POSITIVE_CLASS_LABEL}")
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
    feature_confidence: np.ndarray,
    gated_attention: np.ndarray,
    token_score_mean: np.ndarray,
    gated_probabilities: np.ndarray,
) -> pd.DataFrame:
    trace_frame = bundle.test_frame[[DEFAULT_ID_COLUMN, DEFAULT_TARGET_COLUMN]].reset_index(drop=True).copy()
    trace_frame = trace_frame.rename(columns={DEFAULT_TARGET_COLUMN: "target"})
    base_columns = pd.DataFrame(
        {
            "baseline_probability": probabilities_mean,
            "baseline_variance": probabilities_var,
            "neural_frequency": neural_frequency,
            "neural_confidence": neural_confidence,
            "gated_probability": gated_probabilities,
            "attention_reliability": (gated_attention * feature_confidence).sum(axis=1),
        }
    )

    feature_columns: dict[str, np.ndarray] = {}
    for index, feature_name in enumerate(bundle.preprocessor.feature_names):
        safe_feature_name = feature_name.replace(" ", "_")
        feature_columns[f"attention_mean__{safe_feature_name}"] = attention_mean[:, index]
        feature_columns[f"attention_var__{safe_feature_name}"] = attention_var[:, index]
        feature_columns[f"attention_confidence__{safe_feature_name}"] = feature_confidence[:, index]
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
        gated_attention = apply_confidence_gate(
            attention_mean,
            feature_confidence,
            gamma=gamma,
        )
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


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dirs = _ensure_output_directories(config.output_dir)
    bundle = load_data_bundle(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    bundle.preprocessor.save(output_dirs["traces"] / "preprocessing_metadata.json")
    (output_dirs["traces"] / "split_summary.json").write_text(
        json.dumps(bundle.split_summary, indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformerClassifier(
        input_dim=bundle.preprocessor.input_dim,
        d_model=config.d_model,
        nhead=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    history = _train_model(model=model, bundle=bundle, config=config, device=device)
    history.to_csv(output_dirs["metrics"] / "training_history.csv", index=False)
    _save_training_plot(history, output_dirs["charts"] / "training_history.png")
    tree_baseline = _train_tree_baseline(bundle=bundle, config=config)

    summary = model.predict_with_mc_dropout(
        loader=bundle.test_loader,
        device=device,
        mc_samples=config.mc_samples,
    )
    neural_frequency, neural_confidence = neural_to_nars(
        summary.probabilities_mean,
        summary.probabilities_var,
    )
    _, feature_confidence = attention_to_nars(
        summary.attention_mean,
        summary.attention_var,
    )
    gated_attention = apply_confidence_gate(
        summary.attention_mean,
        feature_confidence,
        gamma=config.gamma,
    )
    flat_feature_confidence = np.full_like(feature_confidence, 0.5, dtype=np.float64)
    flat_attention = apply_confidence_gate(
        summary.attention_mean,
        flat_feature_confidence,
        gamma=config.gamma,
    )
    flat_logits = summary.cls_logit_mean + np.sum(
        flat_attention * summary.token_score_mean,
        axis=1,
    )
    flat_probabilities = _sigmoid(flat_logits)
    gated_logits = summary.cls_logit_mean + np.sum(
        gated_attention * summary.token_score_mean,
        axis=1,
    )
    gated_probabilities = _sigmoid(gated_logits)

    y_true = summary.labels.astype(int)
    tree_probabilities = tree_baseline["test_probabilities"]
    tree_reliability = _build_reliability_frame(y_true, tree_probabilities, n_bins=10)
    baseline_reliability = _build_reliability_frame(y_true, summary.probabilities_mean, n_bins=10)
    flat_reliability = _build_reliability_frame(y_true, flat_probabilities, n_bins=10)
    gated_reliability = _build_reliability_frame(y_true, gated_probabilities, n_bins=10)
    reliability_frame = pd.concat(
        [
            tree_reliability.assign(variant=TREE_BASELINE_LABEL),
            baseline_reliability.assign(variant="baseline"),
            flat_reliability.assign(variant="flat_confidence"),
            gated_reliability.assign(variant="nars_gated"),
        ],
        ignore_index=True,
    )
    auc_bootstrap = _bootstrap_auc_intervals(
        y_true=y_true,
        probability_map={
            TREE_BASELINE_LABEL: tree_probabilities,
            "baseline": summary.probabilities_mean,
            "flat_confidence": flat_probabilities,
            "nars_gated": gated_probabilities,
        },
        iterations=1000,
        seed=config.seed,
    )
    metrics_frame = pd.DataFrame(
        [
            {
                "variant": TREE_BASELINE_LABEL,
                **_compute_metrics(y_true, tree_probabilities),
                "auc_ci_95_lower": auc_bootstrap[TREE_BASELINE_LABEL]["auc_ci_95_lower"],
                "auc_ci_95_upper": auc_bootstrap[TREE_BASELINE_LABEL]["auc_ci_95_upper"],
            },
            {
                "variant": "baseline",
                **_compute_metrics(y_true, summary.probabilities_mean),
                "auc_ci_95_lower": auc_bootstrap["baseline"]["auc_ci_95_lower"],
                "auc_ci_95_upper": auc_bootstrap["baseline"]["auc_ci_95_upper"],
            },
            {
                "variant": "flat_confidence",
                **_compute_metrics(y_true, flat_probabilities),
                "auc_ci_95_lower": auc_bootstrap["flat_confidence"]["auc_ci_95_lower"],
                "auc_ci_95_upper": auc_bootstrap["flat_confidence"]["auc_ci_95_upper"],
            },
            {
                "variant": "nars_gated",
                **_compute_metrics(y_true, gated_probabilities),
                "auc_ci_95_lower": auc_bootstrap["nars_gated"]["auc_ci_95_lower"],
                "auc_ci_95_upper": auc_bootstrap["nars_gated"]["auc_ci_95_upper"],
            },
        ]
    )
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
        feature_confidence=np.asarray(feature_confidence),
        gated_attention=gated_attention,
        token_score_mean=summary.token_score_mean,
        gated_probabilities=gated_probabilities,
    )
    trace_frame.to_csv(output_dirs["traces"] / "test_predictions.csv", index=False)

    gamma_ablation_frame = _build_gamma_ablation_frame(
        y_true=y_true,
        baseline_probabilities=summary.probabilities_mean,
        attention_mean=summary.attention_mean,
        feature_confidence=np.asarray(feature_confidence),
        cls_logit_mean=summary.cls_logit_mean,
        token_score_mean=summary.token_score_mean,
    )
    gamma_ablation_frame.to_csv(output_dirs["metrics"] / "gamma_ablation.csv", index=False)
    _save_gamma_ablation_plot(gamma_ablation_frame, output_dirs["charts"] / "gamma_ablation_auc.png")

    decision_curve_frame = _build_decision_curve_frame(
        y_true=y_true,
        tree_probabilities=tree_probabilities,
        baseline_probabilities=summary.probabilities_mean,
        flat_probabilities=flat_probabilities,
        gated_probabilities=gated_probabilities,
    )
    decision_curve_frame.to_csv(output_dirs["metrics"] / "decision_curve.csv", index=False)
    _save_decision_curve_plot(decision_curve_frame, output_dirs["charts"] / "decision_curve.png")

    _save_roc_plot(
        y_true,
        tree_probabilities,
        summary.probabilities_mean,
        flat_probabilities,
        gated_probabilities,
        output_dirs["charts"] / "roc_curve.png",
    )
    _save_calibration_plot(
        y_true,
        tree_probabilities,
        summary.probabilities_mean,
        flat_probabilities,
        gated_probabilities,
        output_dirs["charts"] / "calibration_curve.png",
    )

    summary_frame = {
        "config": asdict(config),
        "task": {
            "dataset": "TCGA-THCA",
            "positive_class": POSITIVE_CLASS_LABEL,
            "id_column": DEFAULT_ID_COLUMN,
            "target_column": DEFAULT_TARGET_COLUMN,
        },
        "split_summary": bundle.split_summary,
        "metrics": metrics_frame.to_dict(orient="records"),
        "classical_baseline": {
            "variant": TREE_BASELINE_LABEL,
            "selection_objective": "validation_brier",
            "best_config": tree_baseline["best_config"],
            "validation_auc": tree_baseline["val_auc"],
            "validation_brier": tree_baseline["val_brier"],
        },
        "auc_bootstrap": auc_bootstrap,
        "gamma_ablation": gamma_ablation_frame.to_dict(orient="records"),
        "decision_curve": decision_curve_frame.to_dict(orient="records"),
        "calibration_reliability": reliability_frame.to_dict(orient="records"),
        "artifacts": {
            "metrics_csv": str(output_dirs["metrics"] / "metrics.csv"),
            "training_history_csv": str(output_dirs["metrics"] / "training_history.csv"),
            "gamma_ablation_csv": str(output_dirs["metrics"] / "gamma_ablation.csv"),
            "decision_curve_csv": str(output_dirs["metrics"] / "decision_curve.csv"),
            "calibration_reliability_csv": str(output_dirs["metrics"] / "calibration_reliability.csv"),
            "trace_csv": str(output_dirs["traces"] / "test_predictions.csv"),
            "roc_curve": str(output_dirs["charts"] / "roc_curve.png"),
            "calibration_curve": str(output_dirs["charts"] / "calibration_curve.png"),
            "training_history_plot": str(output_dirs["charts"] / "training_history.png"),
            "gamma_ablation_plot": str(output_dirs["charts"] / "gamma_ablation_auc.png"),
            "decision_curve_plot": str(output_dirs["charts"] / "decision_curve.png"),
        },
    }
    (output_dirs["metrics"] / "run_summary.json").write_text(
        json.dumps(summary_frame, indent=2),
        encoding="utf-8",
    )
    return summary_frame
