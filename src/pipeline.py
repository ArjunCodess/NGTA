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
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from torch import nn

from .attention_hook import apply_confidence_gate, attention_to_nars
from .data_loader import load_data_bundle
from .nars_interface import neural_to_nars
from .neural_encoder import TabularTransformerClassifier


@dataclass
class PipelineConfig:
    data_path: str = "data.csv"
    output_dir: str = "."
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mc_samples: int = 50
    gamma: float = 2.0
    seed: int = 0
    patience: int = 12
    validation_size: float = 0.2
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
        for categorical_inputs, numerical_inputs, target in loader:
            categorical_inputs = categorical_inputs.to(device)
            numerical_inputs = numerical_inputs.to(device)
            target = target.to(device)
            output = model(categorical_inputs, numerical_inputs)
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
        for categorical_inputs, numerical_inputs, target in bundle.train_loader:
            categorical_inputs = categorical_inputs.to(device)
            numerical_inputs = numerical_inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(categorical_inputs, numerical_inputs)
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
    return {
        "auc": float(roc_auc_score(y_true, clipped_probabilities)),
        "brier": float(brier_score_loss(y_true, clipped_probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


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

    baseline_interval = intervals["baseline"]
    gated_interval = intervals["nars_gated"]
    overlap = not (
        baseline_interval["auc_ci_95_upper"] < gated_interval["auc_ci_95_lower"]
        or gated_interval["auc_ci_95_upper"] < baseline_interval["auc_ci_95_lower"]
    )
    intervals["comparison"] = {
        "ci_overlap": overlap,
        "interpretation": (
            "The 95% bootstrap AUC confidence intervals overlap, so the observed difference should be treated as uncertain."
            if overlap
            else "The 95% bootstrap AUC confidence intervals do not overlap, so the observed difference is likely real on this test split."
        ),
    }
    return intervals


def _save_roc_plot(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    baseline_fpr, baseline_tpr, _ = roc_curve(y_true, baseline_probabilities)
    gated_fpr, gated_tpr, _ = roc_curve(y_true, gated_probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(baseline_fpr, baseline_tpr, label="Baseline")
    plt.plot(gated_fpr, gated_tpr, label="NARS-gated")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _save_calibration_plot(
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    for label, probabilities in (
        ("Baseline", baseline_probabilities),
        ("NARS-gated", gated_probabilities),
    ):
        fraction_positives, mean_predicted_value = calibration_curve(
            y_true,
            probabilities,
            n_bins=8,
            strategy="uniform",
        )
        plt.plot(mean_predicted_value, fraction_positives, marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
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
    trace_frame = bundle.test_frame.reset_index(drop=True).copy()
    trace_frame["baseline_probability"] = probabilities_mean
    trace_frame["baseline_variance"] = probabilities_var
    trace_frame["neural_frequency"] = neural_frequency
    trace_frame["neural_confidence"] = neural_confidence
    trace_frame["gated_probability"] = gated_probabilities
    trace_frame["attention_reliability"] = (gated_attention * feature_confidence).sum(axis=1)

    for index, feature_name in enumerate(bundle.preprocessor.feature_names):
        safe_feature_name = feature_name.replace(" ", "_")
        trace_frame[f"attention_mean__{safe_feature_name}"] = attention_mean[:, index]
        trace_frame[f"attention_var__{safe_feature_name}"] = attention_var[:, index]
        trace_frame[f"attention_confidence__{safe_feature_name}"] = feature_confidence[:, index]
        trace_frame[f"gated_attention__{safe_feature_name}"] = gated_attention[:, index]
        trace_frame[f"token_score__{safe_feature_name}"] = token_score_mean[:, index]

    return trace_frame


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    set_seed(config.seed)
    output_dirs = _ensure_output_directories(config.output_dir)
    bundle = load_data_bundle(
        data_path=config.data_path,
        batch_size=config.batch_size,
        seed=config.seed,
        validation_size=config.validation_size,
    )
    bundle.preprocessor.save(output_dirs["traces"] / "preprocessing_metadata.json")
    (output_dirs["traces"] / "split_summary.json").write_text(
        json.dumps(bundle.split_summary, indent=2),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformerClassifier(
        categorical_cardinalities=bundle.preprocessor.categorical_cardinalities,
        num_numeric_features=len(bundle.preprocessor.numeric_columns),
        d_model=config.d_model,
        nhead=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    history = _train_model(model=model, bundle=bundle, config=config, device=device)
    history.to_csv(output_dirs["metrics"] / "training_history.csv", index=False)
    _save_training_plot(history, output_dirs["charts"] / "training_history.png")

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
    gated_logits = summary.cls_logit_mean + np.sum(
        gated_attention * summary.token_score_mean,
        axis=1,
    )
    gated_probabilities = _sigmoid(gated_logits)

    y_true = summary.labels.astype(int)
    auc_bootstrap = _bootstrap_auc_intervals(
        y_true=y_true,
        probability_map={
            "baseline": summary.probabilities_mean,
            "nars_gated": gated_probabilities,
        },
        iterations=1000,
        seed=config.seed,
    )
    metrics_frame = pd.DataFrame(
        [
            {
                "variant": "baseline",
                **_compute_metrics(y_true, summary.probabilities_mean),
                "auc_ci_95_lower": auc_bootstrap["baseline"]["auc_ci_95_lower"],
                "auc_ci_95_upper": auc_bootstrap["baseline"]["auc_ci_95_upper"],
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

    _save_roc_plot(
        y_true,
        summary.probabilities_mean,
        gated_probabilities,
        output_dirs["charts"] / "roc_curve.png",
    )
    _save_calibration_plot(
        y_true,
        summary.probabilities_mean,
        gated_probabilities,
        output_dirs["charts"] / "calibration_curve.png",
    )

    summary_frame = {
        "config": asdict(config),
        "split_summary": bundle.split_summary,
        "metrics": metrics_frame.to_dict(orient="records"),
        "auc_bootstrap": auc_bootstrap,
        "artifacts": {
            "metrics_csv": str(output_dirs["metrics"] / "metrics.csv"),
            "training_history_csv": str(output_dirs["metrics"] / "training_history.csv"),
            "trace_csv": str(output_dirs["traces"] / "test_predictions.csv"),
            "roc_curve": str(output_dirs["charts"] / "roc_curve.png"),
            "calibration_curve": str(output_dirs["charts"] / "calibration_curve.png"),
            "training_history_plot": str(output_dirs["charts"] / "training_history.png"),
        },
    }
    (output_dirs["metrics"] / "run_summary.json").write_text(
        json.dumps(summary_frame, indent=2),
        encoding="utf-8",
    )
    return summary_frame
