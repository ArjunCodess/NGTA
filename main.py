from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.attention_hook import apply_confidence_gate
from src.nars_interface import (
    confidence_to_evidence,
    deduce_truth_values,
    evidence_to_confidence,
    neural_to_nars,
    revise_truth_values,
    truth_to_expectation,
)
from src.pipeline import PipelineConfig, run_pipeline


def _run_self_checks() -> None:
    neural_frequency, neural_confidence = neural_to_nars(0.8, 0.05)
    if not np.isclose(neural_frequency, 0.8, atol=1e-6):
        raise RuntimeError("neural_to_nars frequency check failed.")
    if not np.isclose(neural_confidence, 0.7619160992, atol=1e-6):
        raise RuntimeError("neural_to_nars confidence check failed.")

    revised_frequency, revised_confidence = revise_truth_values(0.8, neural_confidence, 0.9, 0.7)
    if not np.isclose(revised_frequency, 0.8421671506, atol=1e-6):
        raise RuntimeError("revise_truth_values frequency check failed.")
    if not np.isclose(revised_confidence, 0.8469434609, atol=1e-6):
        raise RuntimeError("revise_truth_values confidence check failed.")

    clamped_frequency, clamped_confidence = revise_truth_values(
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    if not torch.isfinite(clamped_frequency).all() or not torch.isfinite(clamped_confidence).all():
        raise RuntimeError("revise_truth_values clamp check failed.")
    if not ((clamped_frequency >= 0.0).all() and (clamped_frequency <= 1.0).all()):
        raise RuntimeError("revise_truth_values clamped frequency bounds check failed.")
    if not ((clamped_confidence >= 0.0).all() and (clamped_confidence <= 1.0).all()):
        raise RuntimeError("revise_truth_values clamped confidence bounds check failed.")

    deduced_frequency, deduced_confidence = deduce_truth_values(0.85, 0.8823529411764706, 1.0, 1.0)
    if not np.isclose(deduced_frequency, 0.85, atol=1e-6):
        raise RuntimeError("deduce_truth_values frequency check failed.")
    if not np.isclose(deduced_confidence, 0.75, atol=1e-6):
        raise RuntimeError("deduce_truth_values confidence check failed.")

    confidence = evidence_to_confidence(9.0)
    if not np.isclose(confidence, 0.9, atol=1e-6):
        raise RuntimeError("evidence_to_confidence check failed.")
    evidence = confidence_to_evidence(confidence)
    if not np.isclose(evidence, 9.0, atol=1e-5):
        raise RuntimeError("confidence_to_evidence check failed.")
    expectation = truth_to_expectation(1.0, 0.8)
    if not np.isclose(expectation, 0.9, atol=1e-6):
        raise RuntimeError("truth_to_expectation check failed.")

    gated = apply_confidence_gate(
        np.array([[0.5, 0.3, 0.2]], dtype=np.float64),
        np.array([[0.9, 0.6, 0.2]], dtype=np.float64),
        gamma=2.0,
    )
    if not np.isclose(gated.sum(), 1.0, atol=1e-8):
        raise RuntimeError("apply_confidence_gate normalization check failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NGTA dual-dataset neurosymbolic tabular pipeline.",
    )
    parser.add_argument("--run-all", action="store_true", help="Run the full TCGA and WiDS pipelines sequentially.")
    parser.add_argument(
        "--dataset",
        default="tcga",
        choices=("tcga", "wids"),
        help="Dataset to run when --run-all is not specified.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the TCGA source tables/MAF files and the WiDS CSV file.",
    )
    parser.add_argument("--output-dir", default="results", help="Base directory for all generated result artifacts.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--mc-samples", type=int, default=50, help="Number of MC dropout inference passes.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Confidence gating exponent.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _run_self_checks()

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mc_samples=args.mc_samples,
        gamma=args.gamma,
        seed=args.seed,
        patience=args.patience,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    if args.run_all:
        summaries: dict[str, dict] = {}
        for dataset in ("tcga", "wids"):
            print(f"Running dataset pipeline: {dataset}")
            summaries[dataset] = run_pipeline(replace(config, dataset=dataset))
        summary = {
            "mode": "run_all",
            "datasets": summaries,
        }
        summary_path = Path(args.output_dir) / "run_all_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary = run_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
