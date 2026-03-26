from __future__ import annotations

import argparse
import json

import numpy as np

from src.attention_hook import apply_confidence_gate
from src.nars_interface import nars_revision, neural_to_nars
from src.pipeline import PipelineConfig, run_pipeline


def _run_self_checks() -> None:
    neural_frequency, neural_confidence = neural_to_nars(0.8, 0.05)
    if not np.isclose(neural_frequency, 0.8, atol=1e-6):
        raise RuntimeError("neural_to_nars frequency check failed.")
    if not np.isclose(neural_confidence, 0.7619160992, atol=1e-6):
        raise RuntimeError("neural_to_nars confidence check failed.")

    revised_frequency, revised_confidence = nars_revision(0.8, neural_confidence, 0.9, 0.7)
    if not np.isclose(revised_frequency, 0.8421671506, atol=1e-6):
        raise RuntimeError("nars_revision frequency check failed.")
    if not np.isclose(revised_confidence, 0.8469434609, atol=1e-6):
        raise RuntimeError("nars_revision confidence check failed.")

    gated = apply_confidence_gate(
        np.array([[0.5, 0.3, 0.2]], dtype=np.float64),
        np.array([[0.9, 0.6, 0.2]], dtype=np.float64),
        gamma=2.0,
    )
    if not np.isclose(gated.sum(), 1.0, atol=1e-8):
        raise RuntimeError("apply_confidence_gate normalization check failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NGTA MTC diagnosis pipeline.",
    )
    parser.add_argument("--run-all", action="store_true", help="Run the full training and evaluation pipeline.")
    parser.add_argument("--data-path", default="data.csv", help="Path to the input CSV dataset.")
    parser.add_argument("--output-dir", default=".", help="Base directory for charts and results.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--mc-samples", type=int, default=50, help="Number of MC dropout inference passes.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Confidence gating exponent.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction inside the training split.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _run_self_checks()

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mc_samples=args.mc_samples,
        gamma=args.gamma,
        seed=args.seed,
        patience=args.patience,
        validation_size=args.validation_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
