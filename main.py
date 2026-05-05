from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

TRANSFORMER_VARIANTS = {"baseline", "flat_confidence", "mc_confidence_only", "nars_gated"}


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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run one or more seeds and aggregate submission-ready outputs.",
    )
    parser.add_argument(
        "--baseline-set",
        choices=("minimal", "standard"),
        default="minimal",
        help="Classical baseline suite. 'standard' adds calibrated logistic regression, ExtraTrees, and histogram gradient boosting.",
    )
    parser.add_argument(
        "--ablation-set",
        choices=("quick", "submission"),
        default="quick",
        help="Ablation suite. 'submission' adds symbolic-disabled and rule-truth sensitivity summaries.",
    )
    parser.add_argument(
        "--export-case-traces",
        action="store_true",
        help="Export curated glass-box case traces for representative held-out cases.",
    )
    parser.add_argument(
        "--paper-tables",
        action="store_true",
        help="Write submission-ready aggregate CSV and LaTeX table artifacts under <output-dir>/submission.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _metrics_to_latex(metrics_frame: pd.DataFrame) -> str:
    columns = ["dataset", "variant", "auc_mean", "auc_std", "brier_mean", "brier_std", "ece_mean", "ece_std"]
    available_columns = [column for column in columns if column in metrics_frame.columns]
    if metrics_frame.empty:
        return "% No metrics available.\n"
    return metrics_frame[available_columns].to_latex(index=False, float_format="%.5f")


def _write_submission_outputs(summaries: list[dict[str, Any]], output_dir: str | Path, write_paper_tables: bool) -> None:
    submission_dir = Path(output_dir) / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    case_trace_rows: list[dict[str, Any]] = []

    for summary in summaries:
        dataset = summary["task"]["dataset_key"]
        seed = summary["config"]["seed"]
        for metric in summary["metrics"]:
            row = {"dataset": dataset, "seed": seed, **metric}
            metric_rows.append(row)
            if metric["variant"] not in TRANSFORMER_VARIANTS:
                baseline_rows.append(row)
        for ablation in summary.get("submission_ablation", []):
            ablation_rows.append({"dataset": dataset, "seed": seed, **ablation})
        for case_trace in summary.get("case_traces", []):
            case_trace_rows.append({"dataset": dataset, "seed": seed, **case_trace})

    metrics_frame = pd.DataFrame(metric_rows)
    if not metrics_frame.empty:
        aggregate = (
            metrics_frame.groupby(["dataset", "variant"], as_index=False)
            .agg(
                auc_mean=("auc", "mean"),
                auc_std=("auc", "std"),
                brier_mean=("brier", "mean"),
                brier_std=("brier", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                ece_mean=("ece", "mean"),
                ece_std=("ece", "std"),
                runs=("seed", "nunique"),
            )
            .fillna(0.0)
        )
        aggregate.to_csv(submission_dir / "multiseed_metrics.csv", index=False)
    else:
        aggregate = pd.DataFrame()
        aggregate.to_csv(submission_dir / "multiseed_metrics.csv", index=False)

    pd.DataFrame(baseline_rows).to_csv(submission_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(submission_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(case_trace_rows).to_csv(submission_dir / "case_traces.csv", index=False)

    if write_paper_tables:
        (submission_dir / "paper_tables.tex").write_text(_metrics_to_latex(aggregate), encoding="utf-8")


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
        baseline_set=args.baseline_set,
        ablation_set=args.ablation_set,
        export_case_traces=args.export_case_traces,
    )

    requested_seeds = args.seeds or [args.seed]
    requested_datasets = ("tcga", "wids") if args.run_all else (args.dataset,)
    all_summaries: list[dict[str, Any]] = []

    if args.run_all or args.seeds:
        summaries: dict[str, dict] = {}
        for seed in requested_seeds:
            for dataset in requested_datasets:
                print(f"Running dataset pipeline: {dataset} (seed={seed})")
                seed_output_dir = Path(args.output_dir)
                if args.seeds:
                    seed_output_dir = seed_output_dir / f"seed_{seed}"
                run_summary = run_pipeline(replace(config, dataset=dataset, seed=seed, output_dir=str(seed_output_dir)))
                all_summaries.append(run_summary)
                summary_key = f"{dataset}_seed_{seed}" if args.seeds else dataset
                summaries[summary_key] = run_summary
        summary = {
            "mode": "run_all" if args.run_all else "multi_seed",
            "seeds": requested_seeds,
            "datasets": summaries,
        }
        summary_path = Path(args.output_dir) / "run_all_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    else:
        summary = run_pipeline(config)
        all_summaries.append(summary)

    if args.paper_tables or args.seeds or args.export_case_traces or args.ablation_set == "submission" or args.baseline_set == "standard":
        _write_submission_outputs(all_summaries, args.output_dir, write_paper_tables=args.paper_tables)

    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
