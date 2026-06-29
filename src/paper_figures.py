from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "baseline": "#4C78A8",
    "flat_confidence": "#72B7B2",
    "mc_confidence_only": "#F58518",
    "nars_gated": "#54A24B",
    "random_forest": "#B279A2",
    "accent": "#E45756",
    "gray": "#6B7280",
}


VARIANT_LABELS = {
    "random_forest": "Random forest",
    "baseline": "Baseline",
    "flat_confidence": "Flat confidence",
    "mc_confidence_only": "MC confidence",
    "nars_gated": "NARS-gated",
}


RULE_LABELS = {
    "braf_mutation": "BRAF mutation",
    "age_ge_55_years": "Age >= 55",
    "pathologic_t_t3_t4": "Pathologic T3/T4",
    "extrathyroid_extension_present": "Extrathyroid extension",
    "rule_lactate": "Lactate",
    "rule_hypotension": "Hypotension",
    "rule_age": "Age",
    "rule_creatinine": "Creatinine",
}


FEATURE_LABELS = {
    "d1_sysbp_min": "Systolic BP min",
    "d1_lactate_max": "Lactate max",
    "d1_creatinine_max": "Creatinine max",
    "age": "Age",
}


CLASSIFICATION_THRESHOLD = 0.5


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _dataset_result_dir(results_dir: Path, dataset: str, seeds: list[int] | None) -> Path:
    direct = results_dir / dataset
    if direct.exists():
        return direct
    for seed in seeds or []:
        seeded = results_dir / f"seed_{seed}" / dataset
        if seeded.exists():
            return seeded
    seeded_dirs = sorted(results_dir.glob(f"seed_*{dataset}"))
    if seeded_dirs:
        return seeded_dirs[0]
    nested_seeded_dirs = sorted(results_dir.glob(f"seed_*/{dataset}"))
    if nested_seeded_dirs:
        return nested_seeded_dirs[0]
    raise FileNotFoundError(f"Could not locate results for dataset '{dataset}' under {results_dir}.")


def _case_traces_path(results_dir: Path, wids_dir: Path) -> Path:
    direct = wids_dir / "traces" / "case_traces.csv"
    if direct.exists():
        return direct
    submission = results_dir / "submission" / "case_traces.csv"
    if submission.exists():
        return submission
    raise FileNotFoundError(f"Could not locate WiDS case traces under {results_dir}.")


def _performance_ci(dataset_dirs: dict[str, Path], figures_dir: Path) -> None:
    datasets = [
        ("TCGA-THCA", dataset_dirs["tcga"] / "metrics" / "metrics.csv"),
        ("WiDS ICU", dataset_dirs["wids"] / "metrics" / "metrics.csv"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=False)
    metric = "brier"
    for ax, (dataset, path) in zip(axes, datasets):
        df = pd.read_csv(path)
        order = ["random_forest", "baseline", "flat_confidence", "mc_confidence_only", "nars_gated"]
        df = df.set_index("variant").loc[order].reset_index()
        y = list(range(len(df)))
        values = df[metric]
        lower = values - df[f"{metric}_ci_95_lower"]
        upper = df[f"{metric}_ci_95_upper"] - values
        colors = [COLORS[v] for v in df["variant"]]
        ax.errorbar(values, y, xerr=[lower, upper], fmt="none", ecolor="#374151", elinewidth=1, capsize=2)
        ax.scatter(values, y, s=32, c=colors, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([VARIANT_LABELS[v] for v in df["variant"]])
        ax.invert_yaxis()
        ax.set_title(dataset)
        ax.set_xlabel("Brier score, 95% bootstrap CI")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    _save(fig, figures_dir, "figure-2-performance-ci")


def _rule_firing_counts(dataset_dirs: dict[str, Path], figures_dir: Path) -> None:
    rows = []
    for dataset, key in [("TCGA-THCA", "tcga"), ("WiDS ICU", "wids")]:
        summary = json.loads((dataset_dirs[key] / "metrics" / "run_summary.json").read_text(encoding="utf-8"))
        counts = summary["symbolic_rules"]["per_rule_trigger_counts"]
        for rule, count in counts.items():
            rows.append({"dataset": dataset, "rule": RULE_LABELS.get(rule, rule), "count": count})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    for ax, dataset in zip(axes, ["TCGA-THCA", "WiDS ICU"]):
        sub = df[df["dataset"] == dataset].sort_values("count", ascending=True)
        ax.barh(sub["rule"], sub["count"], color="#4C78A8" if dataset == "TCGA-THCA" else "#54A24B")
        ax.set_title(dataset)
        ax.set_xlabel("Triggered rules on held-out cases")
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        for i, value in enumerate(sub["count"]):
            ax.text(value, i, f" {int(value):,}", va="center", fontsize=8)
    _save(fig, figures_dir, "figure-3-rule-firing")


def _select_case_trace(traces: pd.DataFrame) -> pd.Series:
    if "dataset" in traces.columns:
        traces = traces[traces["dataset"].astype(str).str.lower() == "wids"]
    traces = traces.copy()
    if traces.empty:
        raise ValueError("No WiDS case traces are available for Figure 4.")

    traces["probability_delta"] = (traces["nars_gated_probability"] - traces["baseline_probability"]).abs()
    traces["distance_to_threshold"] = (traces["baseline_probability"] - CLASSIFICATION_THRESHOLD).abs()
    traces["baseline_prediction"] = (traces["baseline_probability"] >= CLASSIFICATION_THRESHOLD).astype(int)
    traces["gated_prediction"] = (traces["nars_gated_probability"] >= CLASSIFICATION_THRESHOLD).astype(int)
    traces["prediction_changed"] = traces["baseline_prediction"] != traces["gated_prediction"]

    if "case_category" in traces.columns:
        near = traces[
            traces["case_category"].astype(str).str.contains("near", case=False, na=False)
            & (traces["symbolic_rule_count"] > 0)
        ]
        if not near.empty:
            return near.sort_values(
                ["prediction_changed", "symbolic_rule_count", "probability_delta", "distance_to_threshold"],
                ascending=[False, False, False, True],
            ).iloc[0]

    symbolic = traces[traces["symbolic_rule_count"] > 0]
    if not symbolic.empty:
        return symbolic.sort_values(
            ["prediction_changed", "symbolic_rule_count", "probability_delta", "distance_to_threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    return traces.sort_values(["probability_delta", "distance_to_threshold"], ascending=[False, True]).iloc[0]


def _latex_escape(value: object) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")


def _write_metadata(figures_dir: Path, row: pd.Series) -> None:
    case_id = row.get("case_id", row.get("encounter_id", "selected"))
    baseline_probability = float(row["baseline_probability"])
    nars_gated_probability = float(row["nars_gated_probability"])
    symbolic_rule_count = int(row["symbolic_rule_count"])
    baseline_prediction = int(baseline_probability >= CLASSIFICATION_THRESHOLD)
    gated_prediction = int(nars_gated_probability >= CLASSIFICATION_THRESHOLD)
    target = int(row["target"]) if "target" in row else -1
    metadata = "\n".join(
        [
            "% Auto-generated by src.paper_figures.generate_paper_figures.",
            f"\\newcommand{{\\GeneratedCaseTraceId}}{{{_latex_escape(case_id)}}}",
            f"\\newcommand{{\\GeneratedCaseTraceBaselineProbability}}{{{baseline_probability:.4f}}}",
            f"\\newcommand{{\\GeneratedCaseTraceNarsProbability}}{{{nars_gated_probability:.4f}}}",
            f"\\newcommand{{\\GeneratedCaseTraceRuleCount}}{{{symbolic_rule_count}}}",
            f"\\newcommand{{\\GeneratedCaseTraceBaselinePrediction}}{{{baseline_prediction}}}",
            f"\\newcommand{{\\GeneratedCaseTraceNarsPrediction}}{{{gated_prediction}}}",
            f"\\newcommand{{\\GeneratedCaseTraceTarget}}{{{target}}}",
            "",
        ]
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "generated_figure_metadata.tex").write_text(metadata, encoding="utf-8")


def _format_truth(frequency: float, confidence: float) -> str:
    return f"$({frequency:.4f},{confidence:.4f})$"


def _write_case_trace_table(figures_dir: Path, row: pd.Series, trace_frame: pd.DataFrame) -> None:
    case_id = row.get("case_id", row.get("encounter_id", "selected"))
    table_rows = []
    for item in trace_frame.sort_values("feature_label").to_dict(orient="records"):
        feature = _latex_escape(item["feature_label"])
        neural = _format_truth(float(item["neural_f"]), float(item["neural_c"]))
        symbolic = _format_truth(float(item["symbolic_f"]), float(item["symbolic_c"]))
        revised = _format_truth(float(item["revised_f"]), float(item["revised_c"]))
        attention = f"${float(item['attention_before']):.4f} \\rightarrow {float(item['attention_after']):.4f}$"
        table_rows.append(f"{feature} & {neural} & {symbolic} & {revised} & {attention} \\\\")

    content = "\n".join(
        [
            "% Auto-generated by src.paper_figures.generate_paper_figures.",
            "\\begin{table}[t]",
            (
                "\\caption{Concrete WiDS rule trace for held-out encounter "
                f"{_latex_escape(case_id)}. Probability changed from "
                "\\GeneratedCaseTraceBaselineProbability{} before routing to "
                "\\GeneratedCaseTraceNarsProbability{} after NARS-gated routing; "
                "the thresholded prediction changed from "
                "\\GeneratedCaseTraceBaselinePrediction{} to "
                "\\GeneratedCaseTraceNarsPrediction{}, while the target was "
                "\\GeneratedCaseTraceTarget.}"
            ),
            "\\label{tab:case_trace}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Feature & Neural $(f,c)$ & Rule $(f,c)$ & Revised $(f,c)$ & Attention before $\\rightarrow$ after \\\\",
            "\\midrule",
            *table_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "generated_case_trace_table.tex").write_text(content, encoding="utf-8")


def _case_trace(results_dir: Path, wids_dir: Path, figures_dir: Path) -> None:
    traces = pd.read_csv(_case_traces_path(results_dir, wids_dir))
    row = _select_case_trace(traces)
    feature_trace = json.loads(row["feature_trace_json"])
    df = pd.DataFrame(feature_trace)
    df["feature_label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df = df.sort_values("attention_delta")

    case_id = row.get("case_id", row.get("encounter_id", "selected"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    probs = [row["baseline_probability"], row["nars_gated_probability"]]
    labels = ["Before routing", "After routing"]
    ax.plot([0, 1], probs, color=COLORS["gray"], linewidth=1.2)
    ax.scatter([0, 1], probs, color=[COLORS["baseline"], COLORS["nars_gated"]], s=40, zorder=3)
    ax.axhline(
        CLASSIFICATION_THRESHOLD,
        color=COLORS["accent"],
        linestyle="--",
        linewidth=1,
        label=f"{CLASSIFICATION_THRESHOLD:.1f} threshold",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Predicted probability")
    ymin = min(0.49, min(probs) - 0.01)
    ymax = max(0.515, max(probs) + 0.01)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.08, 1.08)
    ax.set_title(f"Encounter {case_id}")
    for x, y in zip([0, 1], probs):
        dy = -0.0018 if x == 0 else 0.0012
        ax.text(x, y + dy, f"{y:.4f}", ha="center", fontsize=8)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    y = list(range(len(df)))
    ax.barh(y, df["attention_delta"], color=[COLORS["accent"] if v < 0 else COLORS["nars_gated"] for v in df["attention_delta"]])
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature_label"])
    ax.set_xlabel("Attention change after rule revision")
    ax.set_title("Fired-rule attention deltas")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    xmin = min(df["attention_delta"].min() * 1.18, -0.00025)
    xmax = max(df["attention_delta"].max() * 1.45, 0.00025)
    ax.set_xlim(xmin, xmax)
    for i, value in enumerate(df["attention_delta"]):
        if value < 0:
            ax.text(value / 2, i, f"{value:+.4f}", va="center", ha="center", fontsize=8, color="white")
        else:
            ax.text(value + 0.00008, i, f"{value:+.4f}", va="center", ha="left", fontsize=8)
    _save(fig, figures_dir, "figure-4-case-trace")
    _write_metadata(figures_dir, row)
    _write_case_trace_table(figures_dir, row, df)


def generate_paper_figures(
    results_dir: str | Path = "results",
    figures_dir: str | Path = "paper/figures",
    seeds: list[int] | None = None,
) -> list[Path]:
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    dataset_dirs = {
        "tcga": _dataset_result_dir(results_path, "tcga", seeds),
        "wids": _dataset_result_dir(results_path, "wids", seeds),
    }

    _set_style()
    _performance_ci(dataset_dirs, figures_path)
    _rule_firing_counts(dataset_dirs, figures_path)
    _case_trace(results_path, dataset_dirs["wids"], figures_path)

    return [
        figures_path / "figure-2-performance-ci.png",
        figures_path / "figure-3-rule-firing.png",
        figures_path / "figure-4-case-trace.png",
        figures_path / "generated_figure_metadata.tex",
        figures_path / "generated_case_trace_table.tex",
    ]
