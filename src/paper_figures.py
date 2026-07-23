from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .knowledge_base import SYMBOLIC_RULES
from .wids_knowledge_base import WIDS_RULE_DEFINITIONS


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


def _pipeline_architecture(figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    stages = [
        (0.15, "1  Input", "Sparse clinical\nrecord", "#DBEAFE", "#2563EB"),
        (2.05, "2  Transformer", "Prediction $p$\nattention $\\alpha$\ntoken scores", "#EDE9FE", "#7C3AED"),
        (3.95, "3  MC dropout", "$T$ stochastic passes\nvariance $\\sigma^2$", "#FEF3C7", "#D97706"),
        (5.85, "4  NAL init", "Neural truth\n$(f,c)$", "#DCFCE7", "#16A34A"),
        (7.75, "5  Rule revision", "Independent triggers\nrevised $(f',c')$", "#FCE7F3", "#DB2777"),
        (9.65, "6  Gate + output", "$\\alpha'_i=\\frac{\\alpha_i(c'_i)^\\gamma}{\\sum_j\\alpha_j(c'_j)^\\gamma}$\nfinal $p^*$\nCSV/JSON trace", "#CFFAFE", "#0891B2"),
    ]
    for x, title, body, face, edge in stages:
        width = 2.15 if x == 9.65 else 1.65
        box = FancyBboxPatch(
            (x, 1.35),
            width,
            2.2,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 3.22, title, ha="center", va="center", weight="bold", color=edge, fontsize=9)
        ax.text(x + width / 2, 2.35, body, ha="center", va="center", fontsize=8.2)

    for left, right in zip(stages[:-1], stages[1:]):
        left_width = 2.15 if left[0] == 9.65 else 1.65
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left_width, 2.45),
                (right[0], 2.45),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#475569",
            )
        )

    rule_box = FancyBboxPatch(
        (7.95, 0.15),
        1.25,
        0.65,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="#FFF1F2",
        edgecolor="#E11D48",
        linewidth=1.2,
    )
    ax.add_patch(rule_box)
    ax.text(8.575, 0.48, "Rule base", ha="center", va="center", fontsize=8.5, weight="bold", color="#BE123C")
    ax.add_patch(
        FancyArrowPatch(
            (8.575, 0.8),
            (8.575, 1.35),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color="#E11D48",
        )
    )
    ax.text(5.9, 0.46, "Single inference pass; no recurrent convergence loop", ha="center", va="center", fontsize=8.5, color="#334155")
    _save(fig, figures_dir, "figure-1")


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


def _ablation_and_decision_curves(dataset_dirs: dict[str, Path], figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    for column, (dataset, key) in enumerate((("TCGA-THCA", "tcga"), ("WiDS ICU", "wids"))):
        gamma = pd.read_csv(dataset_dirs[key] / "metrics" / "gamma_ablation.csv")
        decision = pd.read_csv(dataset_dirs[key] / "metrics" / "decision_curve.csv")

        ax = axes[0, column]
        ax.axhline(0.0, color="#6B7280", linewidth=0.8, linestyle="--")
        ax.plot(
            gamma["gamma"],
            gamma["brier_delta_gated_minus_baseline"],
            color=COLORS["nars_gated"],
            marker="o",
            linewidth=1.3,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(gamma["gamma"], [str(value) for value in gamma["gamma"]])
        ax.set_title(f"{dataset}: gating sensitivity")
        ax.set_xlabel("$\\gamma$")
        ax.set_ylabel("Brier delta vs. baseline")
        ax.grid(color="#E5E7EB", linewidth=0.7)

        ax = axes[1, column]
        max_threshold = 0.8 if key == "tcga" else 0.5
        decision = decision[decision["threshold"] <= max_threshold]
        for variant, label in (
            ("baseline_net_benefit", "Baseline"),
            ("flat_confidence_net_benefit", "Flat confidence"),
            ("nars_gated_net_benefit", "NARS-gated"),
            ("treat_all_net_benefit", "Treat all"),
        ):
            style = "--" if variant == "treat_all_net_benefit" else "-"
            color_key = variant.replace("_net_benefit", "")
            ax.plot(
                decision["threshold"],
                decision[variant],
                label=label,
                color=COLORS.get(color_key, COLORS["gray"]),
                linestyle=style,
                linewidth=1.1,
            )
        ax.axhline(0.0, color="#111827", linewidth=0.7, linestyle=":")
        ax.set_title(f"{dataset}: decision curve")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Net benefit")
        ax.grid(color="#E5E7EB", linewidth=0.7)
        if column == 1:
            ax.legend(frameon=False, loc="best", ncol=2)
    _save(fig, figures_dir, "figure-3-ablation")


def _latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _write_rule_table(figures_dir: Path) -> None:
    rows: list[str] = []
    for dataset, definitions in (("TCGA", SYMBOLIC_RULES), ("WiDS", WIDS_RULE_DEFINITIONS)):
        for rule_id, definition in definitions.items():
            truth = definition["truth_value"]
            frequency = float(truth["frequency"])
            confidence = float(truth["confidence"])
            rows.append(
                " & ".join(
                    [
                        dataset,
                        _latex_escape(RULE_LABELS.get(rule_id, rule_id)),
                        _latex_escape(definition["condition"]),
                        f"$({frequency:.2f},{confidence:.2f})$",
                        _latex_escape(definition["clinical_interpretation"]),
                    ]
                )
                + " \\\\"
            )
    content = "\n".join(
        [
            "% Auto-generated by src.paper_figures.generate_paper_figures.",
            "\\begin{table}[t]",
            "\\caption{Prototype symbolic rules. Conditions are evaluated independently; any subset may fire. Truth values are evidential weights for the interface, not validated clinical probabilities.}",
            "\\label{tab:rule_inventory}",
            "\\centering",
            "\\tiny",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{@{}llp{0.22\\linewidth}cp{0.42\\linewidth}@{}}",
            "\\toprule",
            "Data & Rule & Trigger condition & $(f,c)$ & Clinical reading \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "generated_rule_table.tex").write_text(content, encoding="utf-8")


def _write_auditability_table(dataset_dirs: dict[str, Path], figures_dir: Path) -> None:
    audit_paths = {
        key: dataset_dirs[key] / "metrics" / "auditability_metrics.json"
        for key in ("tcga", "wids")
    }
    if not all(path.exists() for path in audit_paths.values()):
        return
    rows: list[str] = []
    for dataset, key in (("TCGA-THCA", "tcga"), ("WiDS ICU", "wids")):
        audit = json.loads(audit_paths[key].read_text(encoding="utf-8"))
        rows.append(
            f"{dataset} & {audit['cases_with_any_trigger']:,}/{audit['held_out_cases']:,} "
            f"({100 * audit['case_coverage']:.1f}\\%) & {audit['mapped_feature_trigger_count']:,} & "
            f"{100 * audit['finite_trace_rate']:.1f}\\% & "
            f"{100 * audit['confidence_changed_event_rate']:.1f}\\% & "
            f"{audit['threshold_flip_count_triggered']:,} & "
            f"{max(audit['max_revision_residual'], audit['max_gate_residual']):.1e} \\\\"
        )
    content = "\n".join(
        [
            "% Auto-generated by src.paper_figures.generate_paper_figures.",
            "\\begin{table}[t]",
            "\\caption{Operational auditability on held-out cases. Completeness and residuals test whether traces faithfully expose the implemented arithmetic; they do not measure clinician usability or rule validity.}",
            "\\label{tab:auditability}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}lrrrrrr@{}}",
            "\\toprule",
            "Data & Cases covered & Events & Complete & $c$ changed & Flips & Max residual \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "generated_auditability_table.tex").write_text(content, encoding="utf-8")


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
    _pipeline_architecture(figures_path)
    _performance_ci(dataset_dirs, figures_path)
    _ablation_and_decision_curves(dataset_dirs, figures_path)
    _write_rule_table(figures_path)
    _write_auditability_table(dataset_dirs, figures_path)

    generated = [
        figures_path / "figure-1.png",
        figures_path / "figure-2-performance-ci.png",
        figures_path / "figure-3-ablation.png",
        figures_path / "generated_rule_table.tex",
        figures_path / "generated_auditability_table.tex",
    ]
    return [path for path in generated if path.exists()]
