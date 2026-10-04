"""Generate manuscript tables from completed, independently replayed studies."""
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

CONDITIONS={
    "WiDS KNN":"results/hospital_study/knn",
    "WiDS median":"results/hospital_study/median",
    "WiDS KNN, no APACHE":"results/hospital_study/knn_no_apache",
    "WiDS median, no APACHE":"results/hospital_study/median_no_apache",
    "TCGA fused":"results/tcga_study/fused",
    "TCGA clinical":"results/tcga_study/clinical",
    "TCGA genomic":"results/tcga_study/genomic",
}


def refresh():
    records=[]
    hierarchy=[]
    for condition,root in CONDITIONS.items():
        frame=pd.read_csv(Path(root)/"analysis/seed_variability.csv")
        for variant in ("baseline","mc_confidence_only","nars_gated","hist_gradient_boosting","transformer_recalibrated"):
            row=frame.set_index("variant").loc[variant]
            records.append(dict(condition=condition,variant=variant,auc=row.auc_mean,brier=row.brier_mean,
                brier_seed_std=row.brier_std))
        report=json.loads((Path(root)/"analysis/hierarchical_symbolic_comparison.json").read_text())
        hierarchy.append(dict(condition=condition,**report))
    root=Path("paper/figures")
    root.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(root/"current_results.csv",index=False)
    labels={"baseline":"Ungated","mc_confidence_only":"MC-only","nars_gated":"NARS","hist_gradient_boosting":"Boosting","transformer_recalibrated":"Recalibrated"}
    lines=[r"\begin{table}[t]",r"\caption{Current development evaluation: five training seeds per condition on fixed held-out cases. Brier values are means with seed standard deviations, not case-bootstrap confidence intervals. The APACHE score remains a score-only comparator when removed from predictor inputs.}",
        r"\label{tab:current_results}",r"\centering\scriptsize",r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{@{}llcc@{}}",r"\toprule",r"Condition & Variant & AUROC & Brier (seed SD) \\",r"\midrule"]
    for record in records:
        if record["condition"] not in ("WiDS KNN","TCGA fused"):
            continue
        lines.append(f'{record["condition"]} & {labels[record["variant"]]} & {record["auc"]:.6f} & {record["brier"]:.6f} ({record["brier_seed_std"]:.6f}) '+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",r"\begin{table}[t]",r"\caption{Matched symbolic contrasts, NARS-minus-MC Brier, with hierarchical 95\% intervals over cases/hospitals, training seeds, and three dropout repeats. Negative is better. These development comparisons do not meet the prespecified $10^{-4}$ minimum benefit.}",r"\label{tab:current_symbolic}",r"\centering\scriptsize",r"\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",r"Condition & Difference & Lower & Upper \\",r"\midrule"]
    for row in hierarchy:
        lines.append(f'{row["condition"]} & {row["brier_nars_minus_mc"]:.2e} & {row["lower_95"]:.2e} & {row["upper_95"]:.2e} '+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (root/"current_results.tex").write_text("\n".join(lines)+"\n")
    (root/"current_results_manifest.json").write_text(json.dumps(dict(conditions=CONDITIONS,
        training_seeds=list(range(5)),mc_samples=50,mc_repeats=3,
        scope="fixed development splits; no prospective or independent-cohort confirmation"),indent=2))
    return records,hierarchy


if __name__=="__main__":
    refresh()
