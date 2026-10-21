"""Reuse fixed NGTA demographic strata and research decision thresholds in Nepal."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.subgroups import subgroup_reports


def diagnostics(root="results/private/nepal_sensitivity"):
    root = Path(root)
    source = Path("results/research_checks/nepal_sensitivity.json")
    report = json.loads(source.read_text())
    if report.get("fits") != 20:
        raise ValueError("Complete and verify all Nepal evaluations first")
    rows, strata_rows, verified = [], 0, 0
    for condition in report["conditions"]:
        scenario, imputation = condition["scenario"], condition["imputation"]
        cohort_path = root/scenario/"cohort.csv"
        if hashlib.sha256(cohort_path.read_bytes()).hexdigest() != report["preparation"]["scenarios"][scenario]["cohort_sha256"]:
            raise ValueError("Cohort changed before diagnostic analysis")
        cohort = pd.read_csv(cohort_path)
        for fit in condition["verification"]:
            seed = fit["seed"]
            output = root/scenario/imputation/f"seed_{seed}"
            path = output/"predictions.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest() != fit["predictions_sha256"]:
                raise ValueError("Prediction identity differs from the verified study")
            predictions = pd.read_csv(path)
            if not predictions.encounter_id.equals(cohort.encounter_id) or not np.array_equal(predictions.target, cohort.icu_death):
                raise ValueError("Diagnostic admission/outcome alignment differs")
            y = predictions.target.to_numpy()
            variants = {name: predictions[name].to_numpy() for name in ("baseline", "flat_confidence", "mc_confidence_only", "nars_gated")}
            metrics, curves = subgroup_reports(cohort, y, variants)
            metrics.to_csv(output/"subgroup_metrics.csv", index=False)
            curves.to_csv(output/"subgroup_decision_curves.csv", index=False)
            strata_rows += len(metrics)
            verified += 1
            for name, p in variants.items():
                for threshold in (.05, .1, .2, .3, .5):
                    chosen = p >= threshold
                    benefit = (np.sum(chosen & (y == 1))-np.sum(chosen & (y == 0))*threshold/(1-threshold))/len(y)
                    rows.append(dict(scenario=scenario, imputation=imputation, seed=seed, variant=name,
                        threshold=threshold, net_benefit=float(benefit), treat_all=float(y.mean()-(1-y.mean())*threshold/(1-threshold)), treat_none=0.))
    table = pd.DataFrame(rows)
    table.to_csv(root/"decision_curves.csv", index=False)
    average = table.groupby(["scenario", "imputation", "variant", "threshold"], as_index=False)[["net_benefit", "treat_all", "treat_none"]].mean()
    summary = dict(evaluation_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        evaluations=verified, private_subgroup_metric_rows=strata_rows,
        fixed_age_cutoff=65, fixed_thresholds=[.05, .1, .2, .3, .5],
        interpretation="Additional descriptive diagnostics using pre-existing NGTA strata/thresholds, not a separately preregistered clinical analysis. Small-stratum results remain private. No protected-group safety, clinical threshold validation or hospital-mortality compatibility is claimed.",
        mean_over_five_seeds=average.to_dict("records"))
    Path("results/research_checks/nepal_diagnostics.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = diagnostics()
    print(f"Verified {result['evaluations']} diagnostic evaluations; subgroup rows={result['private_subgroup_metric_rows']}")
