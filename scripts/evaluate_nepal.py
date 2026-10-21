"""Run and verify every frozen Nepal scenario; publish aggregate evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import binary_metrics
from src.external_validation import evaluate_external
from src.nepal_registry import SCENARIOS
from src.trace_replay import replay_bundle


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_values(value):
    """Keep undefined PPV/calibration values null, never zero or non-JSON NaN."""
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_values(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def hierarchical_brier(predictions, labels, iterations=2000, seed=0):
    """Resample admissions and fitted seeds, fixing their original inference draw."""
    y = np.asarray(labels, dtype=float)
    differences = np.asarray([(n-y)**2-(m-y)**2 for n, m in predictions])
    if differences.shape != (5, len(y)) or not np.isfinite(differences).all():
        raise ValueError("Expected five aligned finite paired seed losses")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        cases = rng.integers(0, len(y), len(y))
        seeds = rng.integers(0, 5, 5)
        samples.append(float(differences[seeds][:, cases].mean()))
    tail = 100*.05/(2*len(SCENARIOS)*2)
    return dict(brier=float(differences.mean()), lower_95=float(np.percentile(samples, 2.5)),
        upper_95=float(np.percentile(samples, 97.5)),
        family_lower_95=float(np.percentile(samples, tail)),
        family_upper_95=float(np.percentile(samples, 100-tail)),
        iterations=iterations, family_comparisons=4,
        sampling="admissions and five fitted training seeds; fixed dropout draw; no hospital or canonical person clusters")


def evaluate(root="results/private/nepal_sensitivity", report_path="results/research_checks/nepal_sensitivity.json"):
    root = Path(root)
    preparation = json.loads((root / "preparation.json").read_text())
    if preparation["protocol_sha256"] != sha256("docs/nepal-analysis.json"):
        raise ValueError("Prepared cohort protocol differs from the recorded analysis")
    report = dict(study_scope="cross_task_sensitivity", clinical_claim_eligible=False,
        independence_verified=False, outcome="ICU mortality", preparation=preparation,
        protocol_sha256=preparation["protocol_sha256"], conditions=[])
    for scenario in SCENARIOS:
        cohort, mapping = root / scenario / "cohort.csv", root / scenario / "mapping.json"
        if sha256(cohort) != preparation["scenarios"][scenario]["cohort_sha256"]:
            raise ValueError("Prepared source cohort changed")
        policy = json.loads(mapping.read_text())
        for imputation in ("median", "knn"):
            rows, contrasts, checks = [], [], []
            ids = labels = None
            for seed in range(5):
                checkpoint = Path(f"results/hospital_study/{imputation}_no_apache/seed_{seed}/wids")
                output = root / scenario / imputation / f"seed_{seed}"
                saved_report = output / "cross_task_sensitivity.json"
                print(f"{scenario} {imputation} seed {seed}", flush=True)
                if not saved_report.exists():
                    if shutil.disk_usage(root).free < 750*1024**2:
                        raise ValueError("Need 750 MB free before another inference bundle")
                    result = evaluate_external(checkpoint, cohort, mapping, output, mc_samples=50, seed=0,
                                               preprocessing_cache=root/"preprocessing_cache")
                else:
                    result = json.loads(saved_report.read_text())
                for path, key in ((checkpoint/"model.pt", "checkpoint_sha256"),
                                  (checkpoint/"preprocessor.joblib", "preprocessor_sha256"), (cohort, "cohort_sha256")):
                    if result[key] != sha256(path):
                        raise ValueError("Saved evaluation source changed")
                if result["mapping"] != policy or result["mc_samples"] != 50:
                    raise ValueError("Saved mapping/pass count differs")
                if result.get("clinical_claim_eligible") is not False or any(k in result for k in ("accepted", "numerical_margins_met")):
                    raise ValueError("Changed-endpoint evaluation cannot claim compatibility acceptance")
                replay = replay_bundle(output / "traces")
                if not replay["passed"]:
                    raise ValueError("Independent Nepal replay failed")
                frame = pd.read_csv(output / "predictions.csv")
                if ids is not None and (not frame.encounter_id.equals(ids) or not np.array_equal(frame.target, labels)):
                    raise ValueError("Seed cohorts differ")
                ids, labels = frame.encounter_id, frame.target.to_numpy()
                metrics = pd.read_csv(output / "metrics.csv")
                residual = 0.
                for row in metrics.to_dict("records"):
                    recomputed = binary_metrics(labels, frame[row["variant"]])
                    residual = max(residual, max(abs(row[k]-v) for k, v in recomputed.items() if np.isfinite(v)))
                    rows.append(dict(seed=seed, **row))
                if residual > 1e-10:
                    raise ValueError("Saved metrics do not reproduce from predictions")
                flat_residual = float(np.max(np.abs(frame.baseline-frame.flat_confidence)))
                if flat_residual > 1e-7:
                    raise ValueError("Uniform gate identity failed")
                contrasts.append((frame.nars_gated.to_numpy(), frame.mc_confidence_only.to_numpy()))
                checks.append(dict(seed=seed, replay=replay, metric_max_residual=residual,
                    uniform_max_residual=flat_residual, checkpoint_sha256=result["checkpoint_sha256"],
                    predictions_sha256=sha256(output/"predictions.csv"), contrast=result))
            table = pd.DataFrame(rows)
            aggregate = []
            for name, group in table.groupby("variant"):
                aggregate.append(dict(variant=name, mean=group.drop(columns=["seed", "variant"]).mean().to_dict(),
                    seed_sd=group.drop(columns=["seed", "variant"]).std(ddof=1).to_dict()))
            # Public evidence includes aggregate results and replay statistics, never rows or IDs.
            for item in checks:
                item["contrast"] = {k:v for k,v in item["contrast"].items()
                    if k not in {"mapping", "independence_verification"}}
            report["conditions"].append(dict(scenario=scenario, imputation=imputation, fits=5,
                metrics=aggregate, nars_minus_mc=hierarchical_brier(contrasts, labels), verification=checks))
            table.to_csv(root/scenario/imputation/"seed_metrics.csv", index=False)
            Path(report_path).write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report["fits"] = sum(c["fits"] for c in report["conditions"])
    Path(report_path).write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/private/nepal_sensitivity")
    parser.add_argument("--report-path", default="results/research_checks/nepal_sensitivity.json")
    args = parser.parse_args()
    report = evaluate(**vars(args))
    print(f"Verified {report['fits']} changed-endpoint sensitivity fits; clinical eligibility=False", flush=True)
