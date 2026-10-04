"""Evaluate all frozen no-APACHE fits on the explicitly exploratory mapping."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external_validation import evaluate_external
from src.trace_replay import replay_bundle


def evaluate(root="results/cross_source_sensitivity/physionet2012"):
    root = Path(root)
    cohort, mapping = root / "cohort.csv", root / "mapping.json"
    rows, verification = [], []
    for imputation in ("median", "knn"):
        for training_seed in range(5):
            checkpoint = Path(f"results/hospital_study/{imputation}_no_apache/seed_{training_seed}/wids")
            output = root / imputation / f"seed_{training_seed}"
            result = output / "cross_source_sensitivity.json"
            print(f"Evaluating {imputation} training seed {training_seed}", flush=True)
            if result.exists():
                report = json.loads(result.read_text())
                for file, field in ((checkpoint / "model.pt", "checkpoint_sha256"),
                                    (checkpoint / "preprocessor.joblib", "preprocessor_sha256"),
                                    (cohort, "cohort_sha256")):
                    if report[field] != hashlib.sha256(file.read_bytes()).hexdigest():
                        raise ValueError(f"Saved sensitivity source changed: {file}")
                if report["mapping"] != json.loads(mapping.read_text()) or report["mc_samples"] != 50:
                    raise ValueError("Saved sensitivity mapping/pass count changed")
            else:
                report = evaluate_external(checkpoint, cohort, mapping, output, mc_samples=50, seed=0)
            if report.get("clinical_claim_eligible") is not False or "accepted" in report:
                raise ValueError("Exploratory results cannot be certified as clinical validation")
            replay = replay_bundle(output / "traces")
            if not replay["passed"]:
                raise ValueError("Sensitivity predictions failed independent replay")
            verification.append(dict(imputation=imputation, training_seed=training_seed, replay=replay))
            metrics = pd.read_csv(output / "metrics.csv")
            rows.extend(dict(imputation=imputation, training_seed=training_seed, **row) for row in metrics.to_dict("records"))
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "seed_metrics.csv", index=False)
    numeric = frame.select_dtypes("number").columns.drop("training_seed")
    frame.groupby(["imputation", "variant"])[numeric].agg(["mean", "std"]).to_csv(root / "seed_summary.csv")
    summary = dict(fits=len(verification), mc_samples=50, dropout_seed=0,
                   clinical_claim_eligible=False, verified_independent_cohort=False,
                   verification=verification,
                   interpretation="Fixed public single-source sensitivity; seed SD is conditional on the same cohort; case intervals exclude institutional/refitting uncertainty")
    (root / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Verified {len(verification)} frozen fits; clinical claim eligible=False", flush=True)


if __name__ == "__main__":
    evaluate()
