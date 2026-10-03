"""Clustered uncertainty intervals from persisted masking predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import paired_bootstrap_indices
from src.robustness import selective_risk


def weighted_uncertainty_statistics(errors, weights, order, starts):
    """Exact duplicated-case statistics, retaining complete uncertainty ties."""
    counts=np.add.reduceat(weights[order],starts)
    positives=np.add.reduceat((errors*weights)[order],starts)
    keep=counts>0
    counts,positives=counts[keep],positives[keep]
    cumulative=np.cumsum(counts)
    area=float(np.sum(np.cumsum(positives)/cumulative*counts/cumulative[-1]))
    negatives=counts-positives
    positive_total,negative_total=positives.sum(),negatives.sum()
    auc=float(np.sum(positives*(np.cumsum(negatives)-.5*negatives))/(positive_total*negative_total)) if positive_total and negative_total else None
    return area,auc


def intervals(root, seeds, iterations=1000):
    for seed in seeds:
        source = Path(root) / f"seed_{seed}" / "wids"
        directory = source / "metrics" / "missingness"
        raw = pd.read_csv(source / "traces" / "raw_test.csv").set_index("encounter_id")
        predictions = pd.read_csv(directory / "missingness_predictions.csv")
        rows = []
        for (scenario, rate, variant), frame in predictions.groupby(["scenario", "mask_rate", "variant"], sort=False):
            y, p, score = frame.target.to_numpy(), frame.probability.to_numpy(), frame.uncertainty.to_numpy()
            groups = raw.loc[frame.case_id, "hospital_id"].to_numpy()
            if not np.array_equal(y, raw.loc[frame.case_id, "hospital_death"]):
                raise ValueError("Persisted masking predictions are not outcome aligned")
            indices = paired_bootstrap_indices(y, iterations, np.random.default_rng(seed), groups)
            errors = (p >= .5) != y
            order=np.argsort(score,kind="stable")
            starts=np.r_[0,np.flatnonzero(np.diff(score[order])!=0)+1]
            areas, aucs = [], []
            for index in indices:
                area,auc=weighted_uncertainty_statistics(errors,np.bincount(index,minlength=len(y)),order,starts)
                areas.append(area)
                if auc is not None:
                    aucs.append(auc)
            point_auc = roc_auc_score(errors, score) if np.unique(errors).size == 2 else None
            rows.append(dict(seed=seed, scenario=scenario, mask_rate=rate, variant=variant,
                selective_risk_area=selective_risk(y,p,score)[1],
                selective_risk_lower_95=np.percentile(areas,2.5), selective_risk_upper_95=np.percentile(areas,97.5),
                error_detection_auroc=point_auc,
                error_detection_lower_95=np.percentile(aucs,2.5) if aucs else None,
                error_detection_upper_95=np.percentile(aucs,97.5) if aucs else None,
                error_detection_bootstrap_replicates=len(aucs), bootstrap_unit="hospital", iterations=iterations))
        pd.DataFrame(rows).to_csv(directory / "uncertainty_intervals.csv", index=False)
        print(f"Saved hospital-clustered uncertainty intervals for seed {seed}", flush=True)
    return {"seeds":seeds,"iterations":iterations,"scope":"per-fit descriptive intervals; no multiplicity-adjusted superiority claim"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--iterations",type=int,default=1000)
    print(json.dumps(intervals(**vars(parser.parse_args())),indent=2))
