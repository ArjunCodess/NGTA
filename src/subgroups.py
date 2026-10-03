"""Fixed demographic strata and decision thresholds, without subgroup tuning."""
import numpy as np
import pandas as pd

from .evaluation import binary_metrics


def subgroup_reports(frame, labels, probability_map):
    groups = {}
    if "age" in frame:
        age = pd.to_numeric(frame.age, errors="coerce")
    elif "diagnoses.age_at_diagnosis" in frame:
        age = pd.to_numeric(frame["diagnoses.age_at_diagnosis"], errors="coerce") / 365.25
    else:
        age = None
    if age is not None:
        groups["age"] = pd.Series(np.select([age.isna(), age < 65], ["unavailable", "under_65"], default="65_or_older"), index=frame.index)
    for column in ("gender", "demographic.gender", "demographic.race"):
        if column in frame:
            groups[column] = frame[column].fillna("unavailable").astype(str)
    if "apache_4a_hospital_death_prob" in frame:
        groups["apache_availability"] = frame.apache_4a_hospital_death_prob.notna().map({True: "available", False: "unavailable"})
    rows, curves = [], []
    labels = np.asarray(labels, dtype=int)
    for group_name, values in groups.items():
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            y = labels[mask]
            for variant, probabilities in probability_map.items():
                p = np.asarray(probabilities)[mask]
                rows.append({"group": group_name, "stratum": value, "variant": variant,
                             "cases": len(y), "positives": int(y.sum()), **binary_metrics(y, p)})
                for threshold in (.05, .1, .2, .3, .5):
                    selected = p >= threshold
                    benefit = (np.sum(selected & (y == 1)) - np.sum(selected & (y == 0))*threshold/(1-threshold)) / len(y)
                    curves.append({"group": group_name, "stratum": value, "variant": variant, "threshold": threshold,
                                   "net_benefit": benefit, "treat_all": y.mean()-(1-y.mean())*threshold/(1-threshold), "treat_none": 0})
    return pd.DataFrame(rows), pd.DataFrame(curves)
