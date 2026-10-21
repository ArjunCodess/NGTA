"""Explicit first-24-hour mapping for an exploratory cross-source sensitivity."""
from __future__ import annotations

import io
from pathlib import PurePosixPath
import tarfile

import numpy as np
import pandas as pd

SOURCE = "https://archive.physionet.org/challenge/2012/"
SERIES = {
    "d1_heartrate_max": (("HR",), "max", "bpm"),
    "d1_sysbp_min": (("SysABP", "NISysABP"), "min", "mmHg"),
    "d1_temp_max": (("Temp",), "max", "degrees Celsius"),
    "d1_lactate_max": (("Lactate",), "max", "mmol/L"),
    "d1_bun_max": (("BUN",), "max", "mg/dL"),
    "d1_creatinine_max": (("Creatinine",), "max", "mg/dL"),
    "d1_glucose_max": (("Glucose",), "max", "mg/dL"),
    "d1_wbc_max": (("WBC",), "max", "10^9/L = 10^3/uL"),
    "d1_platelets_min": (("Platelets",), "min", "10^9/L = 10^3/uL"),
}


def summarize_record(text: str, expected_id: str):
    rows = pd.read_csv(io.StringIO(text))
    if set(rows.columns) != {"Time", "Parameter", "Value"}:
        raise ValueError("Unexpected PhysioNet record schema")
    time = rows.Time.str.extract(r"^(\d+):([0-5]\d)$")
    if time.isna().any().any():
        raise ValueError("Invalid elapsed ICU timestamp")
    minutes = time[0].astype(int) * 60 + time[1].astype(int)
    values = pd.to_numeric(rows.Value, errors="coerce")
    if not values[rows.Parameter.eq("RecordID")].eq(int(expected_id)).all() or not rows.Parameter.eq("RecordID").any():
        raise ValueError("Archive record ID differs from its filename")
    # Source documents -1 as unknown. Other negative/nonfinite values are invalid.
    rows["value"] = values.where((values >= 0) & np.isfinite(values))
    admission = rows.loc[minutes.eq(0)]
    window = rows.loc[minutes.lt(24 * 60)]

    def descriptor(name):
        observed = admission.loc[admission.Parameter.eq(name), "value"].dropna().unique()
        return float(observed[0]) if len(observed) == 1 else np.nan

    height, weight = descriptor("Height"), descriptor("Weight")
    gender = descriptor("Gender")
    result = dict(record_id=str(expected_id), age=descriptor("Age"),
                  gender={0.0: "F", 1.0: "M"}.get(gender, np.nan),
                  bmi=weight / (height / 100) ** 2 if height > 0 and weight > 0 else np.nan,
                  d1_spo2_min=np.nan, elective_surgery=np.nan,
                  apache_4a_hospital_death_prob=np.nan)
    for target, (parameters, aggregation, _) in SERIES.items():
        series = window.loc[window.Parameter.isin(parameters), "value"]
        result[target] = float(getattr(series, aggregation)()) if series.notna().any() else np.nan
    return result


def read_archive(path):
    records = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            record_id = PurePosixPath(member.name).stem
            if not record_id.isdigit():
                raise ValueError("Unexpected archive record filename")
            # Read in memory, never extract untrusted paths into the filesystem.
            records.append(summarize_record(archive.extractfile(member).read().decode("ascii"), record_id))
    result = pd.DataFrame(records)
    if len(result) != 4000 or result.record_id.duplicated().any():
        raise ValueError("Expected 4,000 unique records per published challenge set")
    return result


def mapping_policy(processor):
    features = {}
    columns = processor.numeric_columns + processor.binary_columns + processor.categorical_columns
    for name in columns:
        unavailable = name in {"d1_spo2_min", "elective_surgery", "apache_4a_hospital_death_prob"}
        unit = SERIES[name][2] if name in SERIES else {"age": "years", "bmi": "kg/m^2", "gender": "F/M"}.get(name, "unavailable")
        evidence = SOURCE + " (source parameter table and elapsed timestamps); WiDS equivalence requires source review"
        if name == "d1_spo2_min":
            evidence += "; arterial SaO2 is not substituted for pulse-oximetry SpO2"
        features[name] = dict(column=None if unavailable else name, source_unit=unit, target_unit=unit,
                              availability="unavailable" if unavailable else "observed",
                              measurement_window="admission descriptors; time series 0 <= elapsed minutes < 1440",
                              evidence=evidence)
    return dict(study_scope="cross_source_sensitivity", clinical_claim_eligible=False,
                independence_verified=False, cohort="PhysioNet/CinC Challenge 2012 sets A/B/C",
                source_version="challenge-2012/1.0.0; archive phase-2 records pinned by SHA256",
                prediction_landmark="24 hours after ICU admission, conditional on original ICU stay >=48 hours",
                outcome_definition="hospital_mortality", development_dataset="wids",
                independence_evidence="Source namespaces differ, but canonical identity linkage and WiDS institution/time provenance are unavailable; independence is unverified",
                id_column="encounter_id", patient_id_column="patient_id", target_column="hospital_death",
                cluster_column="hospital_id", features=features,
                limitations=["One source institution cannot estimate institution-level uncertainty",
                             "Original >=48-hour stay selection excludes early deaths/discharges",
                             "WiDS day-1 versus elapsed 24-hour equivalence is unverified",
                             "SpO2, elective surgery and APACHE probability are unavailable",
                             "Admission BMI and mixed-method blood pressure need clinical harmonization review",
                             "Public challenge data are already inspected; this is not untouched confirmation"])
