"""Read the released Nepal registry without combining daily rows into new records."""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

SOURCE_SHA256 = "77eb122ec4db0dfdc27cbb8d805b61bb3817670b5d4610a50bb1780d07a8dd06"
SOURCE_DOI = "10.6084/m9.figshare.22263847.v1"
ARTICLE = "https://wellcomeopenresearch.org/articles/8-180"
DICTIONARY = "https://doi.org/10.6084/m9.figshare.15167406.v3"
SCENARIOS = ("demographics", "admission_core")
ADMISSION_COLUMNS = ("ad_age", "ad_sex", "ad_adm_type", "ad_adm_em_surg",
                     "adas_heart_first", "adas_sbp_first", "ad_year_icu", "icudis_status")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_workbook(path, columns=None):
    """Read OOXML values in memory; never execute formulas or extract archive paths."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.itertext()) for t in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        header, records = None, []
        for _, row in ET.iterparse(archive.open("xl/worksheets/sheet1.xml"), events=["end"]):
            if row.tag != "{" + NS["s"] + "}row":
                continue
            values = {}
            for cell in row:
                key = "".join(c for c in cell.attrib["r"] if c.isalpha())
                if header is not None and columns is not None and header.get(key) not in columns:
                    continue
                if cell.find("s:f", NS) is not None:
                    raise ValueError("Source measurements must be values, not formulas")
                value = cell.find("s:v", NS)
                text = value.text if value is not None else "".join(cell.itertext())
                values[key] = strings[int(text)] if cell.attrib.get("t") == "s" else text
            if header is None:
                header = values
                if len(set(header.values())) != len(header):
                    raise ValueError("Duplicate workbook headers")
            elif values:
                records.append({header[k]: v for k, v in values.items() if k in header})
            row.clear()
    return pd.DataFrame(records)


def select_admissions(frame):
    required = ["patient_number", *ADMISSION_COLUMNS]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing registry admission columns")
    raw = frame[required].replace({"NA": np.nan, "": np.nan}).copy()
    if raw.patient_number.isna().any():
        raise ValueError("Missing source admission identifiers")
    # Including missingness in this check prevents a synthetic first-non-null record.
    conflicts = raw.groupby("patient_number")[list(ADMISSION_COLUMNS)].nunique(dropna=False).gt(1)
    bad = conflicts.any(axis=1)
    unique = raw.drop_duplicates("patient_number", keep="first")
    consistent = unique.loc[~unique.patient_number.isin(bad.index[bad])].copy()
    age = pd.to_numeric(consistent.ad_age, errors="coerce")
    adult = age.between(18, 107) & np.isfinite(age)
    labels = consistent.icudis_status.map({"Alive": 0, "Dead": 1})
    eligible = consistent.loc[adult & labels.notna()].copy()
    eligible["icu_death"] = labels.loc[eligible.index].astype(int)
    eligible["age"] = age.loc[eligible.index]
    eligible["gender"] = eligible.ad_sex.map({"Male": "M", "Female": "F"})
    # Emergency status is relevant only for documented postoperative admissions.
    eligible["elective_surgery"] = np.where(eligible.ad_adm_type.eq("Non operative"), 0.,
        np.where(eligible.ad_adm_type.eq("Post operative"),
                 eligible.ad_adm_em_surg.map({"Yes": 0., "No": 1.}), np.nan))
    invalid = {}
    for source, target, upper in (("adas_heart_first", "d1_heartrate_max", 300),
                                   ("adas_sbp_first", "d1_sysbp_min", 350)):
        values = pd.to_numeric(eligible[source], errors="coerce")
        valid = values.between(1, upper) & np.isfinite(values)
        invalid[source] = int((values.notna() & ~valid).sum())
        eligible[target] = values.where(valid)
    eligible["encounter_id"] = eligible.patient_number.map(lambda v: "nicrf:" + hashlib.sha256(str(v).encode()).hexdigest())
    eligible["source_identifier"] = eligible.patient_number.astype(str)
    # This is an admission identifier, not verified canonical repeat-person linkage.
    eligible["admission_id"] = eligible.encounter_id
    eligible["source_registry"] = "NICRF; hospital identifiers unavailable"
    eligible = eligible.sort_values("encounter_id").reset_index(drop=True)
    report = dict(source_rows=len(frame), source_admissions=len(unique),
        conflicting_admissions=int(bad.sum()), conflicting_fields=conflicts.sum().astype(int).to_dict(),
        missing_invalid_or_nonadult_age=int((~adult).sum()),
        missing_or_unknown_outcome_in_adults=int((adult & labels.isna()).sum()),
        eligible_admissions=len(eligible), deaths=int(eligible.icu_death.sum()),
        invalid_physiology_masked=invalid,
        outcome="ICU death, not hospital death", patient_independence_verified=False)
    return eligible, report


def mapping_policy(processor, scenario, protocol_sha256):
    if scenario not in SCENARIOS or "apache_4a_hospital_death_prob" in processor.numeric_columns:
        raise ValueError("Use a declared scenario and a frozen no-APACHE processor")
    available = {"age": "years", "gender": "F/M", "elective_surgery": "binary"}
    if scenario == "admission_core":
        available.update(d1_heartrate_max="bpm", d1_sysbp_min="mmHg")
    features = {}
    for name in processor.numeric_columns + processor.binary_columns + processor.categorical_columns:
        observed = name in available
        features[name] = dict(column=name if observed else None,
            source_unit=available.get(name, "unavailable"), target_unit=available.get(name, "unavailable"),
            availability="observed" if observed else "unavailable",
            measurement_window="Admission first measurement, not day-one extrema; core collection within first 24 hours",
            evidence=f"{ARTICLE}; {DICTIONARY}; {name} proxy is exploratory, not verified WiDS equivalence")
    return dict(study_scope="cross_task_sensitivity", clinical_claim_eligible=False,
        independence_verified=False, protocol_sha256=protocol_sha256, scenario=scenario,
        cohort="Nepal NICRF adult admissions", source_version=SOURCE_DOI,
        prediction_landmark="First-24-hour admission core data; exact elapsed landmark unavailable",
        outcome_definition="icu_mortality", development_outcome_definition="hospital_mortality",
        development_dataset="wids", label_shift="ICU death substitutes a different endpoint solely for exploratory sensitivity",
        independence_evidence="Admission IDs are not canonical persons; source overlap and hospital clusters unverified",
        id_column="encounter_id", patient_id_column="admission_id", target_column="icu_death",
        cluster_column="source_registry", features=features,
        limitations=["Changed ICU versus hospital mortality endpoint",
            "First measurements are not day-one extrema", "No hospital identifiers or canonical person linkage",
            "Temperature, creatinine and WBC excluded for unresolved unit heterogeneity",
            "BMI, lactate, BUN, glucose, SpO2, platelets unavailable", "No retuning or recalibration"])
