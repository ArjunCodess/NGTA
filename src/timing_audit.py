"""Inspect source chronology without mistaking encounter dates for assay dates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DEFAULT_ID_COLUMN, TABLE_FILES, _collapse_case_table, _read_tcga_table
from .knowledge_base import SYMBOLIC_RULES


def summarize_timing(frame: pd.DataFrame, prefer_primary: bool = False):
    """Use the exact physical-record selection policy of the development loader."""
    selected = _collapse_case_table(frame, prefer_primary=prefer_primary)
    fields = []
    records = []
    for column in frame:
        if "days_to_" not in column and "timepoint_category" not in column and column != "cases.index_date":
            continue
        numeric = pd.to_numeric(selected[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        is_days = "days_to_" in column
        fields.append(dict(field=column, raw_nonmissing=int(frame[column].notna().sum()),
                           selected_nonmissing=int(selected[column].notna().sum()),
                           selected_numeric_days=int(numeric.notna().sum()) if is_days else None,
                           selected_min_days=float(numeric.min()) if is_days and numeric.notna().any() else None,
                           selected_max_days=float(numeric.max()) if is_days and numeric.notna().any() else None,
                           selected_categories=sorted(selected[column].dropna().astype(str).unique().tolist()) if not is_days else None))
        for case, value, day in zip(selected[DEFAULT_ID_COLUMN], selected[column], numeric):
            records.append(dict(case_submitter_id=case, timing_field=column,
                                value=None if pd.isna(value) else str(value),
                                numeric_days=None if not is_days or pd.isna(day) else float(day)))
    return selected, fields, records


def audit_tcga_timing(data_dir, processor):
    root = Path(data_dir)
    tables, records, selected_tables = {}, [], {}
    for name in TABLE_FILES:
        frame = _read_tcga_table(root / name)
        selected, fields, rows = summarize_timing(frame, prefer_primary=name == "clinical.tsv")
        selected_tables[name] = selected
        tables[name] = dict(sha256=hashlib.sha256((root / name).read_bytes()).hexdigest(),
                            raw_rows=len(frame), selected_cases=len(selected), fields=fields)
        records.extend(dict(source_table=name, **row) for row in rows)
    inputs = []
    # These dates refer to their named source entities, not to every field on a row.
    entity_dates = {"diagnoses": ("clinical.tsv", "diagnoses.days_to_diagnosis"),
                    "pathology_details": ("pathology_detail.tsv", "pathology_details.days_to_pathology_detail"),
                    "demographic": ("clinical.tsv", None)}
    encoded_columns = processor.numeric_columns + processor.categorical_columns + processor.binary_columns
    columns = list(dict.fromkeys(encoded_columns + [str(rule["source_column"]) for rule in SYMBOLIC_RULES.values()]))
    for column in columns:
        entity = column.split(".")[0]
        name, date = entity_dates.get(entity, (None, None))
        frame = selected_tables.get(name)
        observed = int(frame[column].notna().sum()) if frame is not None and column in frame else None
        dated = int((frame[column].notna() & pd.to_numeric(frame[date], errors="coerce").notna()).sum()) if frame is not None and date and column in frame else 0
        inputs.append(dict(input=column, role="model_input" if column in encoded_columns else "rule_only", source_table=name, observed_selected_cases=observed,
                           source_entity_date=date, observed_cases_with_entity_date=dated,
                           measurement_time_verified=False,
                           reason="Diagnosis dates are entity-level proxies; pathology, demographic and genomic measurement availability is not established."))
    return dict(schema_version=1, record_policy="loader-selected most-complete physical row per table; clinical primary preference",
                prediction_landmark="retrospective post-pathology association; no numeric pathology landmark is available",
                tables=tables, inputs=inputs, cross_table_chronology_verified=False,
                missing_evidence=["Pathology acquisition/report availability dates relative to each case's diagnosis",
                                  "Genomic specimen collection, assay/report availability and callable case/gene panels",
                                  "Encounter links tying diagnosis, pathology and genomic specimens to the same disease episode"],
                interpretation="Existing dates are audited, not promoted to feature availability or preoperative validation."), pd.DataFrame(records)
