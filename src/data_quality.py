"""Persist cohort lineage and measured validity, without inferring assay negatives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def source_manifest(data_dir: str | Path, dataset: str) -> list[dict]:
    root = Path(data_dir)
    paths = [root / "wids_icu.csv"] if dataset == "wids" else sorted(
        p for p in root.iterdir() if p.suffix in {".tsv", ".maf", ".csv"} and p.name != "wids_icu.csv"
    )
    records = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return records


def export_data_quality(bundle, directory: str | Path, data_dir: str | Path, dataset: str) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    id_column = bundle.preprocessor.id_column
    splits = {name: getattr(bundle, f"{name}_frame") for name in ("train", "val", "test")}
    split_ids = pd.concat([frame[[id_column]].assign(split=name) for name, frame in splits.items()])
    if split_ids[id_column].isna().any() or split_ids[id_column].duplicated().any():
        raise ValueError("Case IDs must be present and disjoint across partitions")
    split_ids.to_csv(root / "split_ids.csv", index=False)
    columns = bundle.preprocessor.numeric_columns + bundle.preprocessor.binary_columns + bundle.preprocessor.categorical_columns
    missingness = pd.DataFrame([
        {"split": name, "feature": column, "rows": len(frame),
         "missing_count": int(frame[column].isna().sum()), "missing_fraction": float(frame[column].isna().mean())}
        for name, frame in {"all": bundle.labeled_frame, **splits}.items() for column in columns
    ])
    missingness.to_csv(root / "missingness.csv", index=False)
    overlap = {}
    for group in ("patient_id", "hospital_id", "icu_id"):
        if group in bundle.labeled_frame:
            overlap[group] = {
                f"{left}_{right}": len(set(splits[left][group].dropna()) & set(splits[right][group].dropna()))
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            }
    report = {"schema_version": 2, "dataset": dataset,
              "sources": source_manifest(data_dir, dataset), "group_overlap": overlap,
              "duplicate_feature_rows": int(bundle.labeled_frame[columns].duplicated().sum()),
              "overall_missing_fraction": float(bundle.labeled_frame[columns].isna().to_numpy().mean()),
              "prediction_landmark": bundle.split_summary.get("prediction_landmark"),
              "genomic_coverage": bundle.split_summary.get("genomic_coverage"),
              "clinical_timing_verified": False}
    if dataset == "wids":
        raw = pd.read_csv(Path(data_dir) / "wids_icu.csv", usecols=["apache_4a_hospital_death_prob"])
        score = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        report["apache_invalid_probability_count"] = int((score.notna() & ~score.between(0, 1)).sum())
        report["apache_policy"] = "values outside [0,1] are unavailable; source sentinel semantics unverified"
        encoded = bundle.encoded_test
        report["imputed_only_rule_triggers_suppressed"] = dict(zip(
            bundle.preprocessor.rule_names,
            (encoded.imputed_rule_triggers - encoded.rule_triggers).sum(axis=0).astype(int).tolist(),
        ))
    else:
        # Audit every repeated source field before selecting one physical record.
        from .data_loader import TABLE_FILES, _read_tcga_table
        conflicts = []
        for filename in TABLE_FILES:
            table = _read_tcga_table(Path(data_dir) / filename)
            counts = table.groupby(id_column).nunique(dropna=True)
            for case_id, row in counts.iterrows():
                for column in row.index[row > 1]:
                    conflicts.append({"source": filename, "case_id": case_id, "field": column,
                                      "distinct_values": int(row[column])})
        pd.DataFrame(conflicts, columns=["source", "case_id", "field", "distinct_values"]).to_csv(root / "record_conflicts.csv", index=False)
        report["conflicting_case_fields"] = len(conflicts)
    (root / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
