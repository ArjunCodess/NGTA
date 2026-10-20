"""Acquire pinned public GDC biotabs and inventory evidence without imputing chronology."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil

import requests

from .gdc_acquisition import digest_file

API = "https://api.gdc.cancer.gov"


def pin_supplements(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Manifest already exists; reuse the pinned snapshot")
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": field, "value": values}}
        for field, values in (("cases.project.project_id", ["TCGA-THCA"]),
                              ("data_type", ["Clinical Supplement", "Biospecimen Supplement"]),
                              ("data_format", ["BCR Biotab"]), ("access", ["open"]))]}
    params = dict(filters=json.dumps(filters), fields="file_id,file_name,file_size,md5sum,data_type,data_format,access,cases.submitter_id",
                  size=100, sort="file_id:asc")
    files, offset, total = [], 0, None
    while total is None or offset < total:
        response = requests.get(API + "/files", params={**params, "from": offset}, timeout=60)
        response.raise_for_status()
        data = response.json()["data"]
        total = data["pagination"]["total"]
        if not data["hits"]:
            raise ValueError("Incomplete supplement pagination")
        files.extend(data["hits"])
        offset += len(data["hits"])
    if not files or len({item["file_id"] for item in files}) != len(files):
        raise ValueError("Empty or duplicate supplement snapshot")
    manifest = dict(project="TCGA-THCA", retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    endpoint=API + "/files", query=params, files=files)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def acquire_supplements(manifest, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pending_bytes = sum(item["file_size"] for item in manifest["files"])
    if shutil.disk_usage(root).free < pending_bytes + 100 * 1024**2:
        raise ValueError("Insufficient disk space for supplement download and reserve")
    for item in manifest["files"]:
        name = item["file_name"]
        if (Path(name).name != name or "/" in name or "\\" in name
                or not re.fullmatch(r"[\w.-]+\.txt", name)
                or item.get("access") != "open" or not item.get("md5sum")
                or item.get("data_format") != "BCR Biotab"):
            raise ValueError("Unsafe or unsupported supplement metadata")
        path = root / name
        if not path.exists():
            response = requests.get(API + "/data/" + item["file_id"], timeout=90)
            response.raise_for_status()
            content = response.content
            if len(content) != item["file_size"] or hashlib.md5(content).hexdigest() != item["md5sum"]:
                raise ValueError("Supplement publisher checksum/size mismatch")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(path)
        if path.stat().st_size != item["file_size"] or digest_file(path, "md5") != item["md5sum"]:
            raise ValueError("Pinned supplement bytes changed")


def read_biotab(path):
    """Preserve duplicate columns and skip only declared metadata rows."""
    rows = list(csv.reader(io.StringIO(Path(path).read_text(encoding="utf-8-sig")), delimiter="\t"))
    if not rows:
        raise ValueError("Empty biotab")
    header = rows[0]
    aliases = None
    records = []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError("Biotab row width differs from header")
        if not any(row) or any(value.startswith("CDE_ID:") for value in row):
            continue
        if row[0] == header[0]:
            if aliases is not None:
                raise ValueError("Multiple biotab alias rows")
            aliases = row
            continue
        records.append(row)
    return header, aliases or header, records


def present(value):
    return any(bool(part.strip()) and not part.strip().startswith("[")
               and part.strip().lower() not in {"na", "nan", "null"}
               for part in value.split("|"))


def link_specimen_records(samples, tumor_records):
    """Check explicit sample UUID links; never select or synthesize a clinical row."""
    by_uuid = {}
    for row in samples:
        key = row["bcr_sample_uuid"].lower()
        if not present(key) or key in by_uuid:
            raise ValueError("Missing or duplicate biospecimen sample UUID")
        by_uuid[key] = row
    matched = 0
    cases = set()
    before_procurement = 0
    paired_dates = 0
    for row in tumor_records:
        sample = by_uuid.get(row["bcr_sample_uuid"].lower())
        if sample is None:
            continue
        case = row["bcr_patient_barcode"]
        if case != sample["bcr_sample_barcode"][:12]:
            raise ValueError("Specimen UUID links to a different patient barcode")
        matched += 1
        cases.add(case)
        try:
            review = float(row["tss_pathology_review_days_to"])
            procurement = float(row["tumor_sample_procurement_days_to"])
        except ValueError:
            continue
        if math.isfinite(review) and math.isfinite(procurement):
            paired_dates += 1
            before_procurement += int(review < procurement)
    return dict(tumor_records=len(tumor_records), matched_sample_uuid_records=matched,
                matched_cases=len(cases), records_with_both_offsets=paired_dates,
                review_before_procurement_records=before_procurement,
                clinical_feature_availability_verified=False,
                interpretation="UUID linkage and source offsets only; tissue review is not the availability date of the selected model pathology or genomic reports.")


def inventory_supplements(manifest, directory):
    tables = []
    specimen_sources = {}
    for item in manifest["files"]:
        path = Path(directory) / item["file_name"]
        # Inventory never trusts an existing local file merely because it has the right name.
        if path.stat().st_size != item["file_size"] or digest_file(path, "md5") != item["md5sum"]:
            raise ValueError("Pinned supplement bytes changed")
        headers, aliases, rows = read_biotab(path)
        for suffix in ("biospecimen_sample_thca.txt", "ssf_tumor_samples_thca.txt"):
            if item["file_name"].endswith(suffix):
                if suffix in specimen_sources:
                    raise ValueError("Ambiguous specimen source tables")
                specimen_sources[suffix] = [dict(zip(headers, row)) for row in rows]
        fields = []
        for index, (name, alias) in enumerate(zip(headers, aliases)):
            values = [row[index] for row in rows if present(row[index])]
            relevant = re.search(r"date|days_to|days_to_|_days|collection|procure|shipment|genotyp|mutation|report|pathologic|barcode|uuid", name + " " + alias)
            if not relevant:
                continue
            entry = dict(column_index=index, field=name, dictionary_alias=alias, observed_rows=len(values))
            if "days_to" in name or "days_to" in alias or "_days" in name:
                numbers = []
                for value in values:
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if math.isfinite(number):
                        numbers.append(number)
                entry.update(finite_numeric_rows=len(numbers), minimum=min(numbers, default=None), maximum=max(numbers, default=None))
            if "genotyp" in name + alias or "mutation" in name + alias:
                entry["reported_values"] = dict(sorted(Counter(row[index] for row in rows).items()))
            fields.append(entry)
        tables.append(dict(file_id=item["file_id"], file_name=item["file_name"], sha256=digest_file(path),
                           bytes=path.stat().st_size, records=len(rows), columns=len(headers), fields=fields))
    linkage = link_specimen_records(specimen_sources["biospecimen_sample_thca.txt"],
                                   specimen_sources["ssf_tumor_samples_thca.txt"]) if len(specimen_sources) == 2 else None
    return dict(schema_version=1, files=len(tables), tables=tables, specimen_linkage=linkage, cross_table_chronology_verified=False,
                callable_panel_verified=False, models_refitted=False,
                interpretation="Source inventory only. Form completion, processing and shipment dates are not clinical result availability. Reported genotyping outcomes do not certify the frozen WXS panel's callability. No source rows are merged into frozen model inputs.",
                missing_evidence=["Source-reviewed linkage of pathology and genomic report availability to the prediction episode",
                                  "Case/gene assay quality and callability for the frozen mutation panel"])
