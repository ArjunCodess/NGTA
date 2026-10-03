"""Pinned open-access GDC acquisition with explicit case coverage."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import requests


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_thca_manifest(path, expected_cases=None):
    path = Path(path)
    if path.exists():
        raise ValueError("Acquisition manifest already exists; reuse its pinned IDs or choose a new version directory")
    filters = {"op": "and", "content": [{"op": "in", "content": {"field": field, "value": [value]}}
        for field, value in (("cases.project.project_id", "TCGA-THCA"), ("data_type", "Masked Somatic Mutation"),
                             ("data_format", "MAF"), ("access", "open"))]}
    params = dict(filters=json.dumps(filters), fields="file_id,file_name,md5sum,file_size,access,cases.submitter_id,analysis.workflow_type,created_datetime,updated_datetime", size=500, sort="file_id:asc")
    hits, offset, total = [], 0, None
    while total is None or offset < total:
        response = requests.get("https://api.gdc.cancer.gov/files", params={**params, "from": offset}, timeout=60)
        response.raise_for_status()
        page = response.json()["data"]
        total = page["pagination"]["total"]
        if not page["hits"]:
            raise ValueError("GDC pagination ended before the reported total")
        hits.extend(page["hits"])
        offset += len(page["hits"])
    workflows = sorted({hit.get("analysis", {}).get("workflow_type", "") for hit in hits})
    selected_workflow = next((w for w in workflows if "ensemble" in w.lower()), None)
    if selected_workflow is None:
        raise ValueError(f"No supported ensemble workflow; review available workflows explicitly: {workflows}")
    selected = [hit for hit in hits if hit.get("analysis", {}).get("workflow_type") == selected_workflow]
    if len({hit["file_id"] for hit in selected}) != len(selected):
        raise ValueError("GDC pagination returned duplicate IDs; pin a fresh consistent snapshot")
    files = []
    for hit in selected:
        name = hit["file_name"]
        if Path(name).name != name or not name.endswith((".maf", ".maf.gz")):
            raise ValueError("GDC returned an unsafe or unsupported file name")
        if not hit.get("md5sum") or hit.get("access") != "open":
            raise ValueError("Acquisition requires publisher checksum and open access")
        files.append(dict(file_id=hit["file_id"], file_name=name, bytes=hit["file_size"], md5=hit["md5sum"],
                          case_ids=sorted(case["submitter_id"] for case in hit.get("cases", [])),
                          created=hit.get("created_datetime"), updated=hit.get("updated_datetime")))
    coverage = sorted({case for record in files for case in record["case_ids"]})
    expected = sorted(set(map(str, expected_cases if expected_cases is not None else [])))
    missing = sorted(set(expected)-set(coverage))
    manifest = dict(schema_version=1, endpoint="https://api.gdc.cancer.gov/files", project="TCGA-THCA",
                    retrieved_utc=datetime.now(timezone.utc).isoformat(), workflow=selected_workflow,
                    query=params, files=files, case_ids=coverage, expected_case_ids=expected,
                    unavailable_expected_cases=missing, complete_expected_file_coverage=not missing if expected else None,
                    assay_interpretation="file-associated case coverage is not verified gene callability or a negative mutation call")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_pinned_manifest(manifest_path, output_dir, workers=4):
    manifest_path, root = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    def acquire(record):
        name = record["file_name"]
        if Path(name).name != name:
            raise ValueError("Unsafe file name in manifest")
        compressed = root / name
        if not compressed.exists():
            temporary = compressed.with_suffix(compressed.suffix + ".part")
            with requests.get("https://api.gdc.cancer.gov/data/"+record["file_id"], stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024*1024):
                        stream.write(block)
            if temporary.stat().st_size != record["bytes"] or digest_file(temporary, "md5") != record["md5"]:
                raise ValueError(f"Publisher checksum/size mismatch for {name}")
            temporary.replace(compressed)
        if compressed.stat().st_size != record["bytes"] or digest_file(compressed, "md5") != record["md5"]:
            raise ValueError(f"Pinned file has changed: {name}")
        maf = compressed.with_suffix("") if name.endswith(".gz") else compressed
        if name.endswith(".gz"):
            with gzip.open(compressed, "rb") as source, maf.with_suffix(".maf.tmp").open("wb") as destination:
                shutil.copyfileobj(source, destination)
            maf.with_suffix(".maf.tmp").replace(maf)
        frame = pd.read_csv(maf, sep="\t", comment="#", low_memory=False)
        cases = sorted(set(frame.Tumor_Sample_Barcode.astype(str).str.slice(0, 12)))
        if set(cases)-set(record["case_ids"]):
            raise ValueError(f"MAF cases disagree with pinned publisher metadata: {name}")
        return dict(file_id=record["file_id"], file_name=maf.name, sha256=digest_file(maf),
                    download_sha256=digest_file(compressed), variant_rows=len(frame), variant_case_ids=cases)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(acquire, manifest["files"]))
    report = dict(manifest_sha256=digest_file(manifest_path), files=downloaded,
                  variant_case_ids=sorted({case for row in downloaded for case in row["variant_case_ids"]}),
                  file_coverage_case_ids=manifest["case_ids"],
                  unavailable_expected_cases=manifest.get("unavailable_expected_cases", []),
                  complete_expected_file_coverage=manifest.get("complete_expected_file_coverage"),
                  verified_negative_calls=False, note="Variant-negative and empty MAF cases still require callable assay evidence")
    (root / "download_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify_pinned_sources(directory):
    root = Path(directory)
    manifest_path = root / "acquisition_manifest.json"
    if not manifest_path.exists():
        return None
    report = json.loads((root / "download_verification.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    if report["manifest_sha256"] != digest_file(manifest_path):
        raise ValueError("Pinned acquisition manifest has changed")
    if {row["file_id"] for row in report["files"]} != {row["file_id"] for row in manifest["files"]}:
        raise ValueError("Acquisition verification does not include every pinned file")
    if {path.name for path in root.glob("*.maf")} != {row["file_name"] for row in report["files"]}:
        raise ValueError("MAF source directory contains missing or unpinned files")
    for record in report["files"]:
        name = record["file_name"]
        if Path(name).name != name or digest_file(root / name) != record["sha256"]:
            raise ValueError("Acquired MAF differs from verified source bytes")
    return report
