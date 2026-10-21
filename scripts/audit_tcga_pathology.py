"""Acquire pinned public pathology PDFs and inventory text without certifying timing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import requests


def verify(record, content):
    if (record.get("access") != "open" or record.get("data_format") != "PDF"
            or not re.fullmatch(r"[0-9a-f-]{36}", record["file_id"])):
        raise ValueError("Unsupported pathology source metadata")
    if (len(content) != record["file_size"] or hashlib.md5(content).hexdigest() != record["md5sum"]
            or not content.startswith(b"%PDF-")):
        raise ValueError("Pathology publisher size, checksum or PDF signature differs")


def text_inventory(text):
    # These are text candidates only. Neither dates nor molecular keywords prove
    # linkage to the exact selected record, negative callability or availability.
    dates = re.findall(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)?\d{2}|(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    return dict(characters=len(text.strip()), numeric_date_candidates=len(dates),
        redaction_mentions=len(re.findall(r"\bredact(?:ed|ion)?\b|\bX{2,}[/.-]X", text, re.I)),
        report_date_mentions=len(re.findall(r"(?:report|sign(?:ed|out)?|collect(?:ion|ed)?|receiv(?:ed|ing)?|procedur\w*)\s*date|date\s*(?:of|:)" , text, re.I)),
        molecular_test_mentions=len(re.findall(r"\b(?:BRAF|molecular|sequenc\w*|mutation\w*)\b", text, re.I)))


def audit(cache=".cache/ngta-pathology", output="results/research_checks/tcga_pathology.json"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = Path("results/research_checks/source_followup.json")
    records = json.loads(manifest.read_text())["pathology_reports"]["files"]
    if len({r["file_id"] for r in records}) != len(records):
        raise ValueError("Duplicate pathology UUIDs")
    if shutil.disk_usage(cache).free < sum(r["file_size"] for r in records)+300*1024**2:
        raise ValueError("Insufficient pathology download/extraction space")
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError("Install Poppler pdftotext for the native PDF audit")

    def inspect(record):
        path = cache/(record["file_id"]+".pdf")
        row = dict(file_id=record["file_id"], cases=[c["submitter_id"] for c in record["cases"]])
        try:
            if not path.exists():
                response = requests.get("https://api.gdc.cancer.gov/data/"+record["file_id"], timeout=(20, 60))
                response.raise_for_status()
                verify(record, response.content)
                temporary = path.with_suffix(".part")
                temporary.write_bytes(response.content)
                temporary.replace(path)
            body = path.read_bytes()
            verify(record, body)
            extracted = subprocess.run([executable, "-layout", str(path), "-"], capture_output=True, timeout=45)
            if extracted.returncode:
                raise ValueError("pdftotext failed: "+extracted.stderr.decode(errors="replace")[:150])
            text = extracted.stdout.decode("utf-8", errors="replace")
            (cache/(record["file_id"]+".txt")).write_text(text, encoding="utf-8")
            row.update(acquired=True, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                       text=text_inventory(text), extraction="native pdftotext; OCR accuracy not certified")
        except (requests.RequestException, ValueError, subprocess.TimeoutExpired) as error:
            row.update(acquired=path.exists(), error=str(error))
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(inspect, records), 1):
            rows.append(row)
            if i%50 == 0 or i == len(records):
                print(f"Pathology {i}/{len(records)}; errors={sum('error' in r for r in rows)}", flush=True)
    complete = [r for r in rows if "text" in r]
    report = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), expected_files=len(records),
        acquired_files=sum(r["acquired"] for r in rows), extracted_files=len(complete),
        native_text_files=sum(r["text"]["characters"] > 0 for r in complete),
        empty_text_files=[r["file_id"] for r in complete if not r["text"]["characters"]],
        cases=len({c for r in complete for c in r["cases"]}),
        files_with_date_candidates=sum(r["text"]["numeric_date_candidates"] > 0 for r in complete),
        files_with_report_date_mentions=sum(r["text"]["report_date_mentions"] > 0 for r in complete),
        files_with_molecular_mentions=sum(r["text"]["molecular_test_mentions"] > 0 for r in complete),
        extraction_complete=len(complete) == len(records),
        cross_table_chronology_verified=False, callable_panel_verified=False,
        interpretation="Native text inventory only; date patterns and molecular mentions do not establish selected-input episode linkage, clinical availability or gene callability. Empty or partial scanned pages require OCR/manual source review. Raw PDFs/text stay in ignored storage.",
        files=rows)
    Path(output).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/ngta-pathology")
    parser.add_argument("--output", default="results/research_checks/tcga_pathology.json")
    report = audit(**vars(parser.parse_args()))
    print(json.dumps({k:v for k,v in report.items() if k != "files"}, indent=2))
