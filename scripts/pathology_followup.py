"""Prepare a private molecular-report queue without deriving gene-negative calls."""
import hashlib
import json
from pathlib import Path
import re


def scope_flags(text):
    return dict(
        hotspot_or_codon_mentions=bool(re.search(r"\bcodon\w*\b|\bhotspot\b|\bV600E\b", text, re.I)),
        pending_test_mentions=bool(re.search(r"review pending|to be separately reported|results will be reported|studies (?:are|will be) ordered", text, re.I)),
        indeterminate_mentions=bool(re.search(r"\bindeterminate\b|poor quality", text, re.I)),
        prior_specimen_mentions=bool(re.search(r"fine needle|fine-needle|previously performed|prior molecular", text, re.I)))


def prepare():
    inventory = Path("results/research_checks/tcga_pathology.json")
    audit = json.loads(inventory.read_text())
    if audit["extracted_files"] != 515:
        raise ValueError("Complete the pinned pathology native-text inventory first")
    queue = []
    for row in audit["files"]:
        if not row["text"]["molecular_test_mentions"]:
            continue
        path = Path(".cache/ngta-pathology")/(row["file_id"]+".txt")
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        selected = set()
        for i, line in enumerate(lines):
            if re.search(r"\bBRAF\b|mutation|molecular", line, re.I):
                selected.update(range(max(0, i-2), min(len(lines), i+4)))
        queue.append(dict(file_id=row["file_id"], cases=row["cases"],
            text_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            flags=scope_flags(text), lines=[dict(line=i+1, text=lines[i]) for i in sorted(selected)]))
    private = Path(".cache/ngta-pathology/molecular_review_queue.json")
    private.write_text(json.dumps(queue, indent=2)+"\n", encoding="utf-8")
    report = dict(inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        queued_reports=len(queue), private_queue_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
        candidate_flags={name: sum(row["flags"][name] for row in queue) for name in scope_flags("")},
        verified_whole_gene_negative_calls_derived=0, assay_manifest_modified=False,
        interpretation="Keyword scope flags are a review queue, not specimen-linked assay results. Hotspot-only, pending, indeterminate and prior-specimen text cannot establish full selected-panel negative callability. Native extraction and literature mentions can be misleading. No mutation status or availability date is inferred.")
    Path("results/research_checks/tcga_pathology_molecular.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
