import argparse
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gdc_acquisition import pin_thca_manifest, download_pinned_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pin and verify public TCGA-THCA MAFs by UUID and checksum")
    parser.add_argument("--output-dir", default="data/acquired_tcga")
    parser.add_argument("--expected-cases-csv")
    parser.add_argument("--id-column", default="case_submitter_id")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)
    manifest = root / "acquisition_manifest.json"
    if not manifest.exists():
        expected = pd.read_csv(args.expected_cases_csv)[args.id_column].astype(str).tolist() if args.expected_cases_csv else None
        pin_thca_manifest(manifest, expected)
    report = json.loads(manifest.read_text()) if args.metadata_only else download_pinned_manifest(manifest, root)
    print(json.dumps(dict(files=len(report["files"]), variant_cases=len(report.get("variant_case_ids", [])),
                         missing_expected_cases=report.get("unavailable_expected_cases", []))))
