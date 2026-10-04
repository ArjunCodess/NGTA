"""Export physical-record timing coverage for the frozen TCGA feature set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.timing_audit import audit_tcga_timing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/acquired_tcga")
    parser.add_argument("--preprocessor", default="results/tcga_study/fused/seed_0/tcga/preprocessor.joblib")
    parser.add_argument("--output-dir", default="results/research_checks/tcga_timing")
    args = parser.parse_args()
    report, records = audit_tcga_timing(args.data_dir, joblib.load(args.preprocessor))
    report["preprocessor_sha256"] = hashlib.sha256(Path(args.preprocessor).read_bytes()).hexdigest()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    records.to_csv(root / "selected_record_timing.csv", index=False)
    print(json.dumps({"sources": len(report["tables"]), "inputs": len(report["inputs"]), "chronology_verified": False}))
