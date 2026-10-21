"""Pin, acquire and inventory public TCGA-THCA clinical/biospecimen biotabs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.supplement_audit import acquire_supplements, inventory_supplements, pin_supplements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".cache/ngta-supplements")
    parser.add_argument("--output-dir", default="results/research_checks/tcga_supplements")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else pin_supplements(path)
    acquire_supplements(manifest, args.cache_dir)
    report = inventory_supplements(manifest, args.cache_dir)
    from src.gdc_acquisition import digest_file
    report["manifest_sha256"] = digest_file(path)
    (output / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tables"}, indent=2))
