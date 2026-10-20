"""Acquire and inventory every public THCA clinical, biospecimen, SSF and OMF XML."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gdc_acquisition import digest_file
from src.xml_source_audit import acquire_xml, inventory_xml, pin_xml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default='.cache/ngta-xml-supplements')
    parser.add_argument('--output-dir', default='results/research_checks/tcga_xml')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    manifest = pin_xml(manifest_path)
    failures = acquire_xml(manifest, args.cache_dir)
    report = inventory_xml(manifest, args.cache_dir)
    report['unavailable_files'] = failures
    report['manifest_sha256'] = digest_file(manifest_path)
    (output/'inventory.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ('file_count','expected_files','acquisition_complete','case_count','formats','cross_table_chronology_verified','callable_panel_verified')}, indent=2))
