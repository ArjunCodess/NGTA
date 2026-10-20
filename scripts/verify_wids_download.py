"""Check the partial WiDS download without modifying any source file."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
DATA_SIZES = {
    'WiDS_Datathon_2020_Dictionary.csv': 29944,
    'samplesubmission.csv': 52,
    'solution_template.csv': 320363,
    'training_v2.csv': 64747327,
    'unlabeled.csv': 27733653,
}
DICTIONARY_SHA256 = '8c9b3b166cb519528687060edfc8b2a7f07b0bc7da18f926e6c52be1402618eb'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def audit(root, study_data, recipe):
    root = Path(root).resolve()
    checks = []
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line)
        if not match:
            raise ValueError('Malformed publisher checksum line')
        checksum, name = match.groups()
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or not path.is_relative_to(root):
            raise ValueError('Unsafe publisher checksum path')
        actual = digest(path) if path.is_file() else None
        checks.append(dict(file=name, expected_sha256=checksum.lower(), actual_sha256=actual,
                           bytes=path.stat().st_size if path.is_file() else None,
                           passed=actual == checksum.lower()))
    if not checks:
        raise ValueError('Publisher checksum manifest is empty')
    sizes = []
    for name, expected in DATA_SIZES.items():
        path = root / 'data' / name
        actual = path.stat().st_size if path.is_file() else None
        sizes.append(dict(file='data/'+name, expected_bytes=expected, actual_bytes=actual, passed=actual == expected))
    dictionary = root / 'data' / 'WiDS_Datathon_2020_Dictionary.csv'
    source_records = [source for fit in recipe['fits'] if fit['dataset'] == 'wids' for source in fit['sources']]
    expected_sources = {(source['bytes'], source['sha256']) for source in source_records if source['file'] == 'wids_icu.csv'}
    if len(expected_sources) != 1:
        raise ValueError('Frozen recipe must identify one consistent WiDS source')
    study_data = Path(study_data)
    study_bytes = study_data.stat().st_size if study_data.is_file() else None
    study_hash = digest(study_data) if study_data.is_file() else None
    return dict(files=checks, data_file_sizes=sizes,
                publisher_checksum_entries_pass=all(row['passed'] for row in checks),
                listed_data_sizes_match=all(row['passed'] for row in sizes),
                dictionary_matches_inspected_copy=dictionary.is_file() and digest(dictionary) == DICTIONARY_SHA256,
                original_study_source=dict(bytes=study_bytes, sha256=study_hash,
                    matches_frozen_recipe=(study_bytes, study_hash) in expected_sources),
                whole_download_verified=False,
                size_provenance='WiDS 1.0.0 publisher data-directory listing saved with the authorized download; byte counts do not establish file integrity.',
                interpretation='Publisher checksums cover only their listed files. Other size checks detect truncation but cannot certify full-release integrity. The existing fitted-study source is checked separately against the frozen recipe. No files were modified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/private/physionet.org/files/widsdatathon2020/1.0.0')
    parser.add_argument('--study-data', type=Path, default=ROOT / 'data/wids_icu.csv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.study_data, json.loads((ROOT / 'study_recipe.json').read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['publisher_checksum_entries_pass'] and report['listed_data_sizes_match'] and report['dictionary_matches_inspected_copy'] and report['original_study_source']['matches_frozen_recipe'] else 1)
