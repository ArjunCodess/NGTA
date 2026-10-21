"""Build or verify a source-only NGTA reproduction package. No datasets or fitted arrays."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = ('main.py', 'requirements.txt', 'requirements-lock.windows-py314.txt', 'study_recipe.json',
       'README.md')
SUFFIXES = {'src': {'.py'}, 'scripts': {'.py', '.json'}, 'tests': {'.py'},
            'docs': {'.md', '.txt', '.csv', '.json'}, 'paper': {'.tex', '.bib', '.bst', '.sty', '.png', '.pdf', '.svg', '.csv', '.json', '.txt'}}
EVIDENCE = tuple('results/research_checks/'+name for name in (
    'artifact_archive_manifest.json', 'artifact_verification.json', 'checkpoint_verification.json',
    'completion_audit.json', 'technical_release.json', 'merge_readiness.json', 'full_plan_remaining.json', 'reviewer_preparation.json', 'gossis_source.json', 'source_followup.json', 'wids_download_integrity.json', 'wids_encoder.json', 'wids_status.json', 'tcga_xml/manifest.json', 'tcga_xml/inventory.json', 'missingness_acceptance.csv', 'creatinine_sensitivity.json', 'reference_metadata.json', 'open_dataset_access.json', 'nepal_sensitivity.json', 'nepal_preprocessing.json', 'nepal_source_evidence.json', 'nepal_diagnostics.json', 'tcga_pathology.json', 'tcga_pathology_date_review.json', 'tcga_pathology_molecular.json'))


def build(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    selected = [root/name for name in TOP]
    selected.extend(root/name for name in EVIDENCE if (root/name).is_file())
    for folder, suffixes in SUFFIXES.items():
        selected.extend(path for path in (root/folder).rglob('*')
                        if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts)
    records = []
    total = sum(path.stat().st_size for path in selected)
    import shutil
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < total + 100*1024**2:
        raise ValueError('Insufficient package disk space')
    if destination.exists():
        raise ValueError('Package already exists; choose a new destination')
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Package source is a link or outside the project')
        records.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    manifest = dict(schema_version=1, base_commit=commit, files=records,
                    scope='Source, tests, aggregate paper assets and instructions only. No data, checkpoints, caches, patient traces, private study materials, credentials or Git history. File hashes identify packaged bytes, including any uncommitted changes.')
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for row in records:
            archive.write(root/row['path'], row['path'])
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    return verify(destination)


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate package members')
        manifest = json.loads(archive.read('PACKAGE_MANIFEST.json'))
        expected = {row['path'] for row in manifest['files']} | {'PACKAGE_MANIFEST.json'}
        if set(names) != expected or len(expected) != len(manifest['files'])+1:
            raise ValueError('Package manifest inventory differs')
        for row in manifest['files']:
            name = PurePosixPath(row['path'])
            if name.is_absolute() or '..' in name.parts or '\\' in row['path'] or ':' in row['path']:
                raise ValueError('Unsafe package path')
            if name.parts[0] not in set(TOP) | set(SUFFIXES) and row['path'] not in EVIDENCE:
                raise ValueError('Unapproved package root')
            content = archive.read(row['path'])
            if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Package hash/size mismatch')
    return dict(passed=True, files=len(manifest['files']), bytes=Path(path).stat().st_size,
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), scope=manifest['scope'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.cache/NGTA-reproducibility.zip')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else build(ROOT, args.output), indent=2))
