"""Check retained local/archive files against the committed artifact manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, manifest):
    root = Path(root).resolve()
    report = json.loads(Path(manifest).read_text())
    failures = []
    for record in report["files"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path is outside the selected archive root")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            failures.append(record["path"])
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            failures.append(record["path"])
    return dict(expected=len(report["files"]), verified=len(report["files"]) - len(failures),
                failed=failures, passed=not failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest", default="results/research_checks/artifact_archive_manifest.json")
    args = parser.parse_args()
    result = verify(args.root, args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
