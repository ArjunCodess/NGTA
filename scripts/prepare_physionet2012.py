"""Download checksum-pinned open challenge records and prepare exploratory inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.physionet2012 import SOURCE, mapping_policy, read_archive


def prepare(cache, output, checkpoint, download=False):
    cache, output = Path(cache), Path(output)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "source_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    sources, cohorts = [], []
    # Freeze mapping before loading outcome files.
    policy = mapping_policy(joblib.load(Path(checkpoint) / "preprocessor.joblib"))
    policy_path = output / "mapping.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Existing sensitivity mapping changed; use a distinct study directory")
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    for subset in "abc":
        paths = []
        for filename in (f"set-{subset}.tar.gz", f"Outcomes-{subset}.txt"):
            path = cache / filename
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f"Download the open source first: {SOURCE + filename}")
                with urllib.request.urlopen(SOURCE + filename, timeout=120) as response:
                    contents = response.read()
                path.write_bytes(contents)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if old and next(item["sha256"] for item in old["files"] if item["filename"] == filename) != digest:
                raise ValueError(f"Pinned source changed: {filename}")
            sources.append(dict(filename=filename, url=SOURCE + filename, sha256=digest, bytes=path.stat().st_size))
            paths.append(path)
        frame = read_archive(paths[0])
        outcomes = pd.read_csv(paths[1], dtype={"RecordID": str})
        if len(outcomes) != 4000 or outcomes.RecordID.duplicated().any() or set(frame.record_id) != set(outcomes.RecordID):
            raise ValueError("Outcome IDs do not exactly cover the source records")
        frame = frame.merge(outcomes[["RecordID", "In-hospital_death"]], left_on="record_id", right_on="RecordID", validate="one_to_one")
        frame = frame.drop(columns="RecordID").rename(columns={"In-hospital_death": "hospital_death"})
        if not frame.hospital_death.isin([0, 1]).all():
            raise ValueError("Invalid mortality labels")
        frame["encounter_id"] = "physionet2012:" + frame.record_id
        frame["patient_id"] = frame.encounter_id  # First ICU stay per subject in the source; canonical identity remains unavailable.
        frame["hospital_id"] = "BIDMC"
        frame["challenge_set"] = subset
        cohorts.append(frame)
    cohort = pd.concat(cohorts, ignore_index=True)
    if cohort.record_id.duplicated().any():
        raise ValueError("Challenge sets overlap")
    cohort.to_csv(output / "cohort.csv", index=False)
    report = dict(source=SOURCE, files=sources, records=len(cohort),
                  mapping_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                  cohort_sha256=hashlib.sha256((output / "cohort.csv").read_bytes()).hexdigest(),
                  clinical_claim_eligible=False, missing_fraction=cohort[list(policy["features"])].isna().mean().to_dict())
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/physionet2012")
    parser.add_argument("--output", default="results/cross_source_sensitivity/physionet2012")
    parser.add_argument("--checkpoint", default="results/hospital_study/median_no_apache/seed_0/wids")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
