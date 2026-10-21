"""Prepare private Nepal admission cohorts under the recorded exploratory protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nepal_registry import ADMISSION_COLUMNS, SOURCE_SHA256, SCENARIOS, read_workbook, select_admissions, mapping_policy


def prepare(source, output, protocol, checkpoint):
    source, output, protocol = Path(source), Path(output), Path(protocol)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Nepal source differs from the pinned release")
    spec = json.loads(protocol.read_text())
    if spec["source_sha256"] != SOURCE_SHA256 or spec["scenarios"] != list(SCENARIOS):
        raise ValueError("Protocol and cohort preparation differ")
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    processor = joblib.load(Path(checkpoint) / "preprocessor.joblib")
    # Save every complete mapping before loading any source outcomes.
    policies = {}
    for scenario in SCENARIOS:
        root = output / scenario
        root.mkdir(parents=True, exist_ok=True)
        policy = mapping_policy(processor, scenario, digest)
        path = root / "mapping.json"
        if path.exists() and json.loads(path.read_text()) != policy:
            raise ValueError("Existing mapping changed; use a new study directory")
        path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        policies[scenario] = policy
    frame = read_workbook(source, ["patient_number", *ADMISSION_COLUMNS])
    cohort, report = select_admissions(frame)
    report.update(source_sha256=SOURCE_SHA256, protocol_sha256=digest,
                  clinical_claim_eligible=False, scenarios={})
    for scenario, policy in policies.items():
        columns = list(dict.fromkeys(["encounter_id", "admission_id", "icu_death", "source_registry"] +
                                   [x["column"] for x in policy["features"].values() if x["column"]]))
        path = output / scenario / "cohort.csv"
        content = cohort[columns].to_csv(index=False).encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Prepared cohort changed; use a new study directory")
        path.write_bytes(content)
        report["scenarios"][scenario] = dict(cohort_sha256=hashlib.sha256(content).hexdigest(),
            missing_fraction=cohort[columns[4:]].isna().mean().to_dict())
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/private/open_candidates/nepal_registry/nicrf_encounters.xlsx")
    parser.add_argument("--output", default="results/private/nepal_sensitivity")
    parser.add_argument("--protocol", default="docs/nepal-analysis.json")
    parser.add_argument("--checkpoint", default="results/hospital_study/median_no_apache/seed_0/wids")
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
