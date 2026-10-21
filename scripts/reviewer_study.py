import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reviewer_study import create_reviewer_package, analyze_reviewer_responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare or analyze the blinded synthetic reviewer study")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--responses-csv")
    parser.add_argument("--reviewers", type=int, default=24)
    parser.add_argument("--cases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--private", action="store_true", help="Generate a fresh secret-seeded package; deliver only individual reviewer folders")
    args = parser.parse_args()
    if args.private and args.responses_csv:
        parser.error("--private prepares materials; do not combine it with response analysis")
    if args.private:
        destination = Path(args.output_dir).resolve()
        repository = Path(__file__).resolve().parents[1]
        if destination.is_relative_to(repository) and not any(destination.is_relative_to(repository / name) for name in ("results/private", "results/reviewer_private", "data/private")):
            parser.error("Fresh private study materials must use an ignored private directory")
        create_reviewer_package(destination, args.reviewers, args.cases, secrets.randbits(64))
        # Master assignments, cross-format tasks and grading templates are investigator-only.
        for name in ("assignments.csv", "tasks.json", "responses_template.csv"):
            (destination / "participant" / name).replace(destination / "investigator" / name)
        print(json.dumps(dict(status="private materials prepared; no participants or responses", reviewers=args.reviewers,
                              cases=args.cases, deliver="one participant/reviewer_* folder per reviewer after protocol finalization")))
    else:
        report = (analyze_reviewer_responses(args.output_dir, args.responses_csv, seed=args.seed) if args.responses_csv
                  else create_reviewer_package(args.output_dir, args.reviewers, args.cases, args.seed))
        print(json.dumps(report, indent=2))
