import argparse
import json
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
    args = parser.parse_args()
    report = (analyze_reviewer_responses(args.output_dir, args.responses_csv, seed=args.seed) if args.responses_csv
              else create_reviewer_package(args.output_dir, args.reviewers, args.cases, args.seed))
    print(json.dumps(report, indent=2))
