"""Command-line entry point for frozen independent-cohort evaluation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external_validation import evaluate_external

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mc-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate_external(**vars(args)), indent=2))
