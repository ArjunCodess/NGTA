import json

import pandas as pd
import pytest

from src.reviewer_study import create_reviewer_package, analyze_reviewer_responses


def test_blinded_assignments_balance_cases_and_reviewers_without_repeated_cases(tmp_path):
    root = tmp_path / "study"
    create_reviewer_package(root, reviewers=4, cases=8)
    assigned = pd.read_csv(root / "participant" / "assignments.csv")
    assert not assigned.duplicated(["participant_id", "case_id"]).any()
    assert assigned.groupby(["case_id", "format"]).size().eq(2).all()
    assert assigned.groupby(["participant_id", "format"]).size().eq(4).all()
    public = (root / "participant" / "tasks.json").read_text()
    assert '"fault"' not in public
    assert not (root / "response_analysis.json").exists()
    with pytest.raises(ValueError, match="overwrite"):
        create_reviewer_package(root, reviewers=4, cases=8)


def test_incomplete_reviewer_responses_do_not_produce_human_results(tmp_path):
    root = tmp_path / "study"
    create_reviewer_package(root, reviewers=4, cases=8)
    template = root / "participant" / "responses_template.csv"
    with pytest.raises(ValueError, match="binary"):
        analyze_reviewer_responses(root, template, iterations=20)
    assert not (root / "response_analysis.json").exists()


def test_synthetic_response_fixture_exercises_crossed_analysis(tmp_path):
    root = tmp_path / "study"
    create_reviewer_package(root, reviewers=4, cases=8)
    frame = pd.read_csv(root / "participant" / "assignments.csv")
    truth = json.loads((root / "investigator" / "truth.json").read_text())
    faults = {row["case_id"]: row["fault"] for row in truth["truth"]}
    frame["localized_fault"] = [faults[case] if i % 3 else "none" for i, case in enumerate(frame.case_id)]
    frame = frame.assign(correction_correct=0, reassured=0, completion_seconds=10.)
    fixture = tmp_path / "synthetic_fixture.csv"
    frame.to_csv(fixture, index=False)
    report = analyze_reviewer_responses(root, fixture, iterations=20)
    assert report["response_count"] == 32
    assert report["mixed_effects_sensitivity"]["method"].startswith("variational Bayes")
    assert not report["acceptance_criteria_met"]


def test_private_review_materials_have_opaque_ids_and_only_assigned_formats(tmp_path):
    root = tmp_path / "sealed"
    create_reviewer_package(root, reviewers=4, cases=8, seed=812391)
    assigned = pd.read_csv(root / "participant" / "assignments.csv")
    assert assigned.case_id.str.match(r"case_[0-9a-f]{16}$").all()
    keys = []
    for participant_id, selected in assigned.groupby("participant_id"):
        directory = root / "participant" / participant_id
        tasks = json.loads((directory / "tasks.json").read_text())
        assert {(row["case_id"], row["format"]) for row in tasks["evaluation"]} == set(zip(selected.case_id, selected["format"]))
        assert len(tasks["evaluation"]) == len(selected)
        assert not any("fault" in row or "seed" in row for row in tasks["evaluation"])
        assert "correction_correct" not in pd.read_csv(directory / "responses.csv").columns
        keys.extend((row["case_id"], row["format"]) for row in tasks["evaluation"])
    second = tmp_path / "different"
    create_reviewer_package(second, reviewers=4, cases=8, seed=812392)
    a = json.loads((root / "investigator" / "truth.json").read_text())
    b = json.loads((second / "investigator" / "truth.json").read_text())
    assert a["truth"] != b["truth"]
