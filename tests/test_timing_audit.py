import pandas as pd

from src.timing_audit import summarize_timing


def test_dates_are_not_borrowed_from_other_physical_records():
    frame = pd.DataFrame({"case_submitter_id": ["a", "a"],
                          "pathology_details.days_to_pathology_detail": [None, "30"],
                          "measurement": ["5", None], "extra": ["x", None], "another": ["y", None]})
    selected, fields, records = summarize_timing(frame)
    assert selected.measurement.iloc[0] == "5"
    assert fields[0]["raw_nonmissing"] == 1
    assert fields[0]["selected_numeric_days"] == 0
    assert records[0]["numeric_days"] is None


def test_unknown_days_are_not_zero_and_categories_are_not_dates():
    frame = pd.DataFrame({"case_submitter_id": ["a", "b", "c"],
                          "diagnoses.days_to_diagnosis": ["0", "not reported", "inf"],
                          "cases.index_date": ["Diagnosis"] * 3})
    _, fields, records = summarize_timing(frame)
    assert fields[0]["selected_numeric_days"] == 1
    assert fields[0]["selected_min_days"] == 0
    assert fields[1]["selected_numeric_days"] is None
    assert fields[1]["selected_categories"] == ["Diagnosis"]
    assert [row["numeric_days"] for row in records[:3]] == [0, None, None]
