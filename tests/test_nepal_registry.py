from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.external_validation import validate_study_policy, task_contrast
from src.nepal_registry import select_admissions, mapping_policy, ADMISSION_COLUMNS
from src.nepal_registry import read_workbook


def source():
    return pd.DataFrame([dict(patient_number=str(i), ad_age="75", ad_sex="Female",
        ad_adm_type="Post operative", ad_adm_em_surg="No", adas_heart_first="90",
        adas_sbp_first="85", ad_year_icu="2021", icudis_status=status)
        for i, status in enumerate(["Alive", "Dead", "Alive", "Dead"])])


def test_daily_rows_do_not_become_patients_or_synthetic_records():
    frame = source()
    frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    cohort, report = select_admissions(frame)
    assert len(cohort) == 4 and report["source_rows"] == 6
    assert cohort.icu_death.sum() == 2
    frame.loc[4, "ad_age"] = "NA"
    cohort, report = select_admissions(frame)
    assert len(cohort) == 3 and report["conflicting_admissions"] == 1
    assert report["conflicting_fields"]["ad_age"] == 1


def test_adult_eligibility_unknown_labels_and_invalid_values():
    frame = source()
    frame.loc[0, "ad_age"] = "17"
    frame.loc[1, "icudis_status"] = "Cancel"
    frame.loc[2, "adas_heart_first"] = "0"
    frame.loc[2, "adas_sbp_first"] = "-1"
    frame.loc[2, "ad_sex"] = "Intersex"
    frame.loc[3, "ad_adm_type"] = "Non operative"
    frame.loc[3, "ad_adm_em_surg"] = "NA"
    cohort, report = select_admissions(frame)
    assert len(cohort) == 2 and cohort.icu_death.sum() == 1
    assert report["missing_invalid_or_nonadult_age"] == 1
    assert report["missing_or_unknown_outcome_in_adults"] == 1
    assert cohort.d1_heartrate_max.isna().sum() == 1
    assert cohort.d1_sysbp_min.isna().sum() == 1
    assert cohort.gender.isna().sum() == 1
    assert set(cohort.elective_surgery) == {0., 1.}
    frame.loc[2, "patient_number"] = None
    with pytest.raises(ValueError, match="identifiers"):
        select_admissions(frame)


def test_no_ambiguous_unit_fields_are_mapped_and_task_shift_cannot_be_certified():
    processor = SimpleNamespace(numeric_columns=["age", "d1_heartrate_max", "d1_sysbp_min",
        "d1_temp_max", "d1_creatinine_max", "d1_wbc_max"],
        binary_columns=["elective_surgery"], categorical_columns=["gender"])
    policy = mapping_policy(processor, "admission_core", "a"*64)
    for name in ("d1_temp_max", "d1_creatinine_max", "d1_wbc_max"):
        assert policy["features"][name]["column"] is None
    assert validate_study_policy(policy, "wids") == "cross_task_sensitivity"
    broken = deepcopy(policy)
    broken["clinical_claim_eligible"] = True
    with pytest.raises(ValueError, match="disclaim"):
        validate_study_policy(broken, "wids")
    broken = deepcopy(policy)
    broken["study_scope"] = "external_validation"
    with pytest.raises(ValueError, match="do not match"):
        validate_study_policy(broken, "wids")
    broken = deepcopy(policy)
    del broken["label_shift"]
    with pytest.raises(ValueError, match="endpoint shift"):
        validate_study_policy(broken, "wids")


def test_changed_endpoint_contrast_has_no_compatibility_acceptance():
    report = task_contrast([0, 1, 0, 1], [.2, .8, .3, .7], [.2, .8, .3, .7], iterations=20)
    assert report["brier_nars_minus_mc"] == 0
    assert report["auc_nars_minus_mc"] == 0
    assert not any("margin" in name or "accept" in name for name in report)
    assert report["clinical_claim_eligible"] is False


def test_workbook_reader_preserves_sparse_columns_and_rejects_formulas(tmp_path):
    import zipfile
    path = tmp_path / "fixture.xlsx"
    xml = '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
    xml += '<row r="1"><c r="A1" t="inlineStr"><is><t>id</t></is></c><c r="C1" t="inlineStr"><is><t>age</t></is></c></row>'
    xml += '<row r="2"><c r="A2" t="inlineStr"><is><t>one</t></is></c><c r="C2"><v>75</v></c></row>'
    xml += '</sheetData></worksheet>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", xml)
    assert read_workbook(path).to_dict("records") == [{"id": "one", "age": "75"}]
    assert read_workbook(path, ["id"]).to_dict("records") == [{"id": "one"}]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", xml.replace('<v>75</v>', '<f>50+25</f><v>75</v>'))
    with pytest.raises(ValueError, match="formulas"):
        read_workbook(path)


def test_hierarchical_brier_uses_float64_and_fixed_seed_alignment():
    from scripts.evaluate_nepal import hierarchical_brier
    y = np.array([0., 1., 0., 1.])
    mc = np.array([.2, .8, .3, .7], dtype=np.float32)
    nars = mc.astype(float) - 1e-9
    result = hierarchical_brier([(nars, mc)]*5, y, iterations=30)
    expected = np.mean((nars-y)**2-(mc.astype(float)-y)**2)
    assert result["brier"] == expected
    assert result["family_comparisons"] == 4
    assert result["family_lower_95"] <= result["lower_95"] <= result["upper_95"] <= result["family_upper_95"]
    with pytest.raises(ValueError, match="five aligned"):
        hierarchical_brier([(nars, mc)]*4, y)
