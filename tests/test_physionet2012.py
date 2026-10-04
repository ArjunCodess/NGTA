import numpy as np
import pytest

from src.physionet2012 import summarize_record


def test_window_sentinel_units_and_unavailable_measurements():
    record = "Time,Parameter,Value\n00:00,RecordID,1\n00:00,Age,75\n00:00,Gender,0\n00:00,Height,200\n00:00,Weight,80\n01:00,Weight,120\n01:00,HR,100\n23:59,HR,110\n24:00,HR,200\n01:00,Lactate,-1\n25:00,Lactate,10\n01:00,SysABP,95\n01:00,NISysABP,85\n01:00,SaO2,50\n"
    result = summarize_record(record, "1")
    assert result["d1_heartrate_max"] == 110
    assert result["bmi"] == 20
    assert result["gender"] == "F"
    assert result["d1_sysbp_min"] == 85
    assert np.isnan(result["d1_lactate_max"])
    assert np.isnan(result["d1_spo2_min"])
    assert np.isnan(result["elective_surgery"])
    assert np.isnan(result["apache_4a_hospital_death_prob"])


def test_identity_and_clock_are_checked_and_ambiguous_admission_is_unknown():
    with pytest.raises(ValueError, match="filename"):
        summarize_record("Time,Parameter,Value\n00:00,RecordID,2\n", "1")
    with pytest.raises(ValueError, match="timestamp"):
        summarize_record("Time,Parameter,Value\n00:99,RecordID,1\n", "1")
    result = summarize_record("Time,Parameter,Value\n00:00,RecordID,1\n00:00,Weight,80\n00:00,Weight,90\n00:00,Height,200\n", "1")
    assert np.isnan(result["bmi"])
