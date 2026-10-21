import hashlib

import pytest

from src.supplement_audit import acquire_supplements, inventory_supplements, link_specimen_records, present, read_biotab


def test_metadata_and_duplicate_columns_are_preserved(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("id\twidth\twidth\tdays_to_test\n"
                    "id\tlength\tdepth\tdays_to_test\n"
                    "CDE_ID:\tCDE_ID:1\tCDE_ID:2\tCDE_ID:3\n"
                    "case\t1\t2\t[Not Available]\n")
    header, aliases, rows = read_biotab(path)
    assert header[1:3] == ["width", "width"]
    assert aliases[1:3] == ["length", "depth"]
    assert len(rows) == 1 and rows[0][3] == "[Not Available]"
    path.write_text("id\tvalue\ncase\n")
    with pytest.raises(ValueError, match="width"):
        read_biotab(path)


def test_inventory_checks_bytes_and_does_not_promote_dates_or_missing_results(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("id\tdays_to_collection\tbraf_mutation_result\n"
                    "case\t0\t[Unknown]\nother\tinf\tNegative\n")
    item = dict(file_id="id", file_name=path.name, file_size=path.stat().st_size,
                md5sum=hashlib.md5(path.read_bytes()).hexdigest(), access="open", data_format="BCR Biotab")
    manifest = dict(files=[item])
    report = inventory_supplements(manifest, tmp_path)
    assert report["tables"][0]["records"] == 2
    assert report["tables"][0]["fields"][0]["finite_numeric_rows"] == 1
    assert report["tables"][0]["fields"][1]["reported_values"] == {"[Unknown]": 1, "Negative": 1}
    assert not report["cross_table_chronology_verified"] and not report["callable_panel_verified"]
    path.write_text(path.read_text() + "changed")
    with pytest.raises(ValueError, match="bytes changed"):
        inventory_supplements(manifest, tmp_path)
    with pytest.raises(ValueError, match="bytes changed"):
        acquire_supplements(manifest, tmp_path)
    item["file_name"] = "../escape.txt"
    with pytest.raises(ValueError, match="Unsafe"):
        acquire_supplements(manifest, tmp_path)


def test_explicit_specimen_linkage_rejects_patient_conflicts():
    samples = [dict(bcr_sample_uuid="ABC", bcr_sample_barcode="TCGA-AA-0001-01A")]
    tumors = [dict(bcr_sample_uuid="abc", bcr_patient_barcode="TCGA-AA-0001",
                   tss_pathology_review_days_to="0", tumor_sample_procurement_days_to="-10")]
    report = link_specimen_records(samples, tumors)
    assert report["matched_cases"] == 1 and report["records_with_both_offsets"] == 1
    assert not report["clinical_feature_availability_verified"]
    assert present("[Not Available]|No BRAF Mutation Detected")
    assert not present("[Not Available]|[Not Evaluated]")
    tumors[0]["bcr_patient_barcode"] = "TCGA-AA-0002"
    with pytest.raises(ValueError, match="different patient"):
        link_specimen_records(samples, tumors)
    with pytest.raises(ValueError, match="duplicate"):
        link_specimen_records(samples * 2, [])
