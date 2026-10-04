import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gdc_acquisition import pin_thca_manifest, download_pinned_manifest, verify_pinned_sources


def test_pinned_query_uses_case_coverage_and_all_pages_instead_of_file_size(tmp_path, monkeypatch):
    pages = [dict(data=dict(hits=[dict(file_id="1", file_name="a.maf", md5sum="a", file_size=50,
                                     access="open", cases=[dict(submitter_id="TCGA-AA-0001")],
                                     analysis=dict(workflow_type="Aliquot Ensemble"))], pagination=dict(total=2))),
             dict(data=dict(hits=[dict(file_id="2", file_name="b.maf", md5sum="b", file_size=10,
                                     access="open", cases=[dict(submitter_id="TCGA-AA-0002")],
                                     analysis=dict(workflow_type="Aliquot Ensemble"))], pagination=dict(total=2)))]
    offsets = []
    def get(url, params, timeout):
        offsets.append(params["from"])
        data = pages[len(offsets)-1]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr("src.gdc_acquisition.requests.get", get)
    path = tmp_path / "manifest.json"
    report = pin_thca_manifest(path, ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"])
    assert offsets == [0,1] and len(report["files"]) == 2
    assert report["unavailable_expected_cases"] == ["TCGA-AA-0003"]
    assert not report["complete_expected_file_coverage"]
    with pytest.raises(ValueError, match="already exists"):
        pin_thca_manifest(path)


def test_reused_pinned_files_require_checksums_and_exact_case_provenance(tmp_path):
    maf = tmp_path / "a.maf"
    maf.write_text("Hugo_Symbol\tTumor_Sample_Barcode\nBRAF\tTCGA-AA-0001-01\n")
    record = dict(file_id="1", file_name="a.maf", bytes=maf.stat().st_size,
                  md5=hashlib.md5(maf.read_bytes()).hexdigest(), case_ids=["TCGA-AA-0001"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(files=[record], case_ids=record["case_ids"])))
    report = download_pinned_manifest(path, tmp_path)
    assert report["variant_case_ids"] == ["TCGA-AA-0001"]
    assert not report["verified_negative_calls"]
    maf.write_text(maf.read_text()+"corruption")
    with pytest.raises(ValueError, match="changed"):
        download_pinned_manifest(path, tmp_path)


def test_verified_source_rejects_unpinned_maf(tmp_path):
    maf=tmp_path/"a.maf"
    maf.write_text("Hugo_Symbol\tTumor_Sample_Barcode\nBRAF\tTCGA-AA-0001-01\n")
    record=dict(file_id="1",file_name="a.maf",bytes=maf.stat().st_size,
                md5=hashlib.md5(maf.read_bytes()).hexdigest(),case_ids=["TCGA-AA-0001"])
    manifest=tmp_path/"acquisition_manifest.json"
    manifest.write_text(json.dumps(dict(files=[record],case_ids=record["case_ids"])))
    download_pinned_manifest(manifest,tmp_path)
    assert verify_pinned_sources(tmp_path)["variant_case_ids"]==["TCGA-AA-0001"]
    (tmp_path/"unexpected.maf").write_text(maf.read_text())
    with pytest.raises(ValueError,match="unpinned"):
        verify_pinned_sources(tmp_path)
