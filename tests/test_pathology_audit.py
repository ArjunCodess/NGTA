import hashlib

import pytest

from scripts.audit_tcga_pathology import verify, text_inventory


def test_pathology_checks_publisher_bytes_and_open_access():
    body = b"%PDF-1.4\nsynthetic fixture"
    row = dict(access="open", data_format="PDF", file_id="a"*8+"-"+"b"*4+"-"+"c"*4+"-"+"d"*4+"-"+"e"*12,
               file_size=len(body), md5sum=hashlib.md5(body).hexdigest())
    verify(row, body)
    with pytest.raises(ValueError, match="checksum"):
        verify(row, body+b"changed")
    row["access"] = "controlled"
    with pytest.raises(ValueError, match="metadata"):
        verify(row, body)


def test_pathology_patterns_do_not_promote_mentions_to_verified_timing():
    report = text_inventory("Report date: 01/02/2003. Received date XX/XX/XXXX. BRAF mutation testing.")
    assert report["numeric_date_candidates"] == 1
    assert report["report_date_mentions"] >= 1
    assert report["molecular_test_mentions"] == 2
    assert not any("verified" in key for key in report)
