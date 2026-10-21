import hashlib
import io
import tarfile

import pytest

from src.xml_source_audit import consume_archive, inspect_xml, inventory_xml


def record(body):
    return dict(file_id='12345678-1234-1234-1234-123456789abc',file_name='source.xml',file_size=len(body),
                md5sum=hashlib.md5(body).hexdigest(),access='open',data_format='BCR XML',cases=[dict(submitter_id='TCGA-AA-0001')])


def test_xml_dates_remain_attached_to_their_element_paths_and_cases():
    body=b'<root><bcr_patient_barcode>TCGA-AA-0001</bcr_patient_barcode><followup><days_to_review>30</days_to_review></followup><pathology><days_to_review/></pathology></root>'
    cases,leaves=inspect_xml(record(body),body)
    assert cases=={'TCGA-AA-0001'}
    assert leaves==[('/root/followup/days_to_review','30'),('/root/pathology/days_to_review','')]
    other=body.replace(b'0001',b'0002')
    with pytest.raises(ValueError,match='patient barcodes'):
        inspect_xml(record(other),other)
    entities=b'<!DOCTYPE root><root/>'
    with pytest.raises(ValueError,match='entities'):
        inspect_xml(record(entities),entities)


def test_xml_archive_cannot_write_outside_cache_or_omit_pinned_members(tmp_path):
    body=b'<root/>';row=record(body)
    def archive(name):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            entry=tarfile.TarInfo(name);entry.size=len(body);tar.addfile(entry,io.BytesIO(body))
        return stream.getvalue()
    with pytest.raises(ValueError,match='unsafe'):
        consume_archive(archive('../source.xml'),{row['file_id']:row},tmp_path)
    consume_archive(archive(row['file_id']+'/source.xml'),{row['file_id']:row},tmp_path)
    assert (tmp_path/(row['file_id']+'.xml')).read_bytes()==body


def test_missing_xml_sources_never_count_as_complete_coverage(tmp_path):
    row = record(b"<root/>")
    row["data_type"] = "Clinical Supplement"
    report = inventory_xml(dict(files=[row]), tmp_path)
    assert report["file_count"] == 0 and report["expected_files"] == 1
    assert not report["acquisition_complete"]
    assert not report["callable_panel_verified"] and not report["cross_table_chronology_verified"]
