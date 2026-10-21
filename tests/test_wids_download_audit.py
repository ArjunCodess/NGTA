import hashlib

import pytest

from scripts.verify_wids_download import audit


def fixture(tmp_path):
    source = tmp_path / 'study.csv'
    source.write_bytes(b'frozen study input')
    recipe = dict(fits=[dict(dataset='wids', sources=[dict(file='wids_icu.csv', bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest())])])
    (tmp_path / 'LICENSE.txt').write_bytes(b'license')
    (tmp_path / 'SHA256SUMS.txt').write_text(hashlib.sha256(b'license').hexdigest()+' LICENSE.txt\n')
    return source, recipe


def test_complete_study_source_does_not_certify_partial_download(tmp_path):
    source, recipe = fixture(tmp_path)
    (tmp_path / 'data').mkdir()
    partial = tmp_path / 'data/training_v2.csv'
    partial.write_bytes(b'partial')
    result = audit(tmp_path, source, recipe)
    assert result['original_study_source']['matches_frozen_recipe']
    assert result['publisher_checksum_entries_pass']
    assert not result['listed_data_sizes_match']
    assert not result['dictionary_matches_inspected_copy']
    assert not result['whole_download_verified']
    assert partial.read_bytes() == b'partial'


def test_publisher_hash_mismatch_is_reported(tmp_path):
    source, recipe = fixture(tmp_path)
    (tmp_path / 'LICENSE.txt').write_bytes(b'different')
    assert not audit(tmp_path, source, recipe)['publisher_checksum_entries_pass']


@pytest.mark.parametrize('name', ['../secret', 'C:/secret', 'data\\secret'])
def test_checksum_paths_cannot_read_outside_download(tmp_path, name):
    source, recipe = fixture(tmp_path)
    (tmp_path / 'SHA256SUMS.txt').write_text('0'*64+' '+name+'\n')
    with pytest.raises(ValueError, match='Unsafe'):
        audit(tmp_path, source, recipe)
