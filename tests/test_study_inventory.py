import pytest

from scripts.verify_studies import verify


def test_missing_studies_cannot_pass_vacuously(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='Full-study inventory differs'):
        verify()
    assert not (tmp_path/'results/research_checks/artifact_verification.json').exists()
