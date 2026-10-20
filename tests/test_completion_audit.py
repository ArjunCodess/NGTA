from pathlib import Path
import json

import pytest

from scripts import update_completion_audit


def test_malformed_requirement_cannot_silently_disappear_from_audit(tmp_path, monkeypatch):
    original = (Path(__file__).resolve().parents[1] / "list.md").read_text()
    malformed = original.replace("**Prove the symbolic component matters.**", "Prove the symbolic component matters.")
    path = tmp_path / "list.md"
    path.write_text(malformed)
    monkeypatch.setattr(update_completion_audit, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="unparsed requirements"):
        update_completion_audit.update()
    assert path.read_text() == malformed


def test_owner_exclusion_is_struck_out_without_becoming_completion(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / 'list.md').write_text((repository / 'list.md').read_text())
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/scope-policy.json').write_text((repository / 'docs/scope-policy.json').read_text())
    (tmp_path / 'results/research_checks').mkdir(parents=True)
    monkeypatch.setattr(update_completion_audit, 'ROOT', tmp_path)
    report = update_completion_audit.update()
    assert report['owner_excluded'] == 4
    assert report['active_total'] == 68
    assert report['completed'] + report['active_unsatisfied'] + report['owner_excluded'] == 72
    assert all(not row['completed'] and row['status'] == 'excluded by owner'
               for row in report['tasks'] if row['owner_excluded'])
    content = (tmp_path / 'list.md').read_text()
    assert content.count('~~') == 8
    assert update_completion_audit.update() == report
    assert (tmp_path / 'list.md').read_text() == content


def test_unknown_scope_exclusion_cannot_silently_remove_requirement(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    original = (repository / 'list.md').read_text()
    (tmp_path / 'list.md').write_text(original)
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/scope-policy.json').write_text(json.dumps(dict(excluded_requirements=['invented task'])))
    monkeypatch.setattr(update_completion_audit, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='unknown requirements'):
        update_completion_audit.update()
    assert (tmp_path / 'list.md').read_text() == original
