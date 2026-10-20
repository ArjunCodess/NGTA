from pathlib import Path

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
