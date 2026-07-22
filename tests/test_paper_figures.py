from __future__ import annotations

from src.paper_figures import _pipeline_architecture, _write_rule_table


def test_static_paper_artifacts_are_generated_from_code(tmp_path) -> None:
    _pipeline_architecture(tmp_path)
    _write_rule_table(tmp_path)

    assert (tmp_path / "figure-1.png").stat().st_size > 0
    rule_table = (tmp_path / "generated_rule_table.tex").read_text(encoding="utf-8")
    assert "BRAF mutation indicator = 1" in rule_table
    assert "maximum day-1 lactate >= 4.0 mmol/L" in rule_table
    assert "not validated clinical probabilities" in rule_table
