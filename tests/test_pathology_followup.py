from scripts.pathology_followup import scope_flags


def test_pending_and_limited_assay_scopes_do_not_become_gene_calls():
    flags = scope_flags("BRAF codons 599-601 NOT identified. Review pending for another specimen. Prior molecular testing of fine needle aspirate.")
    assert flags["hotspot_or_codon_mentions"] and flags["pending_test_mentions"]
    assert flags["prior_specimen_mentions"]
    assert not any("negative" in name or "callable" in name for name in flags)


def test_indeterminate_is_a_review_flag_not_a_negative():
    assert scope_flags("Mutational analysis is INDETERMINATE due to poor quality isolated DNA")["indeterminate_mentions"]
    assert not any(scope_flags("No molecular testing information available").values())
