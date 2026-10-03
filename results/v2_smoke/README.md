# Integration run only

This bundle uses the actual local TCGA sources, two training epochs, d_model=8, and three MC passes. It exercises standard baselines, complete events, independent replay, 100 rule permutations, raw missingness, two ensemble members, and encoder interventions. It is not a replacement clinical study and cannot establish genomic fusion, symbolic benefit, robustness, or clinical utility.

Reproduce with the command in the saved per-dataset run configuration. Every emitted event and matched prediction is replayable from `tcga/traces` using `python -m src.trace_replay`.
