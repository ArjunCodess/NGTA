# Source evidence and remaining inputs

This supplements the data-validity and external-validation sections of [plan.md](../plan.md). Public source inspection narrows the blockers, but does not replace clinical chronology, canonical identity linkage or expert review.

## TCGA chronology

Run `python scripts/audit_tcga_timing.py` to reproduce [timing coverage](../results/research_checks/tcga_timing/coverage.json) and the selected physical-record timing export. The script uses the development loader's record-selection policy and pins all five source tables and the fitted feature set by SHA256.

All 507 selected diagnosis records have diagnosis day zero. Selected follow-up records have numeric follow-up offsets for 378 cases, but these dates are not dates for the model's pathology measurements. None of the 507 pathology-detail records has a populated pathology day or timepoint category. Molecular-test dates are also absent. The audit covers all 68 distinct model and rule-source inputs and does not borrow dates from unselected repeated rows.

Needed from the source custodian: pathology acquisition/report availability dates, specimen-to-encounter links, genomic collection/report availability dates and callable case/gene panels. Diagnosis day zero is an entity reference, not proof that all fields were available at a common prediction time. The current study remains retrospective post-pathology association.

## APACHE provenance

The [official eICU result-table documentation](https://eicu.mit.edu/eicutables/apachepatientresult/) explains APACHE IV/IVa predictions and distinguishes hospital from ICU mortality. It describes eligibility exclusions, but does not establish the meaning of negative probabilities in this WiDS CSV or supply their calculation/availability timestamps. The [official APS table](https://eicu.mit.edu/eicutables/apacheapsvar/) describes worst physiology during the APACHE day; this cannot certify the timestamp of a particular exported probability.

The [WiDS publisher page](https://physionet.org/content/widsdatathon2020/1.0.0/) describes first-24-hour prediction and offers a source dictionary through registered access with a signed data-use agreement. No restricted download was attempted. Needed from an authorized user or custodian: the versioned dictionary definition and sentinel codes for `apache_4a_hospital_death_prob`, its computation window, availability relative to the 24-hour landmark and the exact source release/collection period. The invalid local values remain unavailable while both score-removal studies remain valid descriptive sensitivity analyses.

## Public cross-source sensitivity

The [PhysioNet/CinC 2012 source description](https://archive.physionet.org/challenge/2012/) documents elapsed ICU timestamps, admission descriptors, variable units and unknown values encoded as -1. It excludes ICU stays shorter than 48 hours. The [authors' dataset paper](https://physionet.org/files/challenge-2012/1.0.0/papers/0245.pdf) describes the first eligible ICU stay per subject from the single-center MIMIC-II source. The [versioned publisher release](https://physionet.org/content/challenge-2012/1.0.0/) provides the open dataset under the Open Data Commons Attribution License v1.0.

The preparation script downloads only the six open archive/outcome files. It reads tar members in memory, validates exact record/outcome identity and pins downloaded bytes with locally computed SHA256. These hashes identify the acquired snapshot; they are not publisher-supplied checksums. The mapping is saved before local outcomes are loaded. Raw archives stay in the ignored cache, and the derived cohort, mapping and source manifest are preserved under `results/cross_source_sensitivity/physionet2012/`.

Mapping uses admission age, gender and BMI, and extrema from elapsed minutes in [0,1440). Arterial SaO2 is not substituted for pulse-oximetry SpO2. Elective surgery and APACHE probability remain unavailable. Invasive/noninvasive systolic readings are combined as a stated exploratory mapping. Exact WiDS measurement-window equivalence and clinical harmonization still need source review.

`python scripts/evaluate_cross_source.py` evaluates the ten frozen no-APACHE checkpoints across both imputers and five training seeds, using 50 matched dropout draws. Preprocessors and rules are fixed, and every export must independently replay. Results explicitly use conditional case-bootstrap intervals and cannot carry an accepted independent-clinical-validation flag. Source identifier strings cannot establish canonical patient independence, and a single hospital cannot estimate institution-level uncertainty. This public cohort is a sensitivity analysis, not an untouched confirmation cohort.

All ten fits pass independent replay and the conditional numerical margins. Mean ungated AUROC is 0.708135 for KNN and 0.710732 for median. Every NARS-minus-MC Brier difference is positive, so meeting noninferiority does not establish a symbolic improvement. Complete per-seed metrics and intervals are preserved alongside the source manifest.

## Inputs that still need people

Qualified thyroid-oncology and critical-care reviewers must sign off on the prototype predicates, truth weights and correlated-evidence policies. Real reviewer participants must complete the prepared crossover protocol using a fresh private answer key. Source documentation cannot substitute for either activity. A new independent cohort also needs verified source identities, clinical harmonization and a procedure locked before outcome inspection.
