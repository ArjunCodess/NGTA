# Source evidence and remaining inputs

This supplements the data-validity and external-validation sections of [plan.md](../plan.md). Public source inspection narrows the blockers, but does not replace clinical chronology, canonical identity linkage or expert review.

## TCGA chronology

Run `python scripts/audit_tcga_timing.py` to reproduce [timing coverage](../results/research_checks/tcga_timing/coverage.json) and the selected physical-record timing export. The script uses the development loader's record-selection policy and pins all five source tables and the fitted feature set by SHA256.

All 507 selected diagnosis records have diagnosis day zero. Selected follow-up records have numeric follow-up offsets for 378 cases, but these dates are not dates for the model's pathology measurements. None of the 507 pathology-detail records has a populated pathology day or timepoint category. Molecular-test dates are also absent. The audit covers all 68 distinct model and rule-source inputs and does not borrow dates from unselected repeated rows.

The [GDC supplement documentation](https://gdc.cancer.gov/content/where-can-i-find-clinical-data-elements-specific-my-cancer-research-interest) identifies public clinical/biospecimen biotabs as an additional source. `python scripts/audit_tcga_supplements.py` now pins and verifies all 19 THCA biotabs, totaling 4,062,798 bytes. The [inventory](../results/research_checks/tcga_supplements/inventory.json) retains source field names, dictionary aliases, missingness, offsets and reported genotyping values. Duplicate header names are preserved by column position rather than silently merged. Raw downloads remain under `.cache/ngta-supplements/`.

All 515 tumor specimen rows have procurement and tissue-source-site pathology-review offsets and link by sample UUID to biospecimen records with matching patient barcodes across 507 cases. No review offset precedes its paired procurement offset. These links narrow the specimen-provenance gap, but do not connect the selected model pathology fields or genomic result availability to a verified prediction episode. Patient biotabs report 29 YES genotyping indicators and 27 BRAF results, including explicit negative wording; the source also retains unavailable/not-evaluated results. Those assays and their tested regions have not been reconciled with the frozen WXS gene panel, so no mutation calls are changed.

Needed from the source custodian: pathology acquisition/report availability dates, specimen-to-encounter links, genomic collection/report availability dates and callable case/gene panels. Diagnosis day zero is an entity reference, not proof that all fields were available at a common prediction time. The current study remains retrospective post-pathology association.

## APACHE provenance

The [official eICU result-table documentation](https://eicu.mit.edu/eicutables/apachepatientresult/) explains APACHE IV/IVa predictions and distinguishes hospital from ICU mortality. It describes eligibility exclusions, but does not establish the meaning of negative probabilities in this WiDS CSV or supply their calculation/availability timestamps. The [official APS table](https://eicu.mit.edu/eicutables/apacheapsvar/) describes worst physiology during the APACHE day; this cannot certify the timestamp of a particular exported probability.

The [WiDS publisher page](https://physionet.org/content/widsdatathon2020/1.0.0/) describes first-24-hour prediction and offers a source dictionary through registered access with a signed data-use agreement. The owner downloaded the release dictionary into the ignored `data/private/` directory on 2026-10-20. It parses as 188 unique variable definitions and identifies the score as APACHE IVa, using the APACHE III score plus other covariates. Its SHA256 is `8c9b3b166cb519528687060edfc8b2a7f07b0bc7da18f926e6c52be1402618eb`. The inspected definitions do not resolve sentinel codes or score availability. Needed from the custodian: sentinel semantics, calculation/availability timing relative to the 24-hour landmark, and exact source release/collection provenance. The invalid local values remain unavailable while both score-removal studies remain valid descriptive sensitivity analyses.

## Public cross-source sensitivity

The [PhysioNet/CinC 2012 source description](https://archive.physionet.org/challenge/2012/) documents elapsed ICU timestamps, admission descriptors, variable units and unknown values encoded as -1. It excludes ICU stays shorter than 48 hours. The [authors' dataset paper](https://physionet.org/files/challenge-2012/1.0.0/papers/0245.pdf) describes the first eligible ICU stay per subject from the single-center MIMIC-II source. The [versioned publisher release](https://physionet.org/content/challenge-2012/1.0.0/) provides the open dataset under the Open Data Commons Attribution License v1.0.

The preparation script downloads only the six open archive/outcome files. It reads tar members in memory, validates exact record/outcome identity and pins downloaded bytes with locally computed SHA256. These hashes identify the acquired snapshot; they are not publisher-supplied checksums. The mapping is saved before local outcomes are loaded. Raw archives stay in the ignored cache, and the derived cohort, mapping and source manifest are preserved under `results/cross_source_sensitivity/physionet2012/`.

Mapping uses admission age, gender and BMI, and extrema from elapsed minutes in [0,1440). Arterial SaO2 is not substituted for pulse-oximetry SpO2. Elective surgery and APACHE probability remain unavailable. Invasive/noninvasive systolic readings are combined as a stated exploratory mapping. Exact WiDS measurement-window equivalence and clinical harmonization still need source review.

`python scripts/evaluate_cross_source.py` evaluates the ten frozen no-APACHE checkpoints across both imputers and five training seeds, using 50 matched dropout draws. Preprocessors and rules are fixed, and every export must independently replay. Results explicitly use conditional case-bootstrap intervals and cannot carry an accepted independent-clinical-validation flag. Source identifier strings cannot establish canonical patient independence, and a single hospital cannot estimate institution-level uncertainty. This public cohort is a sensitivity analysis, not an untouched confirmation cohort.

All ten fits pass independent replay and the conditional numerical margins. Mean ungated AUROC is 0.708135 for KNN and 0.710732 for median. Every NARS-minus-MC Brier difference is positive, so meeting noninferiority does not establish a symbolic improvement. Complete per-seed metrics and intervals are preserved alongside the source manifest.

## Inputs that still need people

The owner has excluded external clinical expert review and participant collection as unavailable. Prototype rule validity and human benefit therefore remain unsupported; the prepared materials are retained, with no further recruitment requested. Source documentation does not substitute for either type of evidence. A new independent cohort also needs verified source identities, clinical harmonization and a procedure locked before outcome inspection.

## Newly identified source-unit conflict

The downloaded dictionary labels `d1_creatinine_max` as micromol/L, while the frozen prototype rule assumes mg/dL. The existing training column has 81,544 observed values, median 1.0 and range 0.34?11.11. These values raise a source-definition conflict; their distribution alone does not certify a unit. No automatic conversion, rule change or model refit was performed. Publisher clarification is required before clinical interpretation of that rule or external unit harmonization. The private dictionary audit is saved locally at `data/private/wids_dictionary_audit.json`; source files and private download listings remain out of Git. The GOSSIS message now includes this question.

## Publisher extraction-code audit

The [official GOSSIS extraction repository](https://github.com/MIT-LCP/gossis) is now pinned at revision `95b69b99620d322b088262a41cd8098a1aed2f3e`. Seven source files were downloaded and checked against their publisher Git blob hashes. The eICU SQL directly projects aggregated creatinine into the day-one field and uses a raw creatinine range of 0.1?28.28; its APACHE output casts the source hospital-mortality prediction. The header definitions retain micromol/L. This narrows the inconsistency to the publisher's own extraction/definition sources, but the exact WiDS production revision and source-to-release unit provenance remain unverified. The report records immutable source URLs, hashes and line anchors: [gossis_source.json](../results/research_checks/gossis_source.json). It does not promote a source-family code path into a verified WiDS release lineage.

## Complete public TCGA XML audit

`python scripts/audit_tcga_xml.py` now acquires and verifies all 1,554 pinned public XML supplements, totaling 65,507,832 bytes across 507 cases: 1,014 BCR XML files, 507 specimen-supplement XML files and 33 other-malignancy XML files. The [inventory](../results/research_checks/tcga_xml/inventory.json) records 94 matching full field paths, source SHA256s and case identity checks. Publisher MD5/size and associated patient barcodes are verified before counting a file. Raw XML remains in the ignored cache. The API batch failure was recovered through smaller requests and individual-file fallback; no source files remain unavailable in this snapshot.

The XML confirms 515 tumor procurement/pathology-review offset occurrences, while the generic sample-procurement field is empty for 1,053 sample entries. Diagnosis offsets remain zero in all 507 clinical records. Upload and form-completion dates refer to their named administrative events. The audit does not establish the availability of the selected pathology/model measurements or genomic reports, and does not provide verified case/gene callability. Those criteria remain unmet after checking the complete public supplement snapshot. [Official batch-download API](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/) describes the acquisition mechanism.

## Search follow-up and owner scope decision

The [next-work assessment](next-work.md) records the latest official-source findings: 515 open THCA pathology PDFs totaling 79,840,335 bytes, publisher evidence relevant to APACHE -1 handling, and gated HiRID/SICdb cohort routes. The pathology PDFs are inventoried but their contents have not yet been audited. Four expert/participant requirements are excluded from active work, without becoming completed scientific criteria.

## WiDS execution decision, 2026-10-21

The owner excludes further WiDS source clarification from active execution requirements. Existing input conventions, unavailable-value handling, score-removal and creatinine-rule-off results are preserved; no unit or timestamp is newly certified. This does not block the completed 20-checkpoint WiDS encoder extension. No stopped download was resumed. [Internet-only cohort options](internet-data.md) include a credentialing/DUA route whose listed policy does not require an additional outside researcher or contributor study review; compatibility remains unestablished.
