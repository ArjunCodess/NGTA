# NGTA research audit

The implementation corrections and full internal development studies are complete where their artifacts exist. The plan is not fully achieved: symbolic benefit fails its effect-size criterion, and several clinical claims need provenance or human evidence that code cannot supply. [README.md](../README.md) retains the original project structure. The retained [completion snapshot](../results/research_checks/completion_audit.json) records all 72 tasks, their evidence and exact required actions.

This audit preserves the original research requirements and their evidence as historical context. Planning files were retired during documentation cleanup; the completion snapshot retains their 72 criteria. Earlier `results/v2_validation/` files are historical snapshots, not current completion counts. Running an experiment and obtaining the hoped-for result are separate outcomes.

## 1. Core research and methodology

All readout gates share cached dropout passes and mean per-pass sigmoid probabilities. Uniform gating matches ungated inference within 1e-7. Independent replay reconstructs observed predicates, closed-form revision, normalized gates, predictions and mapped rule-off effects; hashes reject altered exports.

Full controls include all/individual-rule removal, 100 prevalence-preserving predicate permutations, direct boosts, random truth, fixed priors, shuffled confidence assignments, matched irrelevant predicates and frequency/confidence sensitivity. Revised frequency does not drive the gate, and shared measurements do not establish independent evidence. No condition achieves the planned 0.0001 symbolic Brier gain with a favorable hierarchical interval.

| Condition | NARS-minus-MC Brier | Hierarchical 95% interval |
|---|---:|---:|
| WiDS KNN | -2.43e-7 | [-6.91e-7, 1.64e-7] |
| WiDS median | -4.04e-7 | [-1.19e-6, 5.79e-7] |
| WiDS KNN without APACHE | -1.68e-7 | [-4.31e-7, 3.11e-7] |
| WiDS median without APACHE | -2.35e-7 | [-7.85e-7, 6.53e-7] |
| TCGA fused | 1.26e-5 | [-5.69e-5, 1.06e-4] |
| TCGA clinical | 2.63e-5 | [-5.74e-5, 1.33e-4] |
| TCGA genomic | -1.04e-5 | [-4.35e-5, 8.06e-6] |

Negative means lower Brier. These intervals resample cases/hospitals, training seeds and three dropout repeats on fixed development splits; they exclude cohort-selection variability. Preserve these negative/inconclusive findings rather than tuning against the same inspected outcomes.

## 2. Data validity and preprocessing

Filtering, mutation-panel selection, imputation, scaling and encoding fit training data only. Numeric KNN distances are training-standardized. Observed float64 predicates preserve exact boundaries independently of float32 model inputs; imputed-only triggers are suppressed and exported separately. All eight predicates have missing/unknown/invalid/threshold/category tests, including extension and unknown-stage fixes.

WiDS has 91,713 labeled encounters, split 65,737/11,751/14,225, with zero patient/hospital/ICU overlap. Selected-feature missingness is 10.3559%; lactate is 74.5761% missing. The 2,371 invalid APACHE probabilities are unavailable, but source sentinel semantics and score availability remain unverified.

The acquired TCGA source is isolated under `data/acquired_tcga/`. Its immutable manifest pins 498 open GDC files by UUID, workflow, publisher MD5/size and associated cases; extracted SHA256 and tumor-case provenance are verified. Loading rejects altered or extra unpinned MAFs. Sixteen expected clinical cases lack file-associated coverage. The 457 labeled cases split 319/69/69; selected-panel positives cover 254/50/50 cases. An empty MAF or absent variant does not establish a verified negative or gene callability.

Selecting one physical record per source avoids column-wise synthetic records, and 5,170 case/field conflicts remain exported. TCGA is retrospective post-pathology association; cross-table chronology remains unverified.

The selected-record timing audit now checks all five tables and 68 distinct model/rule inputs. All 507 selected diagnosis records have day zero, and 378 selected follow-up records have numeric follow-up dates, but none of the 507 pathology records has a pathology day or timepoint. Molecular-test dates are absent. [Source evidence](source-evidence.md) links the exact coverage export and distinguishes entity dates from feature availability.

The public supplement audit acquired all 19 GDC THCA biotabs, totaling 4,062,798 bytes, with publisher UUID/MD5/size pinning and local SHA256 verification. All 515 tumor specimen procurement/pathology-review offset pairs link by sample UUID to biospecimen records with matching patient barcodes across 507 cases; no paired review precedes procurement. The clinical biotab contains 27 reported BRAF results, including explicit negative wording, and 29 YES genotyping-performed indicators. These reports are not promoted to callable negatives for the frozen WXS panel. Specimen review dates do not date the selected clinical model measurements or genomic reports. [Manifest and inventory](../results/research_checks/tcga_supplements/inventory.json) preserve this additional evidence without changing frozen inputs.

Required input: sourced callable case/gene assay records, TCGA measurement/encounter dates, and the authoritative WiDS/eICU APACHE dictionary plus calculation timestamps. Obtain restricted source documentation through its account/data-use process when necessary; the current CSV and positive-variant files cannot supply these facts.

## 3. Prediction and calibration

Twenty full hospital-held-out fits cover two imputers, APACHE included/excluded and five seeds. Fifteen acquired TCGA fits cover clinical/genomic/fused inputs and five seeds. Each uses up to 60 epochs with validation early stopping, 50 matched dropout passes and three independent repeats. A separately labeled shuffled-neural-target control leaves classical baseline labels unchanged.

Comparators include deterministic/mean-logit inference, validation transformer recalibration, calibrated logistic regression, ExtraTrees, validation-selected boosting, random forest and score-only/raw/recalibrated APACHE. Calibration/discrimination, observed paired deltas, hospital-bootstrap intervals and family-adjusted comparisons are saved.

Removing APACHE lowers KNN mean ungated AUROC from 0.875721 to 0.841118, and median from 0.874062 to 0.842176. Median boosting Brier is 0.057752 versus 0.058157 for its transformer. KNN recalibration Brier is 0.057782 versus 0.057828 for NARS. Ordinary calibration and an existing mortality score have larger effects than symbolic routing; added clinical value over the strongest comparator is not established.

Evidence is under `results/hospital_study/<condition>/analysis/`, `results/tcga_study/<modality>/analysis/`, and `results/research_checks/apache_*.json`, `imputation.json`, `tcga_fused_vs_*.json`.

## 4. Rule system and symbolic intervention

Registries record source notes, versions, prototype review status and collision groups. Same-target collisions are rejected rather than overwritten. TCGA fused fits separately evaluate genuine encoder key-bias intervention with replayed RNG.

Each acquired TCGA test seed exports 39 BRAF, 29 T-stage, 24 extension and 21 age events: 113 total, 89 mapped and 24 explicitly unmapped. This demonstrates positive genomic intervention, not verified negatives or clinical multimodal benefit. WiDS exports 11,457 mapped observed-only events in 8,152/14,225 cases per seed. Repeated seeds reuse patients; their counts must not be summed as independent cases.

Required action: qualified thyroid oncology and critical-care experts must review each of the eight predicates, weights, source interpretations and shared/correlated-evidence policies. Record signed/versioned review evidence before changing `not_reviewed`; tests cannot certify clinical rule semantics.

## 5. Auditability and human oversight

Every trigger is exported with raw/imputed status, provenance, truth values and mapped attention/rule-off effects. Unmapped events receive no fabricated intervention. Complete replay is unconditional; sampled case summaries are supplementary.

The reviewer demonstration includes separate practice/evaluation cases, extraction/imputation/rule faults, balanced 24-reviewer/80-case assignments, response templates, power sensitivity, correction grading, crossed reviewer/case bootstrap and mixed-effects sensitivity. False reassurance uses fault-bearing cases. The committed investigator answers are public, so this is not a blinded participant study.

Required action: finalize an approved protocol with qualified reviewers and statistical review, generate a fresh private answer key, recruit participants, collect independent responses and grade corrections. The 10-point localization and 5-point false-reassurance criteria require real responses. Nominal power at a 10-point alternative does not guarantee an observed 10-point gain.

## 6. Robustness and generalization

Frozen raw masking tests 0/10/30/50/70% additional removal under random, feature-dependent and explicitly labeled outcome-dependent simulations. Imputers/models/rules remain fixed, methods share masks, and the imputed-rule comparator is explicit. Paired integrated degradation and Bonferroni per-level 0.001 harm bounds are exported; error-detection AUROC and tie-aware selective-risk area receive hospital-bootstrap intervals. The machine audit checks every required output before marking completion. Simulation acceptance does not establish population robustness.

Five-member ensembles, MC variance and predictive entropy use a common ensemble-error reference. Each condition reports actual seed means/SDs and seed-mean Student-t intervals separately from case intervals. Fixed demographic subgroup and MC-inclusive decision curves exist for every full hospital fit.

The paired degradation/harm criterion passes in 12/15 median and 1/15 KNN seed/scenario combinations. KNN passes neither random nor feature-dependent masking in any seed. The complete study therefore does not establish general routing robustness; [missingness_acceptance.csv](../results/research_checks/missingness_acceptance.csv) preserves every passing and failing condition.

The fixed WiDS KNN ensemble's mean MC variance detects common ensemble errors with AUROC 0.842276, hospital-bootstrap interval [0.833229, 0.857306]. Neural attention instability gives 0.514950 [0.493533, 0.531916], and revised attention instability gives 0.500647 [0.477047, 0.515520]. The attention-confidence intervals span chance, so a useful predictive uncertainty estimator does not validate the routing trust signal. These intervals condition on the fitted ensemble and exclude model-refitting variability.

Frozen external evaluation requires documented target/landmark, units/windows/availability for every input and rule-only source, unit conversion and canonical identity provenance. It checks supplied case/patient identifier overlap, freezes preprocessing and exports complete replay. Clustered compatibility requires Brier upper bound <0.001 and AUROC lower bound >-0.01. Identifier strings from different namespaces cannot establish patient independence.

The open PhysioNet 2012 source now provides an explicitly exploratory 12,000-case sensitivity using elapsed first-24-hour inputs and ten frozen no-APACHE fits. Its original >=48-hour stay eligibility, unavailable SpO2/surgery inputs, single source institution and unresolved WiDS identity/window equivalence prevent an independent clinical claim. Mapping, source hashes, conditional case intervals, seed summaries and replay checks are preserved under `results/cross_source_sensitivity/physionet2012/`; numerical noninferiority is kept separate from clinical eligibility.

All ten fits pass replay and the conditional numerical noninferiority margins. Five-seed mean ungated AUROC/Brier is 0.708135/0.118159 for KNN and 0.710732/0.115743 for median. NARS-minus-MC Brier is positive in every fit, ranging from 2.81e-7 to 1.18e-5, so this sensitivity does not supply symbolic benefit. These figures describe the mapped public cohort; they cannot isolate a population effect from its missing inputs, selection policy and source differences.

Required input: an authorized independent institution/time cohort with verified feature/target provenance and canonical patient identities. No eligible independent cohort was supplied. Internal benchmark hospitals and separate TCGA/WiDS models do not provide external validation. For confirmation, preregister the entire procedure on genuinely untouched outcomes; inspected labels cannot become unseen.

## 7. Reproducibility and implementation

Caches verify source hashes, partitions, options and preprocessing code identity. Resume validates immutable training provenance before writes. Saved artifacts include checkpoints, fitted preprocessors/estimators, calibration data, exact split/group IDs, native/pass arrays, RNG and dependency versions. `requirements-lock.windows-py314.txt` records the actual environment; the CUDA Torch build requires its matching wheel index.

The 297 generated hospital fitted-estimator/inference-cache files and public sensitivity arrays have been removed from the current Git tree and preserved locally. Their sizes and SHA256 hashes are recorded in the archive manifest and independently verified. New clones require a separate approved archive for exact replay, or regeneration with the training commands. The working directory is not a separate backup. See [artifact storage](artifact-storage.md).

The regression suite passes 89 tests, including exact weighted-bootstrap equivalence with duplicated-case/tie reconstruction, independent replay corruption checks, predicate edges, external patient/unit/rule-only mappings, selected-record timing and public-cohort timestamp/sentinel/identity boundaries. Exploratory evaluation explicitly rejects a clinical eligibility claim. Current figure sources explicitly select five TCGA fused and five WiDS KNN seeds and reject source/split/rule/configuration mismatches. The compiled 12-page PDF was rendered and inspected, with no undefined references or overfull boxes. NeurIPS formatting files are inside `paper/` only; the former top-level directory is absent.

All 36 full-study checkpoints were restored with their fitted processors and saved dropout RNG. Every saved 50-pass probability matrix reproduced exactly, with maximum residual zero. [checkpoint_verification.json](../results/research_checks/checkpoint_verification.json) records source/split/feature checks, independent event replay and fitted-estimator hashes; this is numerical reproducibility, not clinical confirmation.

All 28 cited entries have primary bibliographic metadata verification, links and recorded conflicts. This verifies reference identity, not every scientific claim attributed to each work.

## 8. Claims weakened or withdrawn

Historical unmatched-aggregation numbers remain labeled historical. The unsupported original sparse-source genomic claim stays withdrawn; acquired positive events are reported separately. Numerical fidelity is not clinical safety, human usefulness, equivalence or external noninferiority. Confidence remains heuristic, and shared measurements do not establish independent NARS evidence.

## 9. Prioritized experiments

| Experiment | Execution and remaining criterion |
|---|---|
| 1. Coverage/replay | Pinning, positive coverage, missingness, selection fixes and replay pass; callable assays and timing remain required. |
| 2. Matched aggregation | Full five-seed comparisons and 1e-7 uniform identity pass; benefit threshold is not met. |
| 3. Clinical rule value | Full controls execute; favorable symbolic benefit remains unestablished. |
| 4. Extraction/replay | Complete predicate edges, observed/imputed separation and independent replay pass. |
| 5. APACHE | Twenty paired imputer/score fits and strong baselines execute; score provenance and added clinical value remain unresolved. |
| 6. Missingness/uncertainty | Fully executed; the full acceptance criterion fails. Only 13/30 imputer/seed/scenario combinations pass degradation/harm checks, and attention-confidence intervals span chance. |
| 7. External compatibility | Tooling ready; eligible cohort and verified harmonization required. |
| 8. Human auditability | Protocol/package/power/analysis ready; qualified participant responses required. |

## 10. Immediate priorities and exact blockers

At the owner's request, external validation is deferred rather than a prerequisite for the technical release. Any future clinical validation still requires four kinds of external evidence: callable assays and measurement/score timing; qualified expert review; an authorized independent untouched cohort with verified units/windows/targets/identities; and real reviewer responses from a freshly blinded package. The checklist names the required input for every unfinished task.

Branch publication is resolved. On 2026-10-20 the requested 297 paths were also removed from the published `version-2` history, with current branch-tip contents unchanged. All original local files remain verified. GitHub-side deletion of orphaned LFS objects and old cached/PR references remains unconfirmed. [Push status](push-status.md) records the rewrite verification and GitHub purge request status. [Source evidence](source-evidence.md) records the remaining provenance requirements; outreach drafts have since been retired. The scientific completion count is 52/72.

The final full Git check found one corrupt compressed selective-risk blob. The working CSV matched its exact committed blob identity, so the object was restored without changing the data or rewriting history; damaged bytes remain in the ignored local recovery folder. Full Git and 297-path LFS integrity checks passed before artifact removal; the subsequent local-artifact verification also matched all 297 manifest entries. [Integrity evidence](../results/research_checks/git_integrity.json) records the repaired object and verification.

The remaining scientific blockage is the negative result itself. Keep it, or formulate a clinically motivated new hypothesis and test it on fresh data. Further software work cannot turn these observed comparisons into favorable confirmation.

On 2026-10-20, fresh verification passed all 36 full-study source/checkpoint/preprocessor identity checks and independent persisted event/prediction replays. All 297 removed artifact files again matched their saved sizes and hashes. The checklist now separates experiment 6 execution from its failed scientific acceptance criterion, reducing the count from 53 to 52 without withdrawing any completed implementation. No additional training or outcome-driven tuning was performed.

The owner supplied the WiDS dictionary on 2026-10-20. Its 188 variable definitions confirm APACHE IVa, but leave negative scores and score availability unresolved. A newly identified conflict between the dictionary's micromol/L creatinine unit and the frozen rule's mg/dL assumption also requires publisher clarification. Existing numerical experiments remain reproducible, but clinical rule interpretation and external unit harmonization remain unverified. See [source evidence](source-evidence.md); the source file and its audit stay in ignored private storage. The checklist remains 52/72.

## 11. Technical release with validation deferred

The technical scope includes a frozen 36-fit recipe, protected separate-root regeneration, a source-only reproduction ZIP with per-file integrity hashes, complete-study inventory enforcement and a reproduction runbook. Exact checkpoint/RNG replay was rerun for all 36 fits with maximum per-pass probability residual zero. All 297 preserved generated artifacts again match their archive manifest. Dependency consistency and 89 regression tests pass.

The creatinine rule-off sensitivity reuses the original 50-pass draw from each of 20 hospital fits. Every result agrees with its existing individual-rule control within 1e-12 Brier and produces zero threshold changes at 0.5. Paired hospital/seed intervals for rule-off-minus-MC Brier span zero in every condition. This is an exploratory unit-sensitive exclusion check, not a choice of corrected units or confirmation data. [Sensitivity report](../results/research_checks/creatinine_sensitivity.json).

The paper now centers current development results. Its attention-variance initializer and statistical interval descriptions match the implemented pathways. Confounded historical tables and case examples are retained in `paper/legacy_results.tex` and excluded from the compiled manuscript. The source-unit conflict, heuristic grounding, shared evidence and deferred clinical/human validation remain explicit. [Reproduction instructions](reproducibility.md) distinguish source packaging, data access and exact fitted replay. Scientific completion remains 52/72; deferring validation does not change the failed or unverified criteria.

The [technical verification record](../results/research_checks/technical_release.json) records the completed release checks. All 89 tests also passed from a separately extracted source package using the same installed environment; this verifies packaging completeness, not a fresh environment or independent scientific validation.

## 12. Full-plan continuation

The original scientific criteria remain unmet where indicated; the technical release does not replace those criteria. Rechecking the reviewer preparation exposed a predictable public case/fault sequence and shared cross-format task catalog. Newly generated materials now randomize faults, use opaque case IDs and isolate each reviewer's assigned format; the private CLI seals master tasks and grading templates separately. A fresh 24-reviewer/80-case package is locally generated and its 1,920 balanced assignments are verified, with no responses collected. [Preparation verification](../results/research_checks/reviewer_preparation.json) contains no private key or answers.

The publisher GOSSIS extraction-code audit pins seven source files and confirms an extraction/definition discrepancy relevant to the WiDS creatinine unit. It does not certify the exact production release or resolve score availability. All 1,554 public TCGA XML supplements are now acquired and checksum/case-identity verified across 507 cases. The 94-path inventory confirms additional administrative and specimen dates but does not establish selected-input availability or gene callability. These are additional source checks, not clinical sign-off or a favorable scientific result.

The full public XML audit is complete with zero unavailable files in the pinned snapshot. The fresh private reviewer package contains 78 local files; only its non-answer preparation checks are committed. The regression suite now passes 93 tests. [Full-plan status](../results/research_checks/full_plan_remaining.json) maps the 20 unsatisfied checklist entries to their overlapping scientific, source and human-evidence dependencies, without redefining the plan or treating the technical release as plan completion.

The WiDS recursive download is not certified complete: its 212,974-byte overview PDF fails the publisher SHA256 and Poppler parsing. The dictionary and original training data remain independently inspected. Unauthenticated official download returned HTTP 403; signed-in browser retrieval timed out. The partial source remains untouched and is excluded from scientific evidence. See [download integrity](../results/research_checks/wids_download_integrity.json).

## 13. Source requests and partial-download verification

The two prepared GDC/GOSSIS source requests were sent on 2026-10-20 after a duplicate check. Sent copies were read back, including the GDC public case/gene attachment inventory. Mail identifiers stay private. No reply or new authoritative clarification is claimed; source-provenance criteria remain unsatisfied.

The repeatable download verifier detects the incomplete overview and three truncated data files under the recursive private download. The original 64,747,327-byte study source independently matches the frozen recipe SHA256, and the dictionary matches its inspected copy. No downloaded file was altered. Run `python scripts/verify_wids_download.py`; a failed check returns exit code 1. Publisher hashes and size-only checks are reported separately, without certifying unlisted files. The regression suite now contains 98 tests.

## 14. Owner exclusions and official-source search

The owner explicitly removed external researcher-dependent clinical rule review and the participant study from active work. Four corresponding checklist entries are struck through and tagged excluded, not completed. The original plan remains 52/72 scientifically satisfied; active scope is 52 satisfied and 16 unresolved out of 68. `docs/scope-policy.json` preserves the decision, and the checklist generator retains it across updates. No clinical or human-benefit claims change.

The search identifies 515 open THCA pathology PDFs totaling 79,840,335 bytes as additional obtainable source material to inspect. Public MAF masking documentation confirms why missing variant rows cannot establish negatives. Publisher GOSSIS code and issue evidence support unavailable-value handling but do not certify exact WiDS release provenance or score timing. SICdb and HiRID remain gated cohort candidates rather than acquired independent evidence. See `results/research_checks/source_followup.json`. The current regression suite contains 100 tests.

## 15. Existing-data WiDS completion and encoder extension

The owner explicitly excluded further WiDS source-interpretation checks as execution prerequisites on 2026-10-21 and stopped the unused recursive download. The original study data and frozen conventions are retained. Exact score timing and source-unit discrepancies remain limitations, without being used to hold up available-data experiments. No new patient dataset was downloaded. The original plan still has 52 satisfied criteria; five owner exclusions leave 15 active unresolved criteria out of 67.

The previously unexecuted WiDS encoder comparison is now evaluated on all 20 saved checkpoints: four imputation/APACHE conditions with five seeds each, 50 original matched dropout passes, no new training. Each uniform-encoder control reproduces every original probability pass with maximum residual zero. Both encoder variants are retained as private probability arrays. Direct float64 Brier recomputation from all 40 saved encoder variants has maximum residual zero against the reported metrics; contrast means also match per-seed metric differences. A regression test protects tiny paired differences against float32 loss-subtraction error.

The encoder NARS-minus-MC Brier point estimates range from -1.87e-6 to -2.89e-6, but all four Bonferroni family intervals include zero and the 0.0001 benefit margin is not established. Twelve fixed condition/contrast comparisons receive family intervals; resampling uses hospitals and fitted seeds and fixes the primary dropout draw. These are exploratory results on inspected development cases, not a new untouched confirmation or a replacement for the negative readout findings. The aggregate report is `results/research_checks/wids_encoder.json`; the full WiDS summary and generator are `docs/wids-results.md` and `scripts/summarize_wids.py`.

The regression suite now contains 103 tests. The paper includes the four-condition encoder table with its conditional uncertainty scope. Private arrays, source data and the original untracked research archive remain outside Git. Internet-source options are documented separately, including the Zigong credentialing/DUA route and why SUPPORT2 would be an additional benchmark rather than a first-day WiDS substitute.


## 16. Immediately accessible alternative dataset search

On 2026-10-21, the official Vanderbilt SUPPORT2 ZIP downloaded without credentials. Archive integrity, header inspection and record counting establish 9,105 records and 47 columns. No outcome values or predictive metrics were inspected. The source grants dataset use with citation and acknowledgement. Its mostly study-day-three assessments and absent hospital/patient identity columns prevent treating it as the plan's verified frozen first-day WiDS external cohort; it is an additional benchmark candidate.

The public CC BY 4.0 Nepal 17-ICU release has accessible metadata and a listed 22.6 MB workbook, but direct and normal-page downloads did not produce a usable file here. Its schema, outcome equivalence, timing and source independence remain unestablished. Mendeley AFICILL/Dresden API requests also failed with HTTP 403. [Access evidence](../results/research_checks/open_dataset_access.json) records successes and failures; the acquisition report preserves the earlier failed attempts, superseded by the successful local preparation and evaluation below. Raw files stay ignored. No training, new evaluation, paper result or checklist completion is claimed. The count remains 52 satisfied, five owner-excluded and 15 active unresolved criteria.


The owner subsequently supplied the Nepal workbook locally. Its size and MD5 match the publisher and its ZIP CRC check passes. Read-only structural inspection finds 77,610 rows, 67 columns and 18,603 distinct `patient_number` values, with repeated daily records. Admission physiology headers are available, but only ICU discharge status is explicit; hospital death, hospital identifiers, exact measurement units/windows and patient-versus-encounter identity remain unestablished. No outcome values or predictive metrics were inspected. The access report and dataset guide now mark acquisition resolved; no additional owner download is required. Scientific completion counts remain unchanged.


## 17. Public pathology source audit

All 515 pinned public THCA pathology PDFs totaling 79,840,335 bytes across 507 cases are now acquired, publisher size/MD5 verified and processed with native `pdftotext`. Initial connection resets were retried successfully; no files remain unavailable. The inventory records SHA256 hashes and native-text diagnostics without publishing report text. All five numeric-date candidate reports were read: one tissue-section-range false positive, one explicit procurement date and three unlabeled annotations. This agent source-text review is not clinical expert review, OCR accuracy certification, selected-input episode linkage or an availability timestamp audit. No complete gene callability is established. [Inventory](../results/research_checks/tcga_pathology.json), [candidate review](../results/research_checks/tcga_pathology_date_review.json). The corresponding full scientific criteria remain unchecked.


The pathology follow-up prepares an ignored, line-numbered source queue for all 46 reports with molecular keywords. Aggregate scope flags find 22 hotspot/codon, eight pending-test, four indeterminate/poor-quality and seven prior-specimen candidates, with overlap. These keyword flags are not verified sample-linked results. They demonstrate why a mention or a negative hotspot test cannot become a whole-gene negative call for the selected sequencing specimen. No assay manifest was modified and zero whole-gene negative calls were derived. [Follow-up evidence](../results/research_checks/tcga_pathology_molecular.json).


## 18. Completed Nepal changed-endpoint experiment

The pinned Nepal workbook is now prepared and evaluated, rather than left as an access recommendation. Source-linked CCAA dictionary acquisition and an outcome-independent scale check exposed conflicts between the generic Fahrenheit/creatinine definitions and many exported values; temperature, creatinine and WBC stay unavailable. Selecting one physical row after strict admission/outcome consistency checks yields 18,111 recorded adults and 3,316 ICU deaths from 18,603 source admission identifiers. The 492 age-ineligible records and all unsupported inputs are explicitly reported. No unknown outcome becomes survival.

Ten frozen no-APACHE WiDS checkpoints are evaluated under demographic and admission-core mappings, producing twenty 50-pass evaluations without training, recalibration or outcome-driven mapping selection. Every prediction/event bundle passes independent persisted replay. Saved-probability metric recomputation has maximum residual 1.65e-14 and uniform-gate identity maximum residual 1.65e-8, within the 1e-7 criterion. A duplicate-row preprocessing shortcut failed a source-data equivalence check and was removed; rejected outputs/cache are preserved privately and excluded. Checked whole-transform caching preserves the original KNN batching/tie arithmetic.

All four four-comparison-adjusted Brier intervals favor small NARS-minus-MC reductions, with means from -1.36e-6 to -1.62e-7. These conditional numerical effects remain far below the 0.0001 benefit margin. Mean ungated AUROC is 0.565/0.560 for demographic median/KNN and 0.621/0.610 for admission-core median/KNN. Changed ICU versus hospital mortality, first measurements versus extrema, unsupported inputs and absent hospital/canonical-person linkage prevent independent clinical compatibility claims. The analysis does not estimate institution-level or repeat-person dependence, and the primary dropout draw is fixed. The original development failures remain unchanged.

Fixed pre-existing age/sex strata and research decision thresholds generate 400 private subgroup metric rows across all twenty evaluations; public diagnostics contain whole-cohort seed-mean decision curves. These are additional descriptive diagnostics, not a registered clinical subgroup analysis or verified clinical thresholds. [Full results](nepal-results.md), [aggregate evidence](../results/research_checks/nepal_sensitivity.json), [diagnostics](../results/research_checks/nepal_diagnostics.json), [source evidence](../results/research_checks/nepal_source_evidence.json).

The repository suite now passes 115 tests. All 297 previously removed artifacts were rechecked against saved sizes/SHA256 and remain preserved outside Git. The updated paper compiles to 14 pages without undefined references or overfull boxes; pages 9-12 were rendered and inspected. The checklist remains 52 satisfied, five owner-excluded and 15 active unresolved criteria. The source preparation, experiments and additional public-source audit do not establish the missing scientific acceptance or eligibility criteria.


All 115 tests also pass from a separately extracted, integrity-verified 169-file source-only package using the same installed environment. This checks packaging completeness rather than a fresh environment or independent scientific validation. Final packaging includes the updated verification notes; datasets, fitted arrays, private subgroup details, molecular-source extracts and rejected experiment outputs remain excluded.

## 19. Zigong unavailable; current evidence remains the paper scope

The owner reports that Zigong cannot be accessed. No exact cause was supplied and no new credential or network probe was performed. Earlier Zigong access instructions are removed from active next steps; prepared purpose text remains preserved. No further owner download or researcher recruitment is requested. The current paper and source package report the completed internal studies and the PhysioNet/Nepal sensitivities, with matched independent hospital-mortality validation explicitly unestablished. SUPPORT2 remains an optional separate study-day-three benchmark, not a substitute for that criterion.

This access update does not change scientific completion: 52 of 72 satisfied, five owner-excluded and 15 active unresolved. A documentation consistency check also found stale checklist verification prose naming 103 tests and a 13-page paper. The checklist generator now reflects the existing verified release record of 115 tests and 14 pages, without claiming a new test or paper run.

## 20. Research improvements and merge verification

The README now distinguishes corrected implementation and stronger evidence from unestablished clinical benefit. The paper abstract includes the Nepal changed-endpoint study; its corrections, conclusion and reproducibility disclosure explain why matched inference, explicit unknown calls, observed-only predicates and complete replay improve research validity without establishing superiority. The revised PDF remains 14 pages, with no undefined references or overfull boxes; updated pages 1 and 10-12 were rendered and inspected.

The current local suite passes all 115 tests in 30.28 seconds. GitHub reports PR #7 clean and mergeable against main, with no CI checks configured in the reported check rollup. Both branch histories exclude all 297 removed artifact paths; the simulated merge is conflict-free and excludes those paths and private data roots. These checks support merging a research implementation with its limitations, not declaring the full scientific plan complete. Keep local fitted research files until an approved independent backup exists, and never restore contaminated history. GitHub-side purge remains a separate pending support operation. [Merge evidence](../results/research_checks/merge_readiness.json) pins the checked base/head and updated document hashes; recheck mergeability if either branch changes.

## 21. Retired planning and handoff material

At the owner's request, the maintained source tree no longer contains the original plan/checklist, overlapping access/next-step guides, outreach/action-kit drafts, or the two checklist generators. All sixteen removed files were copied to an ignored local archive before removal; the separate owner action kit and all private research files remain untouched. The final machine-readable completion and remaining-evidence snapshots remain as scientific evidence, with the original 52 satisfied, five excluded and 15 active unresolved criteria unchanged. Earlier sections are a chronological record, not current instructions to retrieve removed documents.

Three checklist-generator tests were retired with their implementation. All remaining 112 tests pass in the repository and in an independently extracted, integrity-verified 154-file source package using the same installed environment. No model or experiment test was removed. Maintained README/doc Markdown links resolve, the package no longer expects retired planning files, and the updated paper remains 14 pages with its current test count corrected and page 12 visually inspected. Negative findings, source evidence, scientific limitations and reproducibility instructions remain maintained.

## 22. Owner-selected preprint template and authorship

The manuscript now uses George Kour's arxiv-style template linked by the owner-selected Overleaf page. `paper/template-source.json` pins upstream revision, source URLs and SHA256 for the unmodified style and its MIT license. The anonymous workshop header and unused NeurIPS style/example/checklist are retired; formatting files remain inside `paper/`. The author block and PDF metadata identify Arjun Vijay Prakash, Independent Researcher, with the supplied high-school description and email. This change does not imply institutional review or endorsement.

The manuscript body is byte-for-byte unchanged after normalizing the bibliography-style switch from plainnat to the template's unsrtnat. The new PDF has 13 pages, three figures and six tables, with no undefined references or overfull boxes. All pages were rendered and reviewed, and author details were checked in extracted text and PDF metadata. Scientific results, missing evidence and claim boundaries remain unchanged.
