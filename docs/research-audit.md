# NGTA research audit

NGTA is an inference-time interface for uncertainty-conditioned symbolic intervention in a tabular transformer. **The plan is not complete:** the corrected core is implemented and replayable, but clinical symbolic benefit, population robustness, external compatibility and reviewer usefulness have not been established.

This document follows the review and ranked experiment structure in [plan.md](../plan.md). [list.md](../list.md) is the task checklist; `results/v2_validation/documentation_audit.json` is its machine-readable evidence matrix. The original plan is preserved as the historical review, so its descriptions of v1 bugs must not be read as descriptions of the current code.

## 1. Audit scope and current evidence

Audit date: 2026-10-03, branch `version-2`. This audit reads the plan, all 72 checklist tasks, implementation modules, tests, source audits and saved experiment bundles. It reruns the regression suite and persisted replay. It does not run new clinical training, recruit reviewers, acquire another cohort or independently verify every bibliography entry.

The 47 regression tests pass. All six saved TCGA bundles and the available synthetic WiDS bundle pass independent replay. The prior pre-push check also restored all six TCGA checkpoints, fitted preprocessors and MC RNG states on the original GPU and reproduced saved MC probabilities exactly; prediction CSVs, split hashes and local source manifests matched. The current audit repeats source-hash and replay checks; checkpoint restoration is identified as prior evidence rather than a new audit result.

| Evidence | What exists | What it establishes |
|---|---|---|
| Actual TCGA source audit | 457 labeled cases; two training cases with recorded variants, zero validation/test cases; no complete verified panel; 5,170 conflicting case/field records | Local lineage and missingness handling, not held-out genomic fusion or verified chronology |
| Actual WiDS source audit | 91,713 labeled stays; hospital-disjoint case manifests with zero patient/hospital/ICU overlap and no duplicate selected-feature rows; 2,371 invalid APACHE probabilities | Split integrity and measured data quality, not a trained hospital-confirmation result |
| TCGA integration | One two-epoch run, three MC passes, standard baselines, readout/encoder controls, two ensemble members, raw masking; 74 events, 24 unmapped | Integrated execution and complete numerical event replay |
| TCGA repeated integration | Five two-epoch training seeds, identical split IDs, three MC passes, actual seed variability | Reproducibility of the short development run, not the plan's five-seed clinical confirmation |
| Synthetic WiDS integration | 256 generated rows, 15 predictor variants, 40 masking comparisons, 65 replayed events, two ensemble members | Pipeline compatibility, not mortality-prediction evidence; the fixture is a local ignored test artifact |
| Symbolic negative control | TCGA smoke Brier gain over MC is about -8.51e-10; 100 permutations, Holm-adjusted p=1 | No demonstrated symbolic improvement in this run |
| Legacy v1 results | `results/tcga`, `results/wids`, `results/submission`, manuscript tables/figures | Historical results with aggregation, imputed-trigger and circular-audit limitations; not current evidence |

WiDS selected-feature missingness is 10.3559% after cleaning; lactate missingness is 74.5761%. TCGA clinical missingness is 5.1229%; overall missingness is 75.7601% when the mostly unavailable genomic panel is included. These denominators describe different feature sets and do not establish robustness under imposed missingness.

## 2. Claim-to-evidence matrix

| Claim from the research plan | Current finding | Boundary |
|---|---|---|
| Genomic fusion/intervention works on held-out TCGA | Unsupported: no recorded held-out variants or verified complete panels | Withdraw the held-out multimodal demonstration |
| Confidence gating improves clinical prediction | Matched inference is implemented; clinical confirmation is absent | Do not carry the legacy aggregation-confounded gain into current |
| NARS adds value beyond MC-only or simple confidence boosts | Controls run; the smoke result does not meet the gain/significance criteria | Implementation of revision does not prove symbolic necessity |
| Rules and neural outputs are independent evidence | Both use the same measurements; revised frequency is logged but does not drive the confidence gate | Treat the mapping as heuristic, not validated evidential semantics |
| Every corrected intervention is auditable | Complete events, independent predicates/revision/gate reconstruction and hash checks pass on available bundles | Numerical replay verifies exported computation, not clinical rule correctness |
| NGTA improves human oversight | No participant study exists | Describe inspectable traces only |
| Robustness, institutional generalization or external noninferiority is established | Raw masking code and internal hospital audits exist; the required studies do not | Keep all population/generalization claims open |

## 3. Corrected implementation and remaining risks

### Data construction and timing

TCGA now chooses one most-complete primary physical record per case within each source table instead of assembling a synthetic record column by column. Conflict exports preserve repeated-field disagreements. Cross-table chronology remains unverified, and the task is retrospective post-pathology association rather than preoperative prediction.

Sparse-column removal, constant removal and mutation-panel selection fit training data only. Numeric KNN distances use training-standardized features. Unavailable mutation calls stay unknown, with separate missing indicators. Recorded variants are positive; an unrecorded variant becomes negative only with explicit callable-coverage provenance.

An optional `data/assay_manifest.csv` contains `case_submitter_id,gene,source,verified`, with unique case/gene pairs. `verified` is `true` or `false`; verified rows need a source documenting callable coverage. File size, MAF presence and absence of variant rows do not establish negative coverage. The supplied cohort cannot support the plan's clinical-only/genomic-only/fused comparison on verified held-out assays.

WiDS defaults to patient-disjoint splitting and offers hospital-disjoint or explicit legacy row splitting. Group IDs must be present and each partition must contain both outcomes. Features represent the first ICU day, so admission-time prediction claims would require another feature window. APACHE values outside [0,1] become unavailable; the underlying sentinel definitions and score availability time are still unverified. Rules use observed measurements; imputed values and suppressed imputed-only triggers are recorded separately.

### Prediction aggregation and symbolic semantics

Every dropout pass preserves attention, token scores, CLS logits, native logits, probabilities and RNG states. Baseline, uniform, MC-confidence-only, NARS and readout ablations reuse identical passes and average probabilities after sigmoid. Uniform normalized gating must agree with the ungated readout within 1e-7. Deterministic and mean-logit inference are separately named controls.

The default gate changes the final attention-weighted token-score readout. It does not erase a feature's earlier influence on contextual tokens or the ungated CLS contribution. `--encoder-intervention` additionally biases feature keys inside every attention layer and recomputes representations using the same dropout RNG states. Identity and context-change tests pass, but the stronger intervention has no confirmed clinical advantage.

Neural frequency is attention whereas symbolic frequency describes a proposition. The predictive gate consumes revised confidence, not revised frequency. Direct confidence boosts and fixed-confidence frequency changes expose this limitation. Eight prototype rules have sources, version IDs and contradiction groups, but all are `not_reviewed`; same-target collision checks do not resolve correlated clinical evidence across different features. This interface is not a complete NARS architecture.

### Statistics, replay and reproducibility

Metrics include AUROC, PR-AUC, Brier, log loss, calibration slope/intercept, fixed/quantile ECE and sensitivity/specificity/PPV/NPV at 0.1, 0.2 and 0.5. Calibration slope/intercept fit on test outcomes are diagnostics only and never recalibrate predictions. Reliability plots and metric bins agree at boundaries. Invalid labels or probabilities are rejected before conversion/clipping.

WiDS paired comparisons resample whole hospitals with common resamples across methods. Observed deltas are distinct from bootstrap means, and Bonferroni family intervals cover comparator/metric pairs; symbolic permutation diagnostics use Holm correction. Per-model bootstrap intervals do not capture all training, split and MC-sampling variation. Risk/degradation acceptance intervals and the external clustered noninferiority procedure are still missing.

Every trigger, including unmapped triggers, is saved with identity/version/source, raw and imputed values, observed status and symbolic truths. Mapped events additionally carry neural/revised truths, before/after attention and rule-off probabilities. Unmapped events have no invented attention effect. Independent replay reconstructs predicates, revision, gates, predictions and counterfactual effects, checks event completeness, finite numerical fields and hashes, and rejects corrupted events even if their hashes are recomputed. Optional sampled case summaries do not replace complete event exports.

Evaluation specifications record configuration, rule definitions, source hashes, split hashes, thresholds and margins. `--evaluation-lock` rejects mismatches before training or artifact writes. A lock file is not prospective preregistration and cannot make already examined outcomes unseen. Requirements remain lower bounds rather than a fully pinned portable environment; run summaries record the versions actually used.

## 4. Ranked experiment plan and acceptance audit

These are the research criteria proposed in `plan.md`, not validated clinical thresholds. No experiment is marked clinically accepted merely because its code runs.

| Experiment | Implemented/evidenced | Required acceptance and remaining work |
|---|---|---|
| 1. Valid coverage and replay | Local hashes, disjoint manifests, missingness indicators, training-only selection and persisted replay | Zero unexplained overlap or missing-to-negative conversion; complete documented lineage. Callable genomic coverage and source timing remain unverified, so genomic claims remain blocked |
| 2. Aggregation versus gating | Matched caches; uniform identity <=1e-7 across five short TCGA seeds | Five training seeds on prospectively locked held-out hospitals, 50 MC passes; corrected 95% interval favoring gating and Brier gain >=1e-4. This study has not run |
| 3. Clinical rule content | MC-only, confidence boost, removal, random truths, frequency controls and 100 prevalence-preserving permutations | On the same locked cohort, Brier gain >=1e-4 over MC-only and adjusted p<0.05 against random controls. Current smoke evidence fails to establish this |
| 4. Extraction and complete replay | Extension/staging fixes, selected missing/negative/boundary tests; all available current events replay <=1e-7 | Exhaustive predicate boundary/unknown/conflict suite, zero false/missed triggers, no undocumented imputed observations, and full clinical observed/imputed comparison. The exhaustive suite/comparison remain unfinished |
| 5. APACHE dependence/baselines | Raw/APACHE-only/recalibrated score, logistic, boosting, deterministic/mean-logit and no-score options | Document score semantics; five paired with/without-score seeds (about ten transformer fits). Brier gain >=1e-4 over best locked comparator with favorable 95% interval. Full runs are undone |
| 6. Missingness/uncertainty | Raw random/feature-dependent 0/10/30/50/70% masks, MC/entropy/ensemble support and selective risk | Outcome-dependent simulations, imputer comparison, five-model ensemble, above-chance error detection and paired degradation intervals. Routing must improve integrated degradation and not worsen any mask level by >0.001; this has not been tested |
| 7. External compatibility | No external cohort evaluation | Freeze mapped units/windows and parameters before external labels; clustered upper Brier-difference bound <0.001 and lower AUROC-difference bound >-0.01. Cohort, mapping and analysis are absent |
| 8. Human auditability benefit | Numerical trace instrumentation only | Prospective powered randomized blinded crossover with seeded faults and mixed-effects analysis; localization gain >=10 percentage points with favorable 95% interval, false reassurance increase <=5 points. Study tooling and participants are absent |

## 5. Every checklist task: completion and reason

`Completed` means the named correction or audit has evidence, not that NGTA's clinical hypothesis succeeds. `Partial` means some required work exists but the whole task is unfinished. `Not started` means its study/analysis has not been built or run. External dependencies and ordinary unfinished implementation/execution are identified separately in the JSON audit. The eight experiments and five priorities overlap the earlier tasks; the counts below are checklist counts, not independent research milestones.

Audit totals: **37 completed, 31 partial, 4 not started**, across 72 checklist entries.

The audit checks the stronger-baseline implementation and aggregation priority as done, and reopens exhaustive rule extraction. Clinical-study items remain unchecked even when their tooling is ready.

### 1. Core Research and Methodology


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Prove the symbolic component matters | Partial | Removal, shuffling, random truth, fixed-prior and 100-permutation controls run in smoke bundles. Symbolic superiority is not established; the TCGA smoke gain over MC is negative and adjusted p=1. Evidence: [symbolic_ablations.py](../src/symbolic_ablations.py), [symbolic_hypothesis_tests.json](../results/v2_smoke/tcga/metrics/symbolic_hypothesis_tests.json). |
| Strengthen uncertainty estimation | Partial | MC, entropy, ensemble and error diagnostics exist. Only two ensemble members were exercised in integration; five-model, five-seed clinical evaluation and independent dropout repeats remain undone. Evidence: [uncertainty.py](../src/uncertainty.py), [robustness.py](../src/robustness.py). |
| Improve the rule base | Partial | Registries record prototype sources, versions, review status and contradiction groups. All eight rules are not_reviewed; clinical expert validation and a defensible evidence-generation process are absent. Evidence: [knowledge_base.py](../src/knowledge_base.py), [wids_knowledge_base.py](../src/wids_knowledge_base.py). |
| Add external validation | Partial | Frozen synthetic masking exists, but no eligible independent cohort, harmonized feature map or external predictions are supplied. Separate TCGA and WiDS training is not external validation. Evidence: [robustness.py](../src/robustness.py), [v2_validation](../results/v2_validation). |
| Tighten the scientific framing | Completed | README and manuscript restrict the contribution to an inspectable inference interface and preserve unproven clinical claims as limitations. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |

### 2. Data Validity and Preprocessing


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Fix TCGA genomic coverage | Partial | Unknown calls retain indicators and negatives require verified assay provenance. Available variants cover two training cases and zero held-out cases; no complete callable panel is verified. Evidence: [data_loader.py](../src/data_loader.py), [data_quality.json](../results/v2_validation/tcga/traces/data_quality.json). |
| Correct feature selection leakage | Completed | Sparse/constant filtering and genomic-panel selection fit training data only. Evidence: [data_loader.py](../src/data_loader.py), [test_data_validity.py](../tests/test_data_validity.py). |
| Audit TCGA record merging | Partial | One physical record per source replaces column-wise mixing, and 5,170 conflicting case/field records are exported. Cross-table chronology and measurement availability need source timing metadata. Evidence: [data_loader.py](../src/data_loader.py), [record_conflicts.csv](../results/v2_validation/tcga/traces/record_conflicts.csv). |
| Clarify prediction timing | Completed | TCGA is explicitly retrospective post-pathology association; WiDS uses end-of-first-day measurements. These landmarks do not verify every source timestamp. Evidence: [data_loader.py](../src/data_loader.py), [wids_loader.py](../src/wids_loader.py), [main.tex](../paper/main.tex). |
| Audit WiDS missingness | Completed | The real-source audit exports feature missingness: 10.3559% overall after cleaning and 74.5761% lactate missingness. No broad extreme-missingness conclusion is drawn. Evidence: [missingness.csv](../results/v2_validation/wids/traces/missingness.csv). |
| Check APACHE data | Partial | All 2,371 invalid APACHE probabilities are masked and no-APACHE mode exists. Authoritative sentinel/availability documentation and paired full-data no-APACHE experiments remain absent. Evidence: [wids_loader.py](../src/wids_loader.py), [apache_baselines.py](../src/apache_baselines.py). |
| Review imputation | Completed | Training-standardized KNN distances, separate raw/imputed values and observed-only rules are implemented and regression-tested. Comparative imputer research is still separate work. Evidence: [test_data_validity.py](../tests/test_data_validity.py), [wids_loader.py](../src/wids_loader.py). |
| Verify data integrity | Completed | Local source hashes, exact case splits, assay status, duplicate checks, overlap counts and conflicts are persisted. This verifies local lineage, not complete assay acquisition. Evidence: [data_quality.py](../src/data_quality.py), [v2_validation](../results/v2_validation). |
| Use stronger data splits | Partial | Patient/hospital split modes exist; the real WiDS hospital audit has zero patient/hospital/ICU overlap. Hospital-confirmation training and TCGA verified genomic coverage are still missing. Evidence: [wids_loader.py](../src/wids_loader.py), [data_quality.json](../results/v2_validation/wids/traces/data_quality.json). |

### 3. Prediction and Calibration


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Fix the aggregation mismatch | Completed | All readout gates use the same cached dropout passes and mean per-pass probabilities; uniform identity is checked at 1e-7. Evidence: [matched_inference.py](../src/matched_inference.py), [test_matched_inference.py](../tests/test_matched_inference.py). |
| Add matched inference baselines | Completed | Ungated, uniform, MC-only and NARS predictions are matched and saved in all corrected smoke bundles. Evidence: [inference_cache.npz](../results/v2_multiseed_smoke/seed_0/tcga/traces/inference_cache.npz). |
| Add stronger predictive baselines | Completed | Deterministic, mean-logit, calibrated logistic, validation-selected boosting and APACHE comparators are implemented and exercised in TCGA or synthetic WiDS integration. Full clinical comparison remains item 18. Evidence: [pipeline.py](../src/pipeline.py), [apache_baselines.py](../src/apache_baselines.py), [integration_checks.json](../results/v2_validation/integration_checks.json). |
| Test APACHE dependence | Partial | APACHE-only and recalibrated baselines ran on synthetic WiDS. Five paired full WiDS runs with and without APACHE have not been executed. Evidence: [apache_baselines.py](../src/apache_baselines.py), [main.py](../main.py). |
| Clarify the symbolic mechanism | Completed | The gate uses revised confidence only. Frequency sensitivity and direct confidence-boost controls expose that NAL frequency does not determine predictive direction or establish independent evidence. Evidence: [symbolic_ablations.py](../src/symbolic_ablations.py), [main.tex](../paper/main.tex). |
| Test encoder-level intervention | Completed | Encoder intervention biases keys in every attention layer and recomputes contextual scores using replayed RNG. Identity/context changes are tested and integration outputs exist; clinical advantage is untested. Evidence: [test_matched_inference.py](../tests/test_matched_inference.py), [encoder_nars_confidence.npz](../results/v2_smoke/tcga/traces/encoder_nars_confidence.npz). |
| Evaluate calibration properly | Completed | AUROC, PR-AUC, Brier, log loss, fixed/quantile ECE, calibration slope/intercept and threshold metrics are implemented. Invalid inputs and bin boundaries are regression-tested. Evidence: [evaluation.py](../src/evaluation.py), [test_evaluation.py](../tests/test_evaluation.py). |
| Correct statistical comparisons | Completed | Paired hospital resampling, observed versus bootstrap deltas and Bonferroni comparator/metric intervals are implemented. These are per-fit intervals, not a full hierarchy of seed/split/MC uncertainty. Evidence: [pipeline.py](../src/pipeline.py), [test_evaluation.py](../tests/test_evaluation.py). |
| Preserve negative results | Completed | Legacy null/losing comparisons remain qualified; smoke permutation tests show no symbolic gain. No failed result is relabeled as clinical success. Evidence: [main.tex](../paper/main.tex), [symbolic_hypothesis_tests.json](../results/v2_smoke/tcga/metrics/symbolic_hypothesis_tests.json). |

### 4. Rule System and Symbolic Intervention


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Expand rule validation | Partial | Prototype provenance and target-collision checks exist. Expert source review, correlated-evidence/contradictory clinical reasoning and validated observation confidence are not complete. Evidence: [knowledge_base.py](../src/knowledge_base.py), [wids_knowledge_base.py](../src/wids_knowledge_base.py). |
| Test rule necessity | Completed | All-rule/individual-rule removal and prevalence-preserving shuffled predicates are evaluated from matched caches. This tests necessity in integration; it does not prove clinical necessity. Evidence: [symbolic_ablations.py](../src/symbolic_ablations.py), [symbolic_isolation.csv](../results/v2_smoke/tcga/metrics/symbolic_isolation.csv). |
| Separate observed and imputed triggers | Completed | Observed-only WiDS rules suppress imputed-only events; synthetic masking exports an explicitly labeled imputed-rule comparator. Evidence: [wids_loader.py](../src/wids_loader.py), [robustness.py](../src/robustness.py), [test_data_validity.py](../tests/test_data_validity.py). |
| Test truth-value sensitivity | Completed | Frequency, confidence, randomized-truth and fixed-prior controls rerun revision rather than replace its output. Evidence: [symbolic_ablations.py](../src/symbolic_ablations.py), [pipeline.py](../src/pipeline.py). |
| Fix rule extraction | Partial | Extension-mask and unknown-stage bugs are fixed, and missing/negative/boundary probes pass. The exhaustive per-rule boundary/unknown/conflict matrix requested by the plan is not implemented for every predicate. Evidence: [test_data_validity.py](../tests/test_data_validity.py), [test_trace_replay.py](../tests/test_trace_replay.py). |
| Check multimodal rule behavior | Partial | Synthetic BRAF triggering works, but no real held-out genomic rule event can be demonstrated with the current coverage. Evidence: [test_trace_replay.py](../tests/test_trace_replay.py), [data_quality.json](../results/v2_validation/tcga/traces/data_quality.json). |
| Test rule-conditioned confidence boosts | Completed | The direct confidence-boost comparator is in the symbolic isolation output; frequency-independent confidence equivalence is disclosed. Evidence: [symbolic_ablations.py](../src/symbolic_ablations.py), [symbolic_isolation.csv](../results/v2_smoke/tcga/metrics/symbolic_isolation.csv). |

### 5. Auditability and Trace Exports


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Make audit checks independent | Completed | Replay uses separately implemented predicates and closed-form revision, rather than calling production revision or gate code. Evidence: [trace_replay.py](../src/trace_replay.py), [auditability.py](../src/auditability.py). |
| Improve trace completeness | Completed | Every trigger is persisted, including 24 unmapped events among 74 TCGA smoke events; mapping failure is explicit. Evidence: [intervention_events.csv](../results/v2_smoke/tcga/traces/intervention_events.csv). |
| Record raw and imputed values | Completed | Events include raw/imputed values, observed status, source/version, truth values and mapped rule-off effects. Unmapped events carry no fabricated numerical intervention. Evidence: [trace_replay.py](../src/trace_replay.py), [test_trace_replay.py](../tests/test_trace_replay.py). |
| Fix incomplete exports | Completed | Complete events are unconditional; optional sampled summaries are supplementary. Evidence: [pipeline.py](../src/pipeline.py), [trace_replay.py](../src/trace_replay.py). |
| Validate end-to-end replay | Completed | All six corrected TCGA bundles plus the available synthetic WiDS bundle pass independent replay. Corruption, missing events and nonfinite event fields are tested. Evidence: [pre_push_checks.json](../results/v2_validation/pre_push_checks.json), [test_trace_replay.py](../tests/test_trace_replay.py). |
| Evaluate human oversight | Not started | No reviewer participants, power analysis, seeded-fault study package, randomized assignments or responses exist. Human benefit cannot be inferred from numerical replay. Evidence: [plan.md](../plan.md). |

### 6. Robustness and Generalization


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Test missingness robustness | Partial | Raw 0/10/30/50/70% random and feature-dependent masks ran in integration. Outcome-dependent simulations, imputer comparisons and paired degradation acceptance intervals remain absent. Evidence: [robustness.py](../src/robustness.py), [raw_missingness.csv](../results/v2_smoke/tcga/metrics/raw_missingness.csv). |
| Compare uncertainty methods | Partial | MC/entropy diagnostics and ensemble support exist, but a five-model clinical ensemble and five-seed confidence-error study have not run. Evidence: [uncertainty.py](../src/uncertainty.py), [robustness.py](../src/robustness.py). |
| Measure selective prediction | Partial | Error-detection AUROC, tie-aware risk curves and per-mask Brier outputs exist. Locked clinical results and uncertainty intervals for degradation/selective risk are absent. Evidence: [robustness.py](../src/robustness.py), [selective_risk.csv](../results/v2_smoke/tcga/metrics/selective_risk.csv). |
| Test institutional generalization | Partial | Internal hospital-disjoint lineage is audited, but full held-out-hospital training has not run and no independent institution cohort is available. Evidence: [data_quality.json](../results/v2_validation/wids/traces/data_quality.json). |
| Check external compatibility | Not started | No external cohort, frozen unit/window mapping or clustered noninferiority analysis is implemented. The proposed Brier/AUROC acceptance margins are not tested. Evidence: [plan.md](../plan.md). |
| Evaluate subgroups | Partial | Fixed demographic strata and decision curves include MC-only and exist in smoke outputs. Locked hospital clinical subgroup evaluation is undone. Evidence: [subgroups.py](../src/subgroups.py), [subgroup_metrics.csv](../results/v2_smoke/tcga/metrics/subgroup_metrics.csv). |

### 7. Reproducibility and Implementation


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Run multiple seeds | Partial | Five TCGA models use identical splits and export actual variability, but each trains only two epochs with three MC passes. Five-seed 50-pass hospital confirmation is absent. Evidence: [multiseed_metrics.csv](../results/v2_multiseed_smoke/submission/multiseed_metrics.csv). |
| Fix stale outputs | Partial | New outputs are separated and mixed historical/current figures are rejected. Legacy paper figures/tables remain, PDF is stale, and figure selection can prefer unseeded results over newer seed folders. Evidence: [paper_figures.py](../src/paper_figures.py), [manuscript_build.json](../results/v2_validation/manuscript_build.json). |
| Correct ablation implementations | Completed | Confidence sensitivity reruns production revision on matched cached passes. Evidence: [pipeline.py](../src/pipeline.py), [symbolic_ablations.py](../src/symbolic_ablations.py). |
| Prevent test-set tuning | Partial | Evaluation-spec comparison occurs before artifact writes and training. A file lock is not preregistration; already inspected labels cannot become unseen, and no prospective locked confirmation has run. Evidence: [pipeline.py](../src/pipeline.py), [test_artifact_exports.py](../tests/test_artifact_exports.py). |
| Improve artifact preservation | Completed | Checkpoints, fitted processors/estimators, source hashes, exact splits, per-pass outputs, RNG and dependency versions are saved; six checkpoints reproduced MC predictions exactly on the original GPU. Evidence: [pre_push_checks.json](../results/v2_validation/pre_push_checks.json). |
| Make data acquisition reproducible | Partial | Local manifests are hashed, but immutable acquisition IDs/version/checksum pinning and complete cohort coverage are not enforced by the downloader. Verified callable genomic records are missing. Evidence: [gdc_downloader.py](../src/gdc_downloader.py), [data_quality.py](../src/data_quality.py). |
| Audit references | Partial | The identified Chudasama metadata error is fixed; 28 citation keys resolve and no duplicate keys exist. The remaining cited authors/titles/years/venues/pages/DOIs have not been verified individually. Evidence: [reference_audit.json](../results/v2_validation/reference_audit.json). |

### 8. Claims to Weaken or Withdraw


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Withdraw the held-out genomic demonstration | Completed | Held-out genomic feasibility and genomic intervention claims are withdrawn. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Restrict extreme-missingness claims | Completed | Measured feature-specific missingness is separated from unproven population robustness. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Qualify confidence-routing claims | Completed | Legacy calibration differences are attributed to confounded inference; corrected matched smoke runs establish identity, not clinical gating benefit. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Withdraw universal trace-completeness claims | Completed | Universal completeness is withdrawn for legacy artifacts; current event completeness is supported by persisted replay. Evidence: [main.tex](../paper/main.tex), [trace_replay.py](../src/trace_replay.py). |
| Clarify revision validation | Completed | Legacy production reuse is labeled a consistency check; current independent arithmetic is described separately. Evidence: [main.tex](../paper/main.tex), [auditability.py](../src/auditability.py). |
| Qualify trust-signal claims | Completed | Confidence remains a heuristic attention-stability signal, not clinically calibrated epistemic trust. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Avoid equivalence claims | Completed | Close point estimates are not presented as equivalence or noninferiority. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Qualify preprocessing claims | Completed | Legacy pre-split selection is disclosed; current selection is training-only. Evidence: [main.tex](../paper/main.tex), [data_loader.py](../src/data_loader.py). |
| Limit human-oversight claims | Completed | Inspectability and numerical replay are separated from unmeasured reviewer benefit. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |
| Separate symbolic effects from inference effects | Completed | Matched MC-only is the symbolic comparator; baseline-to-NARS changes are not attributed wholly to symbolic evidence. Evidence: [main.tex](../paper/main.tex), [pipeline.py](../src/pipeline.py). |

### 9. Prioritized Experiment Plan


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| Experiment 1: Establish valid data coverage and replay | Partial | Local lineage, missingness and replay pass. Verified TCGA callable coverage and source-level clinical timing remain unresolved, so experiment 1 is not fully accepted. Evidence: [v2_validation](../results/v2_validation). |
| Experiment 2: Isolate aggregation from gating | Partial | Uniform identity and matched gates pass across five smoke seeds. The five-seed locked hospital 50-pass study and corrected benefit margin test are missing. Evidence: [v2_multiseed_smoke](../results/v2_multiseed_smoke). |
| Experiment 3: Test clinical rule value | Partial | All principal symbolic controls and 100 permutations run in smoke evaluation. The clinical acceptance margin and adjusted significance are not met or confirmed. Evidence: [symbolic_hypothesis_tests.json](../results/v2_smoke/tcga/metrics/symbolic_hypothesis_tests.json). |
| Experiment 4: Validate extraction and trace replay | Partial | Persisted replay and current regression probes pass, but the complete predicate boundary matrix and locked clinical observed/imputed comparison are incomplete. Evidence: [test_trace_replay.py](../tests/test_trace_replay.py), [test_data_validity.py](../tests/test_data_validity.py). |
| Experiment 5: Challenge APACHE dependence | Partial | Stronger baseline code and synthetic APACHE checks exist. Score documentation and ten full transformer fits for five paired with/without-score seeds are absent. Evidence: [apache_baselines.py](../src/apache_baselines.py), [pipeline.py](../src/pipeline.py). |
| Experiment 6: Test missingness and uncertainty | Partial | Two masking scenarios, risk diagnostics and two-member integration exist. Outcome-dependent masks, imputation controls, five-member clinical ensemble and acceptance intervals remain undone. Evidence: [robustness.py](../src/robustness.py), [uncertainty.py](../src/uncertainty.py). |
| Experiment 7: Test external compatibility | Not started | External harmonization, frozen evaluation and clustered noninferiority must be built once an eligible independent cohort is identified. Evidence: [plan.md](../plan.md). |
| Experiment 8: Evaluate auditability | Not started | The prospective reviewer study, power analysis, fault package and mixed-effects analysis are not built or conducted. Evidence: [plan.md](../plan.md). |

### 10. Immediate Priorities


| Task | Status | Evidence, remaining work and reason |
|---|---|---|
| First: Fix data validity | Partial | Missingness handling and training-only selection are fixed; genomic coverage and cross-source chronology cannot be established from supplied files. Evidence: [data_quality.json](../results/v2_validation/tcga/traces/data_quality.json). |
| Second: Match prediction aggregation | Completed | Prediction aggregation and dropout samples are matched; the scientific benefit question remains experiment 2. Evidence: [matched_inference.py](../src/matched_inference.py), [test_matched_inference.py](../tests/test_matched_inference.py). |
| Third: Make traces independently replayable | Partial | Circularity and persisted completeness are fixed; exhaustive extraction edge-case coverage is still incomplete under the full combined priority. Evidence: [trace_replay.py](../src/trace_replay.py), [test_data_validity.py](../tests/test_data_validity.py). |
| Fourth: Strengthen baselines and validation | Partial | Baseline/control/robustness tooling exists, but clinical runs, remaining planned controls and external evaluation remain open. Evidence: [pipeline.py](../src/pipeline.py), [robustness.py](../src/robustness.py). |
| Fifth: Rewrite unsupported claims | Completed | The source manuscript and documentation preserve legacy negative evidence and distinguish implemented mechanics from unproven research claims. Evidence: [main.tex](../paper/main.tex), [README.md](../README.md). |

## 6. Additional plan requirements not fully represented by the old checklist

The broader review includes work that the old checklist did not spell out. These items are unfinished implementation or study design, not missing-data excuses.

| Requirement | Current gap |
|---|---|
| Transformer probability recalibration | No standalone validation-only recalibration comparator for transformer probabilities; test calibration slope is only a diagnostic |
| Clinical-only/genomic-only/fused TCGA models | No dedicated controlled modality comparison; meaningful evaluation also needs verified held-out genomic coverage |
| Confidence-assignment permutations and irrelevant predicates | No separately prespecified confidence-only permutation/irrelevant-predicate suite; case-shuffled predicate controls exist |
| Training-label shuffle | No full training-label-shuffle negative-control experiment |
| Repeat dropout separately from training seeds | RNG replay exists, but MC-draw stability is not independently replicated or incorporated into uncertainty intervals |
| Outcome-dependent masking and imputation alternatives | Masking currently supports random/feature-dependent scenarios and the fitted KNN preprocessor, not the plan's full scenario/imputer comparison |
| Robustness acceptance analysis | Per-mask losses/risk curves exist, but no integrated-degradation paired cluster interval or every-mask harm-margin acceptance test |
| External noninferiority and frozen feature mapping | No independent-cohort mapping/eligibility package or clustered noninferiority analysis |
| Reviewer study tooling | No power analysis, seeded-fault case package, blinded randomization or mixed-effects analysis |
| Figure freshness/selection | Mixed schemas are rejected, but unseeded folders can still take precedence and figures summarize selected runs rather than fully aggregated seed evidence |
| Portable acquisition/environment | Source hashes and runtime versions exist, but acquisition identities/checksums and dependencies are not completely pinned for a fresh reproducible installation |
| Complete citation verification | Key existence/uniqueness and one metadata correction are complete; all cited publication metadata are not independently verified |

The original plan's review also flags class-weighting differences between random forests and unweighted neural loss, saturated feature confidence and correlated stage/extension evidence. These are disclosed design risks; no new experiment has eliminated them.

## 7. Why unfinished tasks remain unfinished

**Missing inputs or people:** Verified TCGA callable coverage, source-level timing and authoritative APACHE sentinel/availability information are not present. No eligible independent cohort or clinical expert/reviewer responses are supplied. I cannot invent those measurements, provenance or participant outcomes. External data acquisition could resolve some dependencies, but an eligible cohort must first be identified and its units, outcome/window and access conditions checked.

**Work that can still be done here:** Full WiDS confirmation, paired APACHE removal, five-member ensembles and clinical subgroup/robustness evaluation have not run. Training-standardized KNN on the full cohort is expensive, but expense is not proof that execution is impossible. Remaining controls, exhaustive extraction tests, external analysis scaffolding, reviewer-study tooling, figure selection and complete citation verification are unfinished work. They were not completed in the earlier implementation pass; they must not be described as externally blocked.

**Prospective design:** The local sources and legacy test outcomes have already been inspected. A configuration lock preserves a design but cannot undo that exposure. Confirmatory evaluation needs a defensible previously unexamined partition/cohort and a protocol frozen before the new analysis, rather than treating old test labels as new evidence.

**PDF tool failure:** MiKTeX reported incomplete setup; bundled Tectonic execution was denied, including after escalation. `paper/main.tex` is preserved, while `paper/main.pdf` remains stale. This is the recorded host-tool blocker in `results/v2_validation/manuscript_build.json`; no successful rebuild is claimed.

## 8. Run and reproduce

Use the existing Python environment or install `requirements.txt` into a separate environment. The recorded run dependencies include joblib through scikit-learn. Choose a fresh output directory so development and legacy artifacts do not mix.

```powershell
# Audit real sources without neural training or full WiDS KNN fitting.
.\venv\Scripts\python.exe main.py --run-all --audit-data --split-mode hospital --output-dir results/v2_audit

# Development runs; these commands do not by themselves establish confirmation.
.\venv\Scripts\python.exe main.py --dataset tcga --baseline-set standard --ablation-set submission --skip-paper-figures --output-dir results/v2_dev
.\venv\Scripts\python.exe main.py --dataset wids --split-mode hospital --baseline-set standard --ablation-set submission --skip-paper-figures --output-dir results/v2_dev

# Candidate five-seed study execution, after prospectively fixing its protocol.
.\venv\Scripts\python.exe main.py --dataset wids --split-mode hospital --seeds 0 1 2 3 4 --split-seed 0 --mc-samples 50 --ensemble-size 5 --shift-eval --encoder-intervention --baseline-set standard --ablation-set submission --skip-paper-figures --output-dir results/v2_study

# Independently replay a saved bundle and run the regression suite.
.\venv\Scripts\python.exe -m src.trace_replay results/v2_smoke/tcga/traces
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/readme-audit
```

| Option | Behavior |
|---|---|
| `--without-apache` | Excludes the score from feature-based models; score-only comparators remain separate. Run paired seeds/output directories to compare inclusion/removal |
| `--seeds 0 1 2 3 4` | Trains separate models and exports actual seed variability; one run leaves seed standard deviation undefined |
| `--split-seed 0` | Holds cohort partitions and masking draws fixed across training seeds |
| `--ensemble-size 5` | Fits/saves five ensemble members per pipeline invocation; using this with five training seeds trains an ensemble within each run, so compute grows accordingly |
| `--shift-eval` | Runs raw observed-value random/feature-dependent masks at 0/10/30/50/70% through frozen preprocessing and rules; does not implement outcome-dependent masks |
| `--encoder-intervention` | Saves encoder-key intervention separately from matched readout gating |
| `--evaluation-lock PATH` | Validates config, rules, sources and split IDs against an existing specification before training/exports; supports `{seed}` and `{dataset}` path placeholders |
| `--export-case-traces` | Adds sampled case summaries; full intervention export is always enabled |
| `--paper-tables` | Writes tables from selected run summaries; this does not replace or rebuild the manuscript automatically |
| `--skip-paper-figures` | Skips automatic figures; otherwise new figures go into the selected output directory's `paper_figures` |

The supplied default epochs/patience and a lock file are not a substitute for prospective protocol selection. Gamma sweeps and rule sensitivity consume test labels as development diagnostics; do not choose the confirmatory setting from their test performance.

## 9. Repository and artifact map

| Location | Contents and scope |
|---|---|
| `plan.md` | Original research review, proposed experiment designs and acceptance criteria |
| `list.md` | Evidence-qualified completion checklist, with reasons for every open task |
| `src/data_loader.py`, `src/wids_loader.py`, `src/data_quality.py` | Cohort construction, training-only preprocessing and persisted source/split audits |
| `src/neural_encoder.py`, `src/matched_inference.py`, `src/attention_hook.py`, `src/nars_interface.py` | Transformer, matched gates and heuristic truth/revision mechanics |
| `src/knowledge_base.py`, `src/wids_knowledge_base.py`, `src/symbolic_ablations.py` | Prototype registries, extraction and symbolic controls |
| `src/trace_replay.py`, `src/auditability.py` | Complete event export and independent numerical checks |
| `src/evaluation.py`, `src/apache_baselines.py`, `src/robustness.py`, `src/uncertainty.py`, `src/subgroups.py` | Metrics, score comparators, raw masking, uncertainty and subgroup diagnostics |
| `src/pipeline.py`, `main.py`, `src/paper_figures.py` | Experiment orchestration, multi-seed exports and figure generation |
| `results/v2_validation` | Real source audits, integration history, pre-push checks, reference/PDF status and this documentation audit |
| `results/v2_smoke`, `results/v2_multiseed_smoke` | Corrected short TCGA execution and five-seed integration evidence |
| `.test-tmp/wids-output` | Local ignored synthetic WiDS integration; not a tracked clinical result |
| `results/tcga`, `results/wids`, `results/submission` | Archived v1 numerical bundles; do not combine with current metrics |
| `paper/main.tex`, `paper/references.bib`, `paper/figures` | Current manuscript source, bibliography and explicitly legacy paper figures/tables |
| `paper/neurips_2026.sty`, `paper/neurips_2026.tex`, `paper/checklist.tex` | NeurIPS style, formatting sample and checklist, all kept inside `paper/` |
| `paper/main.pdf` | Stale compiled manuscript; current TeX source has not compiled successfully on this host |

Each new run saves `model.pt`, `preprocessor.joblib`, classical estimators and `evaluation_spec.json`. `traces/` contains raw test cases, exact split IDs, preprocessing metadata, source/data-quality reports, mutation assay status where applicable, MC RNG and inference cache, complete events, hashes and replay results. `metrics/` holds predictions' summary metrics, run/environment metadata, calibration/subgroup curves, bootstrap comparisons, symbolic controls and optional uncertainty/masking outputs. `charts/` holds that run's plots; optional ensemble and encoder checkpoints/passes remain separate.

The current source manuscript qualifies legacy aggregation differences, imputed triggers, production-reused audits, absent held-out genomic measurements, unproven equivalence and untested human benefit. Its bibliography correction is documented in `reference_audit.json`; that file explicitly says the full metadata audit is incomplete. New scientific conclusions require the studies above, not a documentation completion mark.
