# NGTA: Complete List of Edits and Changes

## 1. Core Research and Methodology

- [ ] **Prove the symbolic component matters.** Add ablations that isolate NARS revision from confidence gating, including rule removal, rule shuffling, random truth values, and fixed symbolic priors.
- [ ] **Strengthen uncertainty estimation.** Compare MC dropout with deep ensembles and other uncertainty methods across multiple random seeds.
- [ ] **Improve the rule base.** Develop a principled rule-generation and validation process with clinical sources, expert review, provenance, contradiction handling, and sensitivity analysis.
- [ ] **Add external validation.** Test on independent cohorts or distribution-shift settings without retuning rules or models.
- [x] **Tighten the scientific framing.** Present NGTA as an auditable inference-time interface for uncertainty-conditioned symbolic intervention, not as a proven clinical performance improvement.

## 2. Data Validity and Preprocessing

- [ ] **Fix TCGA genomic coverage.** Separate unavailable mutation data from confirmed mutation absence and verify assay coverage before making multimodal claims.
- [x] **Correct feature selection leakage.** Move sparse-column filtering and mutation-panel selection inside the training split.
- [ ] **Audit TCGA record merging.** Check whether collapsed records combine measurements from different times and whether predictors are available at the intended prediction time.
- [x] **Clarify prediction timing.** Define the prediction landmark and assess whether pathological stage, extension, and residual disease are valid inputs.
- [x] **Audit WiDS missingness.** Document feature-level missingness and avoid broad claims of extreme missingness based on one highly incomplete feature.
- [ ] **Check APACHE data.** Resolve negative mortality probability values, document score provenance and availability, and test the model without APACHE.
- [x] **Review imputation.** Evaluate KNN distance weighting, distinguish observed from imputed values, and check whether imputed values trigger symbolic rules.
- [x] **Verify data integrity.** Create case and assay manifests, check source hashes, verify duplicate and overlap counts, and preserve exact data lineage.
- [ ] **Use stronger data splits.** Introduce patient-disjoint and hospital-disjoint evaluation, with verified genomic coverage for TCGA.

## 3. Prediction and Calibration

- [x] **Fix the aggregation mismatch.** Ensure baseline and gated models use identical prediction aggregation and dropout samples.
- [x] **Add matched inference baselines.** Compare ungated, uniform-gate, MC-confidence-only, and NARS-gated predictions under identical aggregation.
- [ ] **Add stronger predictive baselines.** Include deterministic inference, mean-logit inference, calibrated logistic regression, tuned gradient boosting, and recalibrated APACHE.
- [ ] **Test APACHE dependence.** Compare APACHE-only models and transformer variants with and without APACHE to establish whether NGTA adds predictive value.
- [x] **Clarify the symbolic mechanism.** Investigate whether NAL frequency semantics contribute anything beyond a rule-conditioned confidence boost.
- [x] **Test encoder-level intervention.** Compare current readout gating with methods that actually intervene in encoder attention and contextual representations.
- [x] **Evaluate calibration properly.** Report Brier score, log loss, AUROC, PR-AUC, calibration slope and intercept, multiple ECE definitions, and threshold-specific metrics.
- [x] **Correct statistical comparisons.** Use paired hospital-clustered bootstrap intervals, control for multiple comparisons, and distinguish observed effect sizes from bootstrap means.
- [x] **Preserve negative results.** Report cases where MC-only performs better, gating reduces AUC, or comparisons remain inconclusive.

## 4. Rule System and Symbolic Intervention

- [ ] **Expand rule validation.** Document clinical sources, rule provenance, expert review, contradictory rules, and sensitivity to rule choices.
- [x] **Test rule necessity.** Disable individual rules and all rules, shuffle predicates, and compare correct rules against matched random controls.
- [x] **Separate observed and imputed triggers.** Measure how much symbolic intervention depends on imputed rather than directly observed clinical values.
- [x] **Test truth-value sensitivity.** Vary symbolic confidence and frequency while correctly rerunning the revision process.
- [x] **Fix rule extraction.** Test missing, unknown, negative, boundary, unseen-category, and conflicting inputs, including the extension-category trigger-mask bug.
- [ ] **Check multimodal rule behavior.** Verify that genomic rules actually fire on patients with documented genomic measurements.
- [x] **Test rule-conditioned confidence boosts.** Compare the NARS mechanism against simpler alternatives that do not use NAL frequency calculations.

## 5. Auditability and Trace Exports

- [x] **Make audit checks independent.** Implement separate revision arithmetic rather than using the same production function for both actual and expected values.
- [x] **Improve trace completeness.** Export every intervention with rule identifiers, symbolic and neural truth values, revised frequencies, attention effects, and provenance.
- [x] **Record raw and imputed values.** Include raw measurements, imputation status, rule versions, and counterfactual rule-off effects in event traces.
- [x] **Fix incomplete exports.** Remove dependence on optional sampled case traces and ensure all events can be reconstructed from persisted files.
- [x] **Validate end-to-end replay.** Independently reconstruct rule triggers, truth-value revisions, gates, and predictions from saved artifacts.
- [ ] **Evaluate human oversight.** Compare NGTA traces with ordinary structured logs in a blinded fault-localization study to test whether they actually help reviewers.

## 6. Robustness and Generalization

- [ ] **Test missingness robustness.** Evaluate additional masking at 0%, 10%, 30%, 50%, and 70%, including random and feature-dependent missingness scenarios.
- [ ] **Compare uncertainty methods.** Evaluate MC dropout against a five-model ensemble and test whether confidence predicts errors.
- [ ] **Measure selective prediction.** Report error-detection AUROC, selective-risk curves, and Brier degradation under increasing missingness.
- [ ] **Test institutional generalization.** Evaluate on held-out hospitals and independent institutions without retuning rules, thresholds, or preprocessing.
- [ ] **Check external compatibility.** Compare frozen NGTA and MC-only predictions on independent cohorts using prespecified calibration and noninferiority criteria.
- [ ] **Evaluate subgroups.** Report subgroup metrics and clinically relevant decision curves under locked evaluation settings.

## 7. Reproducibility and Implementation

- [ ] **Run multiple seeds.** Replace single-seed evidence with repeated training runs and report actual variability.
- [ ] **Fix stale outputs.** Synchronize submission metrics, prediction exports, case traces, and figures with the current experiment results.
- [x] **Correct ablation implementations.** Fix rule-confidence sensitivity analysis so it reruns revision rather than substituting a confidence value.
- [ ] **Prevent test-set tuning.** Restrict hyperparameter and rule selection to training and validation data, with a new locked evaluation for confirmation.
- [x] **Improve artifact preservation.** Save model checkpoints, fitted preprocessing objects, per-pass MC outputs, exact split IDs, and dependency versions.
- [ ] **Make data acquisition reproducible.** Add immutable dataset manifests and hashes, and verify cohort coverage rather than relying on file size.
- [ ] **Audit references.** Correct the identified bibliographic metadata error and conduct a complete citation audit.

## 8. Claims to Weaken or Withdraw

- [x] **Withdraw the held-out genomic demonstration.** The current TCGA test split does not establish genomic fusion or genomic intervention.
- [x] **Restrict extreme-missingness claims.** Describe measured feature-specific missingness until controlled masking experiments establish robustness.
- [x] **Qualify confidence-routing claims.** Attribute current differences to the overall inference procedure until aggregation is matched.
- [x] **Withdraw universal trace-completeness claims.** Retain them only after every intervention is exported and independently replayable.
- [x] **Clarify revision validation.** Describe the existing audit as a consistency check using production arithmetic.
- [x] **Qualify trust-signal claims.** Treat the signal as heuristic until its relationship with errors and uncertainty is validated.
- [x] **Avoid equivalence claims.** Describe close point estimates on internal splits without claiming formal equivalence or noninferiority.
- [x] **Qualify preprocessing claims.** Acknowledge that some feature selection occurs before splitting.
- [x] **Limit human-oversight claims.** Describe current instrumentation as inspectable numerical traces until independent replay and user evaluation are complete.
- [x] **Separate symbolic effects from inference effects.** Do not attribute baseline-to-NARS prediction changes solely to symbolic intervention.

## 9. Prioritized Experiment Plan

- [ ] **Experiment 1: Establish valid data coverage and replay.** Verify missingness, assay coverage, feature lineage, and reproducibility before further claims.
- [ ] **Experiment 2: Isolate aggregation from gating.** Compare all gate variants under matched aggregation and confirm results across five training seeds.
- [ ] **Experiment 3: Test clinical rule value.** Compare NARS against MC-only, confidence boosts, rule removal, and 100 prevalence-preserving predicate permutations.
- [ ] **Experiment 4: Validate extraction and trace replay.** Test rule correctness, missingness handling, independent revision, and complete event reconstruction.
- [ ] **Experiment 5: Challenge APACHE dependence.** Compare against APACHE-only, recalibrated APACHE, logistic regression, boosting, and transformer variants without APACHE.
- [ ] **Experiment 6: Test missingness and uncertainty.** Evaluate controlled masking, confidence-error relationships, selective prediction, and uncertainty estimation.
- [ ] **Experiment 7: Test external compatibility.** Evaluate frozen NGTA on independent cohorts without retuning.
- [ ] **Experiment 8: Evaluate auditability.** Run a blinded reviewer study comparing NGTA traces with standard structured logs.

## 10. Immediate Priorities

- [ ] **First: Fix data validity.** Resolve TCGA genomic coverage, missingness handling, feature selection, and prediction-time concerns.
- [ ] **Second: Match prediction aggregation.** Isolate the actual effect of confidence gating from changes in prediction computation.
- [ ] **Third: Make traces independently replayable.** Fix audit circularity, export completeness, and rule extraction errors.
- [ ] **Fourth: Strengthen baselines and validation.** Test APACHE dependence, rule necessity, uncertainty quality, robustness, and external generalization.
- [x] **Fifth: Rewrite unsupported claims.** Keep negative findings intact and make every scientific claim match the evidence currently available.
## v2 implementation evidence

- Data selection now fits training only. Genomic unknowns retain missing indicators; negative calls require a sourced, verified case/gene assay manifest. The available cohort still has no recorded held-out variants, so verified multimodal evaluation remains open.
- TCGA selects one physical record per source table and exports repeated-field conflicts. Cross-table chronology is unverified; the defined task is retrospective post-pathology association.
- WiDS defaults to patient-disjoint partitions, offers hospital-disjoint evaluation, standardizes KNN distances, masks invalid APACHE probabilities, and suppresses imputed-only rule triggers. Full robustness and APACHE-dependence experiments remain open.
- Regression checks: `tests/test_data_validity.py`. Real TCGA lineage audit: `results/v2_validation/tcga/traces/data_quality.json`.

- Matched inference uses cached per-pass attention, token scores, and CLS logits for ungated, uniform, MC-only, NARS, and symbolic controls. Uniform gating is checked against ungated predictions. Deterministic and mean-logit controls are separately labelled; revision audit arithmetic has its own closed-form implementation. Checks: `tests/test_matched_inference.py`.

- Every triggered rule now exports its id, source, version, raw value, observed status, truth values, attention effect, and rule-off probability. Unmapped triggers remain explicit events. Replay independently checks predicates, revisions, gates, predictions, counterfactuals, completeness, and hashes from persisted files. `tests/test_trace_replay.py` includes deletion and corruption tests.
- Calibration includes log loss, PR-AUC, slope/intercept diagnostics, fixed and quantile ECE, and locked threshold metrics. Hospital-clustered paired bootstrap reports observed deltas separately from bootstrap means and adds Bonferroni family intervals. `tests/test_evaluation.py` verifies whole-cluster resampling.
- Models, fitted preprocessors, classical estimators, exact split IDs, source hashes, dependency versions, and per-pass MC outputs are preserved. The two-epoch TCGA smoke run is implementation validation, not a new clinical result.

- Completion audit reopened the first four research claims: controls and ensemble/shift tooling exist, but symbolic superiority, a full clinical ensemble study, expert rule review, and independent-cohort validation have not been established. Synthetic checks and a training smoke run do not satisfy those requirements.
- Real WiDS audit now persists case IDs, raw feature missingness, hashes, and patient/hospital/ICU overlap under the hospital-disjoint design. The measured selected-feature missing fraction is 10.36% after masking invalid APACHE scores. No overlap or duplicate selected-feature rows were found.
- APACHE-only logistic and logit recalibration comparators are fitted on training outcomes; the no-APACHE model option is available. Score sentinel documentation and a full paired hospital experiment remain outstanding. Raw missingness evaluation reprocesses 0/10/30/50/70% masks with frozen state; its integration is tested, but population robustness is not yet established.

- Encoder intervention now applies confidence to feature keys inside every attention layer and recomputes contextual representations, while replaying the original dropout RNG states. The readout and encoder variants are saved separately. Identity and context-change checks: `tests/test_matched_inference.py`.

- README and manuscript now distinguish corrected v2 implementation from legacy v1 numerical tables. Legacy aggregation, imputed triggers, circular revision checks, absent held-out genomic measurements, and untested human benefits are stated explicitly. The identified Chudasama bibliography page/author/DOI error was corrected against the authors’ institutional record; a complete citation audit is still open.

- Final validation: 34 tests pass. The integrated real TCGA run independently replays all 74 events, including 24 unmapped events, and preserves standard baselines, two ensemble members, encoder comparisons, masking outputs, and negative permutation results. The synthetic WiDS integration independently replays 65 events across 15 variants and 40 masking comparisons.
- Five two-epoch TCGA runs use identical split IDs, pass replay for every seed, and export actual seed variability under `results/v2_multiseed_smoke`. These short runs validate reproducibility, not the full clinical confirmation protocol. `results/v2_validation/integration_checks.json` records their scope.
- Manuscript PDF rebuilding remains blocked by incomplete MiKTeX setup and an inaccessible bundled Tectonic executable. The corrected source is preserved; `results/v2_validation/manuscript_build.json` explicitly records that the existing PDF is stale.
