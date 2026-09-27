# MY OTHER AI AGENT:

## NGTA Research Review and Recommended Improvements

#### Overview

NGTA, or NARS-Guided Transformer Attention for Clinical Transformers Under Extreme Missingness, proposes a neurosymbolic interface that combines Transformer predictions, MC-dropout uncertainty, NAL-style truth values, and explicit clinical rules. The central idea is to convert neural uncertainty into an evidential representation, revise that representation using symbolic rules, and feed the revised confidence back into attention during inference. The system also records rule triggers, truth values, revisions, and attention changes, making the intervention auditable.

The project has a real methodological contribution. It is not simply a Transformer with post-hoc explanations. The symbolic layer changes the inference computation itself, and the implementation explicitly distinguishes the interface from a complete NARS architecture.

#### Current Evidence

The WiDS experiment is the strongest part of the study. It contains 91,713 ICU stays, and the symbolic pathway is active on 8,551 of 13,757 held-out cases, producing 13,031 feature-level revisions. The system also passes operational audit checks, with complete rule-to-feature mapping and zero recomputation residuals.

However, the central scientific claim is not yet isolated. NARS-gated routing improves Brier score and ECE relative to the ungated Transformer, but comparisons against flat-confidence and MC-confidence-only controls include zero. MC-confidence-only also achieves slightly better Brier and ECE point estimates. Therefore, the current experiments demonstrate that confidence-based routing works and that the symbolic mechanism is operational, but they do not demonstrate that **NARS revision itself causes the improvement**.

#### Major Things to Fix

**1. Prove the symbolic component matters.**
This is the biggest issue. Add stronger ablations that isolate symbolic revision from confidence gating. Useful experiments include disabling rules, shuffling rules between patients, replacing symbolic truth values with random values, and comparing different fixed symbolic priors. A convincing result would show a benefit that disappears when the symbolic evidence is removed or randomized.

**2. Strengthen uncertainty estimation.**
The neural-to-NAL mapping is explicitly heuristic. MC-dropout variance may underestimate uncertainty under distribution shift. Compare MC dropout against deep ensembles and other uncertainty estimators, ideally across multiple random seeds.

**3. Improve the rule base.**
The current four-rule-per-dataset design is useful as a prototype but too small to support broad clinical claims. Build a principled rule-generation and validation process with documented clinical sources, expert review, provenance, contradictory-rule handling, and sensitivity analysis.

**4. Add external validation.**
WiDS currently uses an internal split, while TCGA has only 69 held-out cases. Test the method on an independent cohort or a distribution-shift setting without retuning the rules or model.

**5. Tighten the scientific framing.**
Do not sell NGTA as a clinical predictor that “improves performance.” The strongest defensible contribution right now is an auditable inference-time interface for uncertainty-conditioned symbolic intervention. The paper already recognizes this distinction, and preserving it will make the work more credible.

#### Overall Assessment

NGTA has a strong core idea and unusually good methodological honesty. Its biggest weakness is not lack of sophistication. It is that the experiments currently prove **the mechanism operates**, rather than proving **the mechanism is scientifically necessary or superior**. The next version should focus less on adding more datasets for appearance and more on isolating causality, strengthening uncertainty estimation, validating the rule methodology, and testing generalization. If those experiments succeed, NGTA could move from an interesting prototype into a much more substantial research contribution.

---

# GPT 6 ASTRA

INFERRED: The other agent’s review is directionally sensible but too favorable. I would recommend rejection in the current form because the repository reveals validity problems that go beyond insufficient ablations: the genomic evaluation lacks genomic observations in its held-out sets, the principal calibration comparison changes prediction aggregation, and the auditability evidence is partly circular and incompletely exported.

OBSERVED: I made no edits and ran no training. I read the manuscript source and compiled PDF, README, source modules, embedded configurations, result bundles, and data files; recomputed metrics from saved predictions; reconstructed data splits; and ran a small in-memory rule-extraction check. No notebooks or standalone data dictionary were present. Existing working-tree changes remained untouched.

**Labels:** OBSERVED means directly checked in artifacts or read-only calculations. INFERRED means an interpretation or plausible risk. PROPOSED means work not yet performed.

#### 1. Claim-to-evidence matrix

Quoted claims below come from [the manuscript](/C:/Users/USER/Desktop/code/NGTA/paper/main.tex).

| Major claim | Exact supporting artifact or implementation | Audit finding |
|---|---|---|
| “TCGA-THCA tests the mechanism under clinical-genomic feature fusion” | [TCGA loader](/C:/Users/USER/Desktop/code/NGTA/src/data_loader.py:334), both MAF files, reconstructed split | **OBSERVED:** The MAFs contain 5 and 82 records, each for one patient. Only 2/457 labeled patients have nonzero mutation inputs, both in training. Validation and test mutation matrices are entirely zero. **INFERRED:** Held-out multimodal feasibility is not demonstrated. |
| “The processed input has 105 dimensions” | [TCGA preprocessing metadata](/C:/Users/USER/Desktop/code/NGTA/results/tcga/traces/preprocessing_metadata.json) | **OBSERVED:** Supported structurally: 5 numeric, 50 mutation, 50 categorical dimensions. Dimension count conceals the absence of genomic observations for nearly the entire cohort. |
| “primary validation under scale and extreme missingness” | [WiDS loader](/C:/Users/USER/Desktop/code/NGTA/src/wids_loader.py:22), raw CSV | **OBSERVED:** The selected 15 raw features have 10.18% missing cells overall; lactate has 74.58% missingness. **INFERRED:** One highly incomplete feature is established; robustness under extreme missingness is not. |
| “91,713 labeled stays split into 64,199 training, 13,757 validation, and 13,757 held-out test stays” | [WiDS split summary](/C:/Users/USER/Desktop/code/NGTA/results/wids/traces/split_summary.json), reconstructed split | **OBSERVED:** Supported. Test contains 1,187 deaths. |
| “maps Monte Carlo dropout uncertainty into…truth values” | [attention initializer](/C:/Users/USER/Desktop/code/NGTA/src/attention_hook.py:18), [inference pipeline](/C:/Users/USER/Desktop/code/NGTA/src/pipeline.py:1025) | **OBSERVED:** Routing uses attention means and attention variances. Patient-level predictive probability and variance are converted separately for reporting but do not determine the gate. |
| “revised confidence gates attention” | [gate](/C:/Users/USER/Desktop/code/NGTA/src/attention_hook.py:32), [model readout](/C:/Users/USER/Desktop/code/NGTA/src/neural_encoder.py:89) | **OBSERVED:** Supported for the final attention-weighted token-score readout. Encoder attention and contextual representations are not recomputed after revision. |
| “flat-confidence…isolates the architectural effect of adding a gate” | [prediction recomputation](/C:/Users/USER/Desktop/code/NGTA/src/pipeline.py:1033) | **OBSERVED:** A uniform normalized gate cancels mathematically. Predictions differ because baseline and flat control aggregate dropout outputs differently. **INFERRED:** This interpretation is incorrect. |
| “ICU rules fired on 8,551…stays, with 13,031 feature-level revisions” | [WiDS audit metrics](/C:/Users/USER/Desktop/code/NGTA/results/wids/metrics/auditability_metrics.json), [prediction export](/C:/Users/USER/Desktop/code/NGTA/results/wids/traces/test_predictions.csv) | **OBSERVED:** Counts agree with saved confidence changes. Of 1,936 lactate revisions, 1,271 occur where raw lactate is missing. |
| “symbolic interventions execute on both modalities” | [TCGA run summary](/C:/Users/USER/Desktop/code/NGTA/results/tcga/metrics/run_summary.json) | **OBSERVED:** BRAF triggers: zero. The 79 TCGA events comprise age, stage, and extrathyroid extension. **INFERRED:** Genomic symbolic intervention is not demonstrated. |
| “Every intervention exports its trigger, neural and symbolic truth values, revision, and attention effect” | [trace builder](/C:/Users/USER/Desktop/code/NGTA/src/pipeline.py:690), [optional case export](/C:/Users/USER/Desktop/code/NGTA/src/pipeline.py:889) | **OBSERVED:** The default full export omits per-event rule identifiers, symbolic truth values, and revised frequencies. Detailed case export is optional and selects at most 20 cases. Current runs have `export_case_traces=False`. The universal export claim is unsupported. |
| “Independently recomputed revision…residuals were zero” | [audit implementation](/C:/Users/USER/Desktop/code/NGTA/src/auditability.py:87) | **OBSERVED:** Revision is checked by calling the same `revise_truth_values` function used in production. **INFERRED:** This is a consistency check, not independent validation of revision correctness. |
| “paired Brier and ECE differences favor NARS-gated routing over the ungated transformer” | [WiDS metrics](/C:/Users/USER/Desktop/code/NGTA/results/wids/metrics/metrics.csv), run-summary paired bootstrap | **OBSERVED:** Supported for this saved run, conditional on its aggregation mismatch and statistical procedure. It does not identify a gating effect. |
| “comparisons with flat-confidence and MC-confidence-only gating include zero” | WiDS and TCGA run-summary `metric_bootstrap.comparison` | **OBSERVED:** Supported. This negative result must remain. |
| “Across the complete WiDS export, MC-confidence-only and NARS-gated produced no different threshold decisions” | WiDS prediction export | **OBSERVED:** Confirmed at 0.5. The same is true for TCGA. |
| “preserving comparable held-out predictive behavior” | Both metric tables and prediction exports | **OBSERVED:** Point estimates are close. **INFERRED:** Formal equivalence or noninferiority has not been established. |
| “preprocessing must be fitted on the training split” | [TCGA merge and selection](/C:/Users/USER/Desktop/code/NGTA/src/data_loader.py:376) | **OBSERVED:** Imputation/scaling/encoding fit training data, but sparse-column filtering and mutation-panel selection occur before splitting. |
| “stronger gating monotonically lowers the Brier point estimate” on WiDS | [gamma sweep](/C:/Users/USER/Desktop/code/NGTA/results/wids/metrics/gamma_ablation.csv) | **OBSERVED:** Supported descriptively across the five saved values. AUC deltas are negative; at gamma 4, accuracy is also below baseline. |
| “This supports human oversight” | Trace machinery; no human evaluation artifact | **INFERRED:** Plausible intended use, not an evaluated outcome. |

#### 2. The five strongest rejection reasons

1. **The multimodal experiment is substantively invalid.**  
   **OBSERVED:** The loader fills absent genomic records with zero. That turns unavailable mutation data into apparent mutation absence for 455/457 labeled patients. All validation and test mutation inputs are zero, and BRAF never fires. **INFERRED:** The reported TCGA experiment cannot establish useful clinical-genomic fusion or genomic intervention, even as a held-out feasibility demonstration.

2. **The positive calibration interpretation is confounded by prediction aggregation.**  
   **OBSERVED:** The baseline computes
   \[
   \mathbb E[\operatorname{sigmoid}(b+\textstyle\sum_i\alpha_i s_i)],
   \]
   while flat-confidence computes
   \[
   \operatorname{sigmoid}(\mathbb E[b]+\textstyle\sum_i\mathbb E[\alpha_i]\mathbb E[s_i]).
   \]
   These differ through sigmoid nonlinearity and attention–score covariance, although the uniform gate leaves attention unchanged. Approximately **79.8%** of the WiDS baseline-to-NARS Brier reduction is already present in this control. **INFERRED:** Even “confidence routing improves calibration” is stronger than the experiment isolates.

3. **The implemented symbolic mechanism lacks the claimed evidential semantics.**  
   **OBSERVED:** Neural feature frequency is an attention weight, whereas symbolic frequency describes a clinical proposition. Revised frequency never enters the final prediction; revised confidence depends only on the two confidences. Grounding algebraically reconstructs the preassigned symbolic confidence. **INFERRED:** The predictive mechanism is effectively a rule-conditioned confidence boost, without evidence that NAL’s frequency semantics or clinical proposition content are necessary.

4. **Auditability is the primary contribution, but its validation is incomplete.**  
   **OBSERVED:** The revision audit reuses production arithmetic; completeness checks finite in-memory numbers rather than complete persisted events; detailed traces are sampled; the existing WiDS case file is stale. A negative-input probe also exposes an extractor bug. **INFERRED:** Arithmetic activity does not establish dependable end-to-end auditability or useful human oversight.

5. **The evaluation cannot support robustness or generalization claims.**  
   **OBSERVED:** Evidence comes from one seed, internal row splits, four rules per dataset, and a minimal baseline suite. All 144 WiDS test hospitals also occur in training. APACHE mortality probability is an input, without an APACHE-only comparison. **INFERRED:** Performance may depend on an existing prognostic score and familiar institutions; neither missingness robustness nor independent institutional generalization is established.

#### 3. Leakage, confounding, circularity, duplication, and reporting risks

###### Data construction and leakage

- **OBSERVED — TCGA selection before splitting.** Sparse-column filtering uses the merged cohort; the mutation panel uses all loaded MAF records. **INFERRED:** This permits held-out feature-distribution information to influence representation. In this particular split, the two mutation-bearing patients happen to be in training, so held-out mutation-driven panel leakage is not demonstrated here.
- **OBSERVED — Missing genomic data become negatives.** The genomic left join uses `fillna(0)`. **INFERRED:** Assay coverage and biology are conflated; a model could learn data availability rather than mutation effects.
- **OBSERVED — WiDS institutional overlap.** Every test hospital and all 232 test ICU identifiers also appear in training. **INFERRED:** Shared measurement practice and case mix can make an internal split optimistic for new institutions.
- **OBSERVED — Duplicate-patient leakage was not found.** The raw WiDS file has unique encounter and patient IDs, no exact duplicate rows, and no exact duplicate vectors across the selected 15 features. TCGA test case IDs are unique. Neither MAF file duplicates variants from the other under the checked variant key.
- **OBSERVED — TCGA repeated records are collapsed column by column using first non-null values.** Clinical data contain 2,344 rows for 507 cases; follow-up contains 2,966 rows for 507 cases. **INFERRED:** The merged record can combine fields from different records or times. This requires a timing audit, although I did not establish that follow-up outcomes enter the selected feature matrix.
- **OBSERVED — TCGA predictors include pathological stage, extension, and residual disease.** **INFERRED:** If the intended task is preoperative prediction, these may be unavailable at the decision time. The manuscript does not define a sufficiently precise prediction landmark to resolve this.
- **OBSERVED — WiDS uses day-one extrema.** The official benchmark explicitly uses the first 24 hours, so these variables are not automatically leakage. **INFERRED:** Admission-time claims would nevertheless be invalid without a different feature window. [Official dataset description](https://physionet.org/content/widsdatathon2020/1.0.0/)

###### Confounding and unsupported evidence independence

- **OBSERVED — APACHE mortality probability is an input.** **INFERRED:** This is a strong existing prediction embedded inside the proposed model. Its inclusion is not proof of label leakage, but score provenance, availability time, score-only performance, and removal sensitivity are essential.
- **OBSERVED — 2,371 APACHE probability values are negative.** The loader does not explicitly convert these values to missing. **INFERRED:** They may encode unavailability or another sentinel condition; their meaning is unresolved without the dictionary.
- **OBSERVED — KNN imputation precedes scaling.** **INFERRED:** Nearest-neighbor selection depends on heterogeneous raw units and may be dominated by large-scale variables. Imputed rule triggers therefore inherit an undocumented distance weighting.
- **OBSERVED — WiDS rules fire after imputation.** Revisions on missing raw values include 1,271 lactate, 240 creatinine, 5 age, and 2 blood-pressure events. **INFERRED:** These are model-derived estimates being treated like observations with the same grounding confidence.
- **OBSERVED — The neural model and rules consume the same measurements.** **INFERRED:** Separate software modules do not establish disjoint evidential bases. Imputation further couples the symbolic evidence to the training data.
- **OBSERVED — TCGA stage and extension rules can encode related pathological information.** **INFERRED:** Rejection of identical feature targets does not prevent duplicated evidence across correlated features.
- **OBSERVED — Routing uses attention variance, not patient-level predictive variance.** **INFERRED:** Calling the gate calibrated epistemic confidence requires validation beyond the presence of MC dropout.
- **OBSERVED — Saved feature confidences are high:** approximately 0.914–0.9999 on TCGA and 0.857–0.9998 on WiDS. **INFERRED:** Saturated confidence leaves little room for symbolic revision and may explain its tiny incremental effect.
- **OBSERVED — Revised frequency is unused downstream.** **INFERRED:** Changing a symbolic frequency while preserving its grounded confidence cannot express a different predictive evidential direction through the gate.
- **OBSERVED — The CLS contribution remains ungated, and token representations already mix features.** **INFERRED:** Reducing a feature’s readout attention does not remove that feature’s influence from the prediction.

###### Circular and incomplete audit evaluation

- **OBSERVED — Production revision audits itself.** The same function computes actual and expected revision, so a shared formula error can pass.
- **OBSERVED — “Complete” means finite numerical fields.** It does not verify clinical source provenance, observed versus imputed status, correct condition evaluation, or complete disk export.
- **OBSERVED — Attention/probability changes compare NARS against the original baseline.** **INFERRED:** They include uncertainty gating and aggregation effects, not just the effect of symbolic intervention.
- **OBSERVED — The extension branch fails to enforce its trigger mask.** In an in-memory probe with encoded `None`, `Unknown`, and `Minimal (T3)` categories, it reported one condition trigger but assigned three feature revisions. See [the branch](/C:/Users/USER/Desktop/code/NGTA/src/knowledge_base.py:240). **OBSERVED:** The saved 69-case test split contains only recognized extension categories or missing values, so I did not find this bug contaminating its reported counts.
- **OBSERVED — The default export cannot independently reconstruct every claimed truth-value event.** My independent reconstruction of saved gates agreed within about \(1.7\times10^{-8}\); reconstructed probabilities agreed within \(10^{-9}\). These support numerical consistency, but do not validate rule semantics or missing provenance.

###### Selective reporting and statistical inflation

- **OBSERVED — Only one current seed is evidenced.** [The “multiseed” file](/C:/Users/USER/Desktop/code/NGTA/results/submission/multiseed_metrics.csv) has `runs=1`; aggregation replaces undefined standard deviations with zero. **INFERRED:** Zero must not be interpreted as demonstrated stability.
- **OBSERVED — Stale outputs coexist.** The submission file reports WiDS baseline AUC 0.882963, while the current run reports 0.880340. The WiDS case file differs from current predictions by as much as 0.105301. The README acknowledges stale submission artifacts, which mitigates but does not resolve provenance confusion.
- **OBSERVED — Figure selection prefers unseeded result folders.** [The selector](/C:/Users/USER/Desktop/code/NGTA/src/paper_figures.py:138) can use old default results after a new multi-seed run; otherwise it selects one seed rather than aggregating.
- **OBSERVED — The richer ablation CSV is empty.** Standard baselines and submission ablations exist as code options, not completed evidence in the current bundle.
- **OBSERVED — Rule-confidence sensitivity is incorrectly implemented as replacement.** [The ablation](/C:/Users/USER/Desktop/code/NGTA/src/pipeline.py:804) substitutes scaled symbolic confidence for revised confidence instead of rerunning revision. **INFERRED:** Its label would misdescribe the tested mechanism.
- **OBSERVED — Gamma is evaluated repeatedly on test labels.** **INFERRED:** This creates test-set tuning opportunities, but I found no proof that the default gamma was selected that way.
- **OBSERVED — Bootstrap intervals resample rows from one fitted model and split.** **INFERRED:** They omit training, split, MC-sampling, and hospital-level uncertainty.
- **OBSERVED — Several comparisons and metrics are tested without a recorded multiplicity policy.** **INFERRED:** The ECE interval barely excluding zero is especially vulnerable to analysis flexibility.
- **OBSERVED — ECE uses ten equal-count rank bins, rebuilt in each bootstrap sample.** **INFERRED:** Binning choices and small TCGA bin sizes can materially affect estimates.
- **OBSERVED — Reported paired “delta” is the mean bootstrap delta.** WiDS’s direct ECE difference is approximately **0.001663**, whereas the manuscript reports **0.00120** from the bootstrap mean. **INFERRED:** This mixes the observed effect estimate with the bootstrap distribution’s center.
- **OBSERVED — AUC comparisons use overlap of marginal intervals.** **INFERRED:** Overlap neither supplies a paired difference test nor establishes equivalence.
- **OBSERVED — Decision curves omit MC-confidence-only.** **INFERRED:** The most important comparator for symbolic added value is absent from this analysis.
- **OBSERVED — RF uses balanced class weights while the neural loss is unweighted.** **INFERRED:** This is an additional calibration-design difference that should be controlled.
- **OBSERVED — Accuracy uses 0.5 on a WiDS test set with about 8.63% mortality.** **INFERRED:** Accuracy alone conceals sensitivity, precision, and the practical meaning of threshold changes.

###### Reproducibility and source integrity

**OBSERVED:** Run summaries record commit `5817742…`, while the current checkout is `d4bd42d…`. The pipeline does not save trained checkpoints, fitted imputers/encoders, per-pass MC outputs, or exact train/validation ID manifests. Dependency requirements are lower bounds, and data acquisition lacks an immutable manifest with hashes. These omissions prevent exact replay from the saved artifacts alone.

**OBSERVED:** The downloader picks a large matching MAF, but does not verify cohort coverage. The supplied files demonstrate why file size is insufficient as a coverage check.

**OBSERVED:** Bibliographic spot-checking found an actual metadata error: `chudasama2025` lists pages 12345–12360; the authors’ institution lists 39489–39509. This is not evidence that the paper is fabricated. A complete citation audit remains necessary. [Institutional publication record](https://www.sustainability.uni-hannover.de/en/research/publications/details/fis-details/publ/0b1b2ddb-7035-4582-bf48-b6f63f371606?cHash=d64f976c7573485623271fbcfd5f4ee5)

#### 4. Missing comparisons and validation

**PROPOSED — Baselines:** Add matched-aggregation ungated inference, deterministic inference, mean-logit inference, APACHE-only and recalibrated APACHE, calibrated logistic regression, tuned gradient boosting, and validation-only recalibration of the transformer. Include a simple rule-conditioned confidence boost without NAL frequency calculations.

**PROPOSED — Ablations:** Remove APACHE; separate clinical-only/genomic-only/fused TCGA models on verified genomic coverage; disable individual rules and all rules; separate observed-only from imputed triggers; vary observation confidence; correctly rerun revision under truth-value changes; compare readout gating with actual encoder intervention.

**PROPOSED — Negative controls:** Permute rule assignments while preserving trigger prevalence, permute confidence assignments, use matched irrelevant predicates, hold confidence fixed while changing frequency, shuffle training labels, and test false/unknown/missing rule inputs. Preserve failures and null effects.

**PROPOSED — Uncertainty analysis:** Repeat training and dropout sampling separately; evaluate confidence against errors and selective-prediction risk; report log loss, calibration slope/intercept, multiple ECE bin definitions, PR-AUC, subgroup results, and hospital-clustered paired intervals.

**PROPOSED — External tests:** Use held-out hospitals first, then a genuinely independent cohort with locked feature mapping, prediction window, preprocessing, rules, and thresholds. Two unrelated tasks trained separately do not constitute external validation.

**PROPOSED — Auditability evaluation:** Compare the traces against ordinary structured logging in a blinded fault-localization task. Verify every exported event against independently implemented extraction and arithmetic.

#### 5. Ranked experiment plan

PROPOSED: The following criteria should be frozen before new execution. The numerical margins are proposed research criteria, not established clinical thresholds. Use the already-inspected test sets for debugging only; confirmatory claims require a new locked evaluation. Cost estimates use training-run and inference-pass counts because the repository does not record trustworthy runtimes.

###### 1. Establish valid data coverage and replay

- **Hypothesis:** Every modeled feature is backed by documented measurements or explicitly marked missingness.
- **Implementation:** Build case/assay manifests, separate unassayed from mutation-negative patients, verify source hashes, move learned selection inside training, and save preprocessing plus exact split IDs.
- **Split:** Patient-disjoint; hospital-disjoint for WiDS confirmation. TCGA genomic analysis uses verified assay coverage.
- **Metrics/test:** Coverage rates, duplicate/overlap counts, lineage completeness, and deterministic integrity assertions.
- **Failure modes:** Insufficient genomic coverage, unavailable acquisition metadata, or incompatible source versions.
- **Compute:** CPU audit; one rerun per dataset after corrections.
- **Acceptance:** Zero unexplained overlaps or missingness-to-negative conversions; 100% feature lineage; no multimodal claim until evaluated patients have verified assay status.

###### 2. Isolate aggregation from gating

- **Hypothesis:** Confidence gating improves prediction after aggregation is held constant.
- **Implementation:** Save per-pass logits, attention, and token scores. Cross aggregation choice with no gate, uniform gate, MC gate, and NARS gate using identical dropout draws.
- **Split:** Development on current splits; confirmation on locked hospital-held-out WiDS data across five training seeds.
- **Metrics/test:** Brier primary; log loss, AUC, ECE secondary. Paired hospital-cluster bootstrap, with multiplicity control for secondary comparisons.
- **Failure modes:** The improvement vanishes under matched aggregation or varies by seed.
- **Compute:** Five training runs and 50 inference passes per checkpoint; gate variants reuse those outputs.
- **Acceptance:** Uniform and no-gate predictions agree within \(10^{-7}\) under identical aggregation. Claim gating benefit only if the corrected 95% interval favors gating and Brier improves by at least \(10^{-4}\).

###### 3. Test whether clinical rule content adds value

- **Hypothesis:** Correct clinical predicates outperform matched arbitrary confidence boosts.
- **Implementation:** Compare NARS with MC-only, direct confidence boosts, individual-rule removal, 100 prevalence-preserving predicate permutations, and frequency changes at fixed confidence.
- **Split:** Same locked confirmation design as experiment 2; no test-driven rule revision.
- **Metrics/test:** Paired Brier and log loss; empirical permutation test with Holm correction.
- **Failure modes:** Random predicates perform similarly; frequency changes have no effect; benefit comes from one rule.
- **Compute:** Cached inference plus roughly 100–150 inexpensive readout evaluations per checkpoint.
- **Acceptance:** Correct rules outperform MC-only by the preregistered \(10^{-4}\) Brier margin and beat matched random controls at adjusted \(p<0.05\). Otherwise retain only an implementation claim.

###### 4. Validate extraction and complete trace replay

- **Hypothesis:** Every exported intervention corresponds to a valid condition and can be independently reproduced.
- **Implementation:** Test missing, unknown, negative, boundary, unseen-category, and conflicting inputs; compare observed-only with imputed triggers; independently implement revision; export every event with raw value, imputation status, rule version, and counterfactual rule-off effect.
- **Split:** Exhaustive synthetic boundary suite plus all held-out events.
- **Metrics/test:** Trigger precision/recall, missing-field rate, replay error, and paired observed-only versus imputed-rule performance.
- **Failure modes:** False triggers, missing provenance, stale exports, or loss of most claimed coverage.
- **Compute:** CPU checks and cached readout evaluation; no training required initially.
- **Acceptance:** Zero false/missed triggers in the specified suite; every event replayable within \(10^{-7}\); zero undocumented imputed observations.

###### 5. Challenge APACHE dependence and baseline strength

- **Hypothesis:** The model adds value beyond the embedded mortality score and ordinary tabular prediction.
- **Implementation:** Evaluate raw/recalibrated APACHE, APACHE-only logistic regression, selected-feature logistic regression, tuned boosting, and transformer variants with/without APACHE. Resolve negative score values using documentation.
- **Split:** Hospital-held-out outer evaluation; all tuning and calibration inside training hospitals.
- **Metrics/test:** Brier/log loss primary; PR-AUC, AUROC, calibration slope, and threshold-specific sensitivity/PPV. Paired cluster bootstrap.
- **Failure modes:** Recalibrated APACHE matches the model; removal collapses performance; boosting wins.
- **Compute:** Approximately ten transformer fits for five paired seeds, plus a fixed CPU baseline budget.
- **Acceptance:** Claim added predictive value only if Brier improves by at least \(10^{-4}\) over the best locked comparator with a favorable 95% interval. Report losing comparisons.

###### 6. Test missingness robustness and uncertainty usefulness

- **Hypothesis:** Confidence predicts failures and routing reduces degradation as information is removed.
- **Implementation:** Apply 0%, 10%, 30%, 50%, and 70% additional masking to originally observed eligible values; include random, feature-dependent, and predefined outcome-dependent simulation scenarios. Compare imputation strategies and MC dropout with a five-model ensemble.
- **Split:** Locked hospitals; masks fixed across methods and never tuned against test outcomes.
- **Metrics/test:** Brier degradation curves, log loss, error-detection AUROC, and area under the selective-risk curve; paired cluster bootstrap.
- **Failure modes:** Confidence remains high as errors rise, imputed rules dominate, or simple missingness indicators work as well.
- **Compute:** About five model fits per selected uncertainty setup, plus inference across masks.
- **Acceptance:** Confidence predicts errors above chance; routing reduces integrated Brier degradation with a favorable 95% interval and causes no prespecified mask level to worsen Brier by more than 0.001.

###### 7. Test external compatibility without retuning

- **Hypothesis:** The frozen method remains predictively compatible with MC-only routing on an independent cohort.
- **Implementation:** Map equivalent features and prediction windows; freeze all parameters before labels are accessed. Report unsupported feature mappings explicitly.
- **Split:** Entire independent institution/time cohort; check that it does not overlap the development source.
- **Metrics/test:** Paired Brier, AUROC, calibration slope/intercept, subgroup metrics, and clinically prespecified decision curves; clustered noninferiority analysis.
- **Failure modes:** Rules rarely fire, measurement units differ, calibration shifts, or no independent eligible cohort exists.
- **Compute:** One frozen 50-pass evaluation per checkpoint; data harmonization is the major unknown.
- **Acceptance:** Upper 95% bound on NARS-minus-MC Brier below 0.001 and lower bound on AUROC difference above −0.01. These establish compatibility only, not superiority.

###### 8. Test the claimed auditability benefit

- **Hypothesis:** NGTA traces help reviewers identify and correct errors better than ordinary structured logs.
- **Implementation:** Randomized, blinded crossover study with seeded extraction, imputation, and rule errors; compare trace formats using identical underlying cases.
- **Split:** Separate training and evaluation cases; randomize reviewer order and account for reviewer/case effects.
- **Metrics/test:** Correct fault localization, correction accuracy, completion time, and false reassurance; mixed-effects analysis.
- **Failure modes:** More detail increases time without accuracy gains, or reviewers trust incorrect symbolic evidence.
- **Compute:** Minimal model compute; participant effort determined by a prospective power analysis.
- **Acceptance:** At least a 10-percentage-point improvement in localization accuracy with a favorable 95% interval and no increase exceeding 5 points in false reassurance.

#### 6. Claims that must be weakened or withdrawn now

| Claim | Required evidential boundary |
|---|---|
| “clinical-genomic feasibility check” / “symbolic interventions execute on both modalities” | **OBSERVED:** Withdraw the held-out genomic demonstration. Only two training cases contain mutations; no genomic intervention occurs on test. |
| “under extreme missingness” | **OBSERVED:** Restrict to the measured feature-specific missingness. **PROPOSED:** Reserve robustness language for controlled masking/shift results. |
| “controls support confidence routing” | **INFERRED:** Restrict to differences between inference procedures until aggregation is matched. |
| “Every intervention exports…” | **OBSERVED:** Withdraw universal export completeness until every event is actually persisted and replayable. |
| “Independently recomputed revision…” | **OBSERVED:** Describe the current revision check as reuse of production arithmetic. |
| “exposes a trust signal” | **INFERRED:** Treat this as a heuristic attention-stability signal until its relationship to errors and uncertainty is validated. |
| “preserving comparable” or “competitive” prediction | **INFERRED:** Limit to close point estimates on these internal splits; no formal equivalence or broad competitiveness is established. |
| “preprocessing…fitted on the training split” | **OBSERVED:** Qualify the statement to imputation, scaling, encoding, and training-based constant removal; selection before splitting remains. |
| “supports human oversight” / “demonstrated value is auditability” | **INFERRED:** Limit to inspectable numerical instrumentation until independent replay and user evaluation are complete. |
| Baseline-to-NARS flips as symbolic effects | **OBSERVED:** Keep them explicitly attributed to the entire inference change. MC-only and NARS make identical 0.5-threshold decisions in both saved datasets. |

OBSERVED: The negative findings already acknowledged by the manuscript are real and should remain: MC-only has slightly better WiDS Brier/ECE, NARS has no isolated advantage over the simpler gates, TCGA comparisons are inconclusive, and stronger WiDS gating slightly reduces AUC.

INFERRED: The next revision should start with data validity, matched prediction aggregation, and independently replayable traces. Expanding the rule base or upgrading uncertainty estimation before those checks would make the system more elaborate without resolving the strongest rejection arguments.