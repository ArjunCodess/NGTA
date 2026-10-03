# NGTA research audit

The implementation corrections and full internal development studies are complete where their artifacts exist. The plan is not fully achieved: symbolic benefit fails its effect-size criterion, and several clinical claims need provenance or human evidence that code cannot supply. [README.md](../README.md) retains the original project structure. [list.md](../list.md) and [completion_audit.json](../results/research_checks/completion_audit.json) record all 72 tasks, their evidence and exact required actions.

This audit follows [plan.md](../plan.md), preserving the original review as historical context. Earlier `results/v2_validation/` files are historical snapshots, not current completion counts. Running an experiment and obtaining the hoped-for result are separate outcomes.

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

Frozen external evaluation implements documented target/landmark, units/windows/availability for every input and rule-only source, unit conversion, canonical identities, case/patient overlap checks, frozen preprocessing and complete replay. Clustered compatibility requires Brier upper bound <0.001 and AUROC lower bound >-0.01.

Required input: an authorized independent institution/time cohort with verified feature/target provenance and canonical patient identities. No eligible independent cohort was supplied. Internal benchmark hospitals and separate TCGA/WiDS models do not provide external validation. For confirmation, preregister the entire procedure on genuinely untouched outcomes; inspected labels cannot become unseen.

## 7. Reproducibility and implementation

Caches verify source hashes, partitions, options and preprocessing code identity. Resume validates immutable training provenance before writes. Saved artifacts include checkpoints, fitted preprocessors/estimators, calibration data, exact split/group IDs, native/pass arrays, RNG and dependency versions. `requirements-lock.windows-py314.txt` records the actual environment; the CUDA Torch build requires its matching wheel index.

Hospital model/NPZ binaries use Git LFS. Run `git lfs install` and `git lfs pull` after cloning; pointers cannot restore models. This checkout's local LFS cache uses G: to avoid duplicating large files on constrained C:; other clones need no such drive configuration.

The regression suite passes 76 tests, including exact weighted-bootstrap equivalence with duplicated-case/tie reconstruction, independent replay corruption checks, predicate edges and external patient/unit/rule-only mappings. Current figure sources explicitly select five TCGA fused and five WiDS KNN seeds and reject source/split/rule/configuration mismatches. The compiled 13-page PDF was rendered and inspected, with no undefined references or overfull boxes. NeurIPS formatting files are inside `paper/` only; the former top-level directory is absent.

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
| 6. Missingness/uncertainty | Full masks, ensembles and intervals are checked individually before completion; findings are scoped to their simulations. |
| 7. External compatibility | Tooling ready; eligible cohort and verified harmonization required. |
| 8. Human auditability | Protocol/package/power/analysis ready; qualified participant responses required. |

## 10. Immediate priorities and exact blockers

The owner must supply four kinds of external evidence: callable assays and measurement/score timing; qualified expert review; an authorized independent untouched cohort with verified units/windows/targets/identities; and real reviewer responses from a freshly blinded package. The checklist names the required input for every unfinished task.

The remaining scientific blockage is the negative result itself. Keep it, or formulate a clinically motivated new hypothesis and test it on fresh data. Further software work cannot turn these observed comparisons into favorable confirmation.
