# NGTA

**NARS-Guided Transformer Attention for clinical transformers with incomplete inputs**

NGTA is an auditable inference-time interface for uncertainty-conditioned symbolic intervention. MC dropout initializes heuristic truth values; explicit prototype rules revise confidence and change the transformer readout or encoder attention. Matched dropout controls, complete event exports and independent replay make the computation inspectable. Clinical improvement and reviewer benefit require further evidence.

NGTA is a neurosymbolic clinical prediction architecture that maps neural uncertainty into NARS truth values and feeds revised confidence back into Transformer attention during inference. The repository supports two benchmark tasks:

- `tcga`: TCGA-THCA lymph node metastasis prediction from merged clinical tables plus a mutation-derived binary gene panel
- `wids`: WiDS Datathon 2020 ICU hospital mortality prediction from first-day ICU measurements with feature-specific missingness

The anonymous NeurIPS 2026 workshop manuscript is [`paper/main.tex`](paper/main.tex), with its compiled PDF at [`paper/main.pdf`](paper/main.pdf). NeurIPS formatting files are local to `paper/`.

## April 21, 2026 Feedback Update

After an email exchange on April 21, 2026, Pei Wang pointed out two conceptual issues that now shape this repository:

- Statistical variance is not itself NARS evidence amount or native NARS confidence. In NGTA, MC-dropout variance is now described explicitly as a heuristic initializer for neural confidence that can later be revised by symbolic evidence.
- The strong-deduction confidence calculation in the manuscript needed to match the standard NAL rule rather than the custom form previously written in the paper.

Repository updates made from that feedback:

- [`src/nars_interface.py`](src/nars_interface.py) now exposes standard NAL strong deduction, revision, evidence-confidence conversion, and expectation helpers.
- Triggered symbolic rules are grounded by an explicit deduction step from empirical observations before neural-symbolic revision.
- The README and paper now attribute these clarifications to Pei Wang and describe the variance-to-confidence mapping more carefully.

## Key Achievements

- **Matched inference:** Ungated, uniform, MC-only and NARS readout gates use identical cached dropout passes and mean per-pass probabilities. Uniform gating reproduces ungated inference within 1e-7.
- **Valid data handling:** Training-only feature selection, standardized KNN distances, explicit unknown genomic calls, disjoint WiDS split options and observed-only rule triggers prevent the identified preprocessing errors.
- **Independent auditability:** Every rule trigger is exported, including unmapped triggers. Independent replay reconstructs predicates, revisions, gates, predictions and rule-off effects and checks file integrity.
- **Research controls:** Standard tabular/APACHE baselines, rule removal and random controls, encoder intervention, multi-seed exports, ensemble uncertainty and raw missingness experiments are supported. These are implemented controls, not demonstrated clinical superiority.

## Overview

### What it does

NGTA processes incomplete clinical tables with a tabular transformer, estimates attention and predictive stability across dropout passes, and applies explicit prototype rules at inference. It saves the full intervention path alongside risk predictions so a reader can reconstruct what changed.

### Why it matters

The interface makes uncertainty-conditioned interventions explicit and inspectable. It is intended to help investigate how rules change inference on incomplete measurements. Neither clinical safety nor improved human oversight follows from numerical replay; those outcomes require their own studies.

### What is novel here

The interface grounds triggered symbolic rules with selected NAL deduction/revision functions and feeds revised confidence into the inference computation. The default changes the final attention-weighted token-score readout; `--encoder-intervention` biases feature keys inside every attention layer and recomputes contextual representations with replayed dropout RNG.

This is not a complete NARS cognitive architecture. Neural feature frequency is an attention weight, whereas symbolic frequency describes a proposition. Revised frequency is logged but does not drive the confidence gate, and the neural/rule pathways share clinical inputs. Direct confidence-boost and frequency-sensitivity controls test those limitations.

### How it works

1. Fit feature selection, preprocessing and the transformer using training data; use validation data for model selection.
2. Cache every MC-dropout pass's native logits, probabilities, attention, token scores, CLS logits and RNG state.
3. Initialize feature truth values heuristically from attention mean and variance.
4. Extract rules from observed measurements, ground their prototype truth values and revise the corresponding neural values.
5. Score ungated, uniform, MC-only, NARS and ablation gates on the same passes; average probabilities after sigmoid. Save encoder interventions separately.
6. Export calibration/discrimination, paired comparisons, subgroup/decision curves and complete events; independently replay the persisted inference.

### Why there are two datasets

- `tcga` exercises the clinical/genomic input interface for retrospective post-pathology lymph-node association. The acquired cohort now has recorded mutation positives in 50 of 69 held-out cases and 39 BRAF rule events; complete callable assay panels and measurement chronology remain unverified.
- `wids` supplies a larger first-day ICU mortality task. Patient- and hospital-disjoint splitting is supported. Training separate models on these two tasks is not independent-cohort validation.

### What we found

The current development studies contain 20 hospital-held-out WiDS fits across KNN/median imputation, APACHE inclusion/exclusion and five seeds, plus 15 TCGA clinical-only/genomic-only/fused fits. Each uses up to 60 epochs with validation early stopping, 50 matched dropout passes and three independent dropout repeats. Five-model ensembles, validation recalibration, symbolic controls, subgroup metrics and decision curves are exported alongside complete intervention replay.

The planned symbolic Brier benefit of 0.0001 is not established. For WiDS KNN, mean NARS-minus-MC Brier is -0.000000243 with hierarchical 95% interval [-0.000000691, 0.000000164]. Removing APACHE reduces mean ungated AUROC from 0.875721 to 0.841118. These are development results on already inspected cohorts; internal held-out hospitals do not establish external compatibility. The original numerical tables below remain historical evidence with their aggregation and audit limitations disclosed.

The full masking study passes its paired degradation/harm criteria in 12 of 15 median seed/scenario combinations and 1 of 15 KNN combinations. None of the five KNN random or feature-dependent scenarios passes, so the results do not support a general routing-robustness claim. Individual acceptance reports and uncertainty intervals remain available for inspection.

For the fixed WiDS KNN ensemble, predictive variance detects threshold errors above chance (AUROC 0.842276), while neural and revised attention-confidence scores yield 0.514950 and 0.500647 with hospital-bootstrap intervals spanning 0.5. The routing confidence therefore remains an unvalidated trust signal even when predictive uncertainty is informative.

[list.md](list.md) tracks genuinely completed work. The detailed research audit is in [docs/research-audit.md](docs/research-audit.md), leaving this README focused on the project and its use.

## Running

Create an environment and install dependencies:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run one dataset:

```bash
python main.py --dataset tcga
python main.py --dataset wids
```

Run the whole repository pipeline:

```bash
python main.py --run-all
```

`--run-all` is the full orchestration entrypoint. It runs the complete TCGA pipeline and the complete WiDS pipeline sequentially, computes every metric/chart/trace artifact for both datasets, and writes an aggregate `run_all_summary.json` at the chosen output root.

Useful flags:

- `--data-dir`: directory containing the TCGA tables / MAF files and `wids_icu.csv`
- `--output-dir`: base directory for per-dataset outputs
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`
- `--mc-samples`, `--gamma`, `--seed`; `--split-seed` holds partitions fixed across training seeds
- `--d-model`, `--num-heads`, `--num-layers`, `--dropout`, `--patience`
- `--seeds 0 1 2 3 4`: run multiple seeds and aggregate submission-ready metrics
- `--baseline-set standard`: add calibrated logistic regression, ExtraTrees, and histogram gradient boosting baselines
- `--ablation-set submission`: add symbolic-disabled and rule-truth sensitivity summaries
- `--export-case-traces`: write curated glass-box case traces for representative held-out patients
- `--paper-tables`: export aggregate CSV and LaTeX tables under `results/submission`
- `--skip-paper-figures`: skip automatic generation under `<output-dir>/paper_figures`
- `--ensemble-size 5`: train five seeded models and compare MC-dropout variance, predictive entropy, and deep-ensemble variance
- `--shift-eval`: mask originally observed raw values and rerun frozen preprocessing, rules and matched inference
- `--audit-data`: export source hashes, missingness, coverage, conflicts and exact case partitions without training
- `--split-mode hospital`: use hospital-disjoint WiDS partitions; patient-disjoint is the default
- `--without-apache`: remove APACHE from feature-based predictors while preserving score-only comparators
- `--encoder-intervention`: compare intervention inside every encoder attention layer with readout gating
- `--evaluation-lock PATH`: reject changed configuration, sources, splits or rules before training/exports; this is not prospective preregistration
- `--imputation knn|median`: select the training-fitted imputer
- `--mc-repeats 3`: save independent dropout repetitions in addition to training-seed variation
- `--cache-dir .cache/ngta`: reuse preprocessing only when source hashes, splits, options and preprocessing code match
- `--resume`: reuse a compatible saved checkpoint; reject changed training provenance before writing artifacts
- `--feature-mode all|clinical|genomic`: compare TCGA modalities on identical case partitions with training-only preprocessing
- `--shuffle-training-labels`: run a labeled neural negative control without changing classical baseline targets

Notes:

- WiDS uses a dataset-specific batch-size override of `512`
- `--dataset` is used for single-dataset execution; `--run-all` runs both datasets regardless
- outputs are namespaced by dataset so TCGA and WiDS artifacts do not overwrite each other
- multi-seed runs are written under `results/seed_<seed>/...` so repeated submission runs do not overwrite each other

Submission-oriented run:

```bash
python main.py --run-all --seeds 0 1 2 3 4 --baseline-set standard --ablation-set submission --export-case-traces --paper-tables
```

This writes submission artifacts under the selected output directory:

- `results/submission/multiseed_metrics.csv`
- `results/submission/baseline_comparison.csv`
- `results/submission/ablation_summary.csv`
- `results/submission/case_traces.csv`
- `results/submission/auditability_metrics.csv`
- `results/submission/paired_metric_deltas.csv`
- `results/submission/paper_tables.tex`
- run-specific figures under `<output-dir>/paper_figures`

Paper figures require matching result schemas and independently replayed sources. The manuscript's existing figure paths now contain explicitly selected five-seed TCGA fused and WiDS KNN results. Current tables record their sources in `paper/figures/current_results_manifest.json`; seed-mean intervals are distinct from case-bootstrap intervals. Regenerate them with:

```bash
python -c "from src.paper_figures import generate_paper_figures; generate_paper_figures(seeds=[0,1,2,3,4], dataset_roots={'tcga':'results/tcga_study/fused','wids':'results/hospital_study/knn'})"
python scripts/refresh_paper_results.py
```

The submission artifacts support an auditability-first framing: NGTA is a glass-box evidential routing interface for clinical transformers, with performance treated as compatibility evidence rather than as a claim of universal superiority. `auditability_metrics.csv/json` reports held-out rule coverage, trace completeness, arithmetic residuals, attention effects, probability changes, and threshold flips. These are operational checks, not a clinician usability evaluation.

Rules are evaluated independently in a single inference pass; any subset can fire. The current prototype permits at most one rule per logical feature and rejects collisions instead of applying an order-dependent overwrite. Larger same-feature rule bases require an explicit provenance-aware conflict policy.

## Data

TCGA expects the following in [`data/`](data):

- `clinical.tsv`
- `exposure.tsv`
- `family_history.tsv`
- `follow_up.tsv`
- `pathology_detail.tsv`
- one or more `*.maf` files

WiDS expects:

- `wids_icu.csv` with patient/hospital IDs for the corresponding grouped split

An optional `assay_manifest.csv` contains `case_submitter_id,gene,source,verified`, with unique case/gene pairs and sourced callable coverage for verified negatives. A missing mutation record is unknown, not a negative. The acquired source is isolated under `data/acquired_tcga/`: its immutable manifest pins 498 GDC files, publisher checksums, exact extracted-byte hashes and coverage. Sixteen expected clinical cases lack file-associated coverage; coverage is not gene-level callability. Original sparse source files remain historical. Cross-table timing is unverified; pathology features support retrospective association, not preoperative prediction.

`python scripts/audit_tcga_timing.py` checks dates on the selected physical records and all frozen model/rule inputs. Diagnosis dates exist, but all 507 pathology records lack pathology dates. [docs/source-evidence.md](docs/source-evidence.md) records those counts, the public APACHE evidence and the exact metadata still required.

```powershell
python main.py --run-all --audit-data --split-mode hospital --output-dir results/source_audit
python -m src.trace_replay results/v2_smoke/tcga/traces
```

## WiDS Configuration

The default feature set contains 13 numeric measurements (`age`, `bmi`, day-one vital/laboratory extrema and `apache_4a_hospital_death_prob`), binary `elective_surgery` and categorical `gender`. `--without-apache` removes the score from feature-based models.

Preprocessing fits training data only: clean invalid values, standardize numeric KNN distances, impute, scale final numeric features and one-hot encode gender. Patient-disjoint splitting is the default; hospital-disjoint evaluation is selected explicitly. The source audit has zero patient/hospital/ICU overlap under the hospital split.

APACHE values outside [0,1] are unavailable: 2,371 such values occur locally. Selected-feature missingness after cleaning is 10.3559%, while lactate is 74.5761% missing. Authoritative sentinel semantics and score availability remain to be verified.

Rules require observed values: lactate >=4 mmol/L, systolic pressure in [0,90] mmHg, age >=75 years and creatinine >=2 mg/dL. Imputed values and suppressed imputed-only triggers are separate diagnostics; the masking study names its imputed-rule comparator explicitly.

## Interpretation Caveats

This repository is a research implementation, not a clinical validation package.

- The acquired TCGA held-out split has 69 cases with recorded positives and replayed genomic interventions, but complete callable panels and verified negatives remain unavailable. The earlier sparse source did not demonstrate held-out genomic intervention.
- The four thyroid and four ICU rules are prototype probes with version/source metadata; all require clinical expert review. Shared inputs do not establish independent evidence, and same-target collision checks do not resolve correlated propositions.
- Following Pei Wang's April 21, 2026 feedback, variance-to-confidence is an application-specific heuristic, not native NARS evidence amount.
- Matched inference and complete numerical replay establish implementation behavior. Full development comparisons fail the symbolic benefit threshold; masking criteria apply to their stated simulations. Clinical equivalence, external compatibility and reviewer usefulness remain unestablished.
- Existing outcomes have already been inspected. A file lock preserves configuration but cannot make those outcomes unseen; prospective confirmation needs a defensible fresh evaluation.
- Verified assay/timing metadata, an eligible independent cohort, expert review and reviewer participants are external dependencies. [docs/research-audit.md](docs/research-audit.md) lists the exact evidence and actions needed for each unfinished task. [Data access instructions](docs/get-needed-data.md) give the official websites and steps; [email drafts](docs/email-templates.md) cover source custodians, clinical reviewers and Pei Wang.

## Outputs

Each dataset writes a full artifact bundle under the chosen output root:

- `<output-dir>/tcga/charts`
- `<output-dir>/tcga/metrics`
- `<output-dir>/tcga/traces`
- `<output-dir>/wids/charts`
- `<output-dir>/wids/metrics`
- `<output-dir>/wids/traces`

Top-level orchestration output:

- `<output-dir>/run_all_summary.json`

Per-dataset bundles also preserve `model.pt`, fitted `preprocessor.joblib`, classical estimators, `evaluation_spec.json`, source hashes and split IDs. Complete replay uses `raw_test.csv`, `inference_cache.npz`, `replay_spec.json`, `intervention_events.csv`, hashes and saved MC RNG. Optional ensemble/encoder outputs remain separate.

Full studies are under `results/hospital_study/` and `results/tcga_study/`; each condition has five seed directories, aggregate submission exports and `analysis/` seed/ensemble/hierarchical comparisons. Missingness outputs preserve raw masks, matched probabilities, selective-risk curves and paired acceptance reports. Generated hospital estimators and inference arrays are stored separately from Git. Exact replay after cloning requires restoring the approved artifact archive; the training commands can regenerate these files. See [artifact storage and verification](docs/artifact-storage.md).

To reproduce the main hospital study and its frozen masking analysis:

```bash
python main.py --dataset wids --split-mode hospital --imputation knn --seeds 0 1 2 3 4 --split-seed 0 --epochs 60 --mc-samples 50 --mc-repeats 3 --cache-dir .cache/ngta --baseline-set standard --ablation-set submission --skip-paper-figures --output-dir results/hospital_study/knn
python scripts/evaluate_study.py --root results/hospital_study/knn --mask-seeds 0 1 2 3 4
python scripts/uncertainty_intervals.py --root results/hospital_study/knn
```

`scripts/tcga_controls.py` runs the clinical/genomic controls. `scripts/evaluate_external.py --help` documents frozen independent-cohort evaluation with explicit units, windows, rule-only inputs and patient overlap checks. `scripts/reviewer_study.py --help` prepares and analyzes a reviewer study; the committed package is a demonstration with public investigator answers, so generate a fresh private package before recruitment. `requirements-lock.windows-py314.txt` records the actual run environment; its CUDA Torch wheel uses the matching PyTorch wheel index.

The open PhysioNet 2012 sensitivity uses 12,000 cases, first-24-hour inputs and the frozen no-APACHE checkpoints. Reproduce acquisition with `python scripts/prepare_physionet2012.py --download` and evaluation with `python scripts/evaluate_cross_source.py`. Source hashes, missing inputs, seed metrics and independently replayable traces are saved under `results/cross_source_sensitivity/physionet2012/`. Its single-center case intervals and unverified source identity/window equivalence do not establish independent clinical validation. [docs/artifact-storage.md](docs/artifact-storage.md) explains how to restore the separately stored inference arrays for exact replay.

Per-dataset metrics/traces include:

- `metrics.csv`
- `training_history.csv`
- `gamma_ablation.csv`
- `decision_curve.csv`
- `calibration_reliability.csv`
- `run_summary.json`
- `test_predictions.csv`
- `auditability_metrics.csv` and `auditability_metrics.json`
- ROC, calibration, training-history, gamma-ablation, and decision-curve plots

`metrics.csv` reports 95% bootstrap confidence intervals for AUC, Brier score, and ECE across the random forest, baseline transformer, flat-confidence transformer, MC-confidence-only ablation, and NARS-gated transformer. Run summaries and paired-comparison exports include observed AUROC/Brier/ECE/log-loss deltas separately from bootstrap means and multiplicity-adjusted intervals against every saved comparator. WiDS resamples hospitals. Subgroup metrics and threshold decision curves include MC-only.

## Latest Full Run

The numbers in this section are the archived earlier implementation, not results from the corrected main implementation. Different aggregation confounds baseline/gated comparisons, rules could trigger after imputation, and revision audits reused production arithmetic. Preserve the numerical record without interpreting it as a clinical or symbolic benefit. Current run status and replay evidence are linked in [docs/research-audit.md](docs/research-audit.md).

The archived default full run was produced with:

```bash
python main.py --run-all
```

This was a single-seed default run with `baseline_set=minimal`, `ablation_set=quick`, `mc_samples=50`, `gamma=2.0`, and seed `0`. The richer multi-seed submission artifacts are produced only by the longer `--seeds ... --baseline-set standard --ablation-set submission --export-case-traces --paper-tables` command.

Result bundles written by that run:

- [`results/run_all_summary.json`](results/run_all_summary.json)
- [`results/tcga/metrics/run_summary.json`](results/tcga/metrics/run_summary.json)
- [`results/wids/metrics/run_summary.json`](results/wids/metrics/run_summary.json)

The numerical summary below is sourced from these per-dataset artifacts. The existing files under `results/submission/` were not regenerated by this default `--run-all` invocation and should not be used as the source of the refreshed numbers.

TCGA-THCA full-run summary:

Historical role: clinical/genomic input demonstration; its sparse held-out source did not establish genomic fusion.

- Split: `319 / 69 / 69` train/validation/test from `457` labeled cases
- Best default AUC: `0.73277` for `flat_confidence` with 95% CI `[0.60282, 0.84794]`
- Best Brier: `0.21091` for `flat_confidence` with 95% CI `[0.18160, 0.24265]`
- Best ECE: `0.11779` for `flat_confidence` with 95% CI `[0.11300, 0.26405]`
- Accuracy: `0.68116` for all four Transformer variants
- MC-confidence-only ablation: AUC `0.73109`, Brier `0.21095`, ECE `0.11843`, accuracy `0.68116`
- NARS-gated: AUC `0.73109`, Brier `0.21094`, ECE `0.11837`, accuracy `0.68116`
- Symbolic activity: `42 / 69` held-out cases with any trigger, `79` total feature-level revisions
- Operational auditability: `100%` trigger mapping and finite traces; all `79` rule events changed confidence and attention; mean/median/maximum absolute probability changes among triggered cases were `0.00205 / 0.00185 / 0.00600`; no triggered case crossed the `0.5` threshold; revision and gate residuals were `0`.
- Interpretation: the flat-confidence control is strongest by these historical point estimates, but all paired Brier/ECE comparisons between NARS-gated and the transformer controls include zero. This sparse held-out source did not establish multimodal feasibility or symbolic benefit.

WiDS ICU full-run summary:

Role in the paper: primary scale test for missingness, calibration, and operational auditability

- Split: `64199 / 13757 / 13757` train/validation/test from `91713` labeled rows
- Input width: `16` model features after preprocessing
- Best AUC: `0.88034` for `baseline` with 95% CI `[0.87001, 0.89021]`
- Best Brier: `0.056468` for `mc_confidence_only` with 95% CI `[0.053647, 0.059625]`
- Best ECE: `0.005808` for `mc_confidence_only` with 95% CI `[0.004090, 0.010617]`
- Best accuracy: `0.92862`, tied across `flat_confidence`, `mc_confidence_only`, and `nars_gated`
- NARS-gated: AUC `0.88024`, Brier `0.056469`, ECE `0.005833`, accuracy `0.92862`
- Symbolic activity: `8551 / 13757` held-out cases with any trigger, `13031` total feature-level revisions
- Operational auditability: `100%` trigger mapping and finite traces; all `13031` rule events changed confidence and attention; mean/median/maximum absolute probability changes among triggered cases were `0.00214 / 0.00159 / 0.03102`; `10` triggered cases crossed the `0.5` threshold relative to the ungated baseline; revision and gate residuals were `0`.
- Paired bootstrap comparisons:
- `baseline -> nars_gated` Brier `0.056527 -> 0.056469`; paired delta CI `[0.0000249, 0.0000911]`
- `baseline -> nars_gated` ECE `0.007496 -> 0.005833`; paired delta CI `[0.0000282, 0.0020241]`
- `flat_confidence -> nars_gated` Brier `0.056480 -> 0.056469`; paired delta CI `[-0.0000050, 0.0000290]`
- `flat_confidence -> nars_gated` ECE `0.006182 -> 0.005833`; paired delta CI `[-0.0003862, 0.0008464]`
- `mc_confidence_only -> nars_gated` Brier `0.056468 -> 0.056469`; paired delta CI `[-0.00000286, 0.00000070]`
- `mc_confidence_only -> nars_gated` ECE `0.005808 -> 0.005833`; paired delta CI `[-0.0001681, 0.0001255]`
- `random_forest -> nars_gated` Brier `0.058169 -> 0.056469`; paired delta CI `[0.0009628, 0.0023993]`
- `random_forest -> nars_gated` ECE `0.007372 -> 0.005833`; paired delta CI `[-0.0038679, 0.0079440]`
- AUC confidence intervals overlap across all WiDS variants.
- Interpretation: unmatched dropout aggregation confounds the historical baseline comparison, so its small calibration difference cannot establish a gating benefit. The `10` baseline-to-NARS threshold flips measure the entire inference change; MC-confidence-only and NARS-gated predictions had no different threshold decisions in that archived run.
- Per-rule test triggers:
  - `rule_lactate: 1936`
  - `rule_hypotension: 5338`
  - `rule_age: 3443`
  - `rule_creatinine: 2314`

## Repository Layout

- [`main.py`](main.py): CLI entry point and `--run-all` orchestration
- [`src/data_loader.py`](src/data_loader.py): TCGA ingestion and preprocessing
- [`src/wids_loader.py`](src/wids_loader.py): WiDS ingestion, preprocessing, and ICU rule-mask generation
- [`src/knowledge_base.py`](src/knowledge_base.py): TCGA symbolic rule base
- [`src/wids_knowledge_base.py`](src/wids_knowledge_base.py): WiDS symbolic ICU rule base
- [`src/neural_encoder.py`](src/neural_encoder.py): tabular Transformer with MC-dropout inference
- [`src/nars_interface.py`](src/nars_interface.py): heuristic neural truth mapping plus standard NAL deduction, revision, and evidential utility operators
- [`src/attention_hook.py`](src/attention_hook.py): confidence-based attention gating
- [`src/pipeline.py`](src/pipeline.py): training, baselines, evaluation, plotting, and summary generation
- [`paper/main.tex`](paper/main.tex): anonymous NeurIPS 2026 workshop manuscript source
- [`paper/main.pdf`](paper/main.pdf): compiled submission PDF

## Acknowledgments

The repository updates in this snapshot were shaped directly by Pei Wang's email feedback on April 21, 2026. In particular, he pointed out that statistical variance is not the same thing as NARS evidence amount and that the manuscript's deduction confidence formula needed to match standard NAL. The current code and paper now reflect those corrections.

The author also thanks Prof. Leilani H. Gilpin for reviewing the manuscript and for guidance on its central contribution: an auditable inference-time neurosymbolic interface rather than a claim of clinically validated prediction improvement. Her feedback informed the paper's framing, NAL/NARS boundary, pipeline figure, case-trace presentation, rule-base discussion, and cautious interpretation of the experimental results.

The project also relies on public TCGA-THCA data from the NCI Genomic Data Commons.
