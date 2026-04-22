# NGTA

**NARS-Guided Transformer Attention for clinical transformers under extreme missingness**

**TL;DR:** NGTA is a clinical transformer that does not just rank patients; it tries to tell the truth about its own uncertainty. It estimates epistemic uncertainty with MC Dropout, heuristically converts that uncertainty into initial NARS-style truth values, injects explicit human-written medical rules at inference time, and feeds the revised confidence back into attention so brittle evidence is downweighted before the final prediction is made.

NGTA is a neurosymbolic clinical prediction architecture that maps neural uncertainty into NARS truth values and feeds revised confidence back into Transformer attention during inference. The repository now supports two benchmarks in parallel:

- `tcga`: TCGA-THCA lymph node metastasis prediction from merged clinical tables plus a mutation-derived binary gene panel
- `wids`: WiDS Datathon 2020 ICU hospital mortality prediction from a high-missingness ICU tabular cohort

The research paper lives in [`paper/main.pdf`](paper/main.pdf), with source in [`paper/main.tex`](paper/main.tex).

## April 21, 2026 Feedback Update

After an email exchange on April 21, 2026, Pei Wang pointed out two conceptual issues that now shape this repository:

- Statistical variance is not itself NARS evidence amount or native NARS confidence. In NGTA, MC-dropout variance is now described explicitly as a heuristic initializer for neural confidence that can later be revised by symbolic evidence.
- The strong-deduction confidence calculation in the manuscript needed to match the standard NAL rule rather than the custom form previously written in the paper.

Repository updates made from that feedback:

- [`src/nars_interface.py`](src/nars_interface.py) now exposes standard NAL strong deduction, revision, evidence-confidence conversion, and expectation helpers.
- Triggered symbolic rules are grounded by an explicit deduction step from empirical observations before neural-symbolic revision.
- The README and paper now attribute these clarifications to Pei Wang and describe the variance-to-confidence mapping more carefully.

## Key Achievements

- **Inference-Time Logic Injection:** Fuses MC-Dropout epistemic uncertainty with NARS symbolic logic and pushes the revised confidence signal directly into Transformer attention during inference.
- **Scale & Safety:** Benchmarked on `91,713` ICU stays. In the current full run, the baseline transformer is best on AUC at `0.88294`, the flat-confidence control is best on ECE at `0.00490` with 95% CI `[0.00411, 0.00969]`, and the NARS-gated variant is best on Brier score at `0.05618` with 95% CI `[0.05327, 0.05945]`.
- **Glass-Box Activity:** On held-out WiDS ICU data, explicit symbolic rules fired in `8551` of `13757` stays for `13031` total feature-level revisions, showing that the logic layer is active rather than decorative.
- **Multi-Modal Ready:** Demonstrated on fused clinical tabular features and genomic mutation matrices on TCGA-THCA, where the same interface remains operational as a clinical-plus-genomic proof of concept. The TCGA transformer variants are not statistically separated from one another on the 69-case held-out split.

## Overview

### What it does

NGTA is a medical prediction system for messy hospital-style tables where many values are missing. It uses a Transformer to make predictions, but it does not stop at producing a single risk score. It estimates epistemic uncertainty, checks a set of human-written medical rules, and then uses both pieces of information to adjust how the model pays attention to the input features before the final output is emitted.

### Why it matters

Many clinical AI systems can give a strong prediction even when the data are incomplete or unreliable. That is dangerous in real settings because missing hospital data can produce overconfident probabilities that look trustworthy when they are not. NGTA is designed to separate "high score" from "high confidence" and to expose a human-readable revision path when symbolic rules intervene. That makes the system more useful in high-missingness environments like ICU data, where safer calibration matters as much as raw accuracy.

In standard clinical prediction, models optimize for point-estimate accuracy but lack native mechanisms to express epistemic doubt, leading to overconfident extrapolation when faced with missing features. NGTA is built around the opposite design goal: instead of a black-box predictor that guesses blindly across data gaps, it calculates feature-level uncertainty and can route attention toward explicit medical heuristics when uncertainty is high. In that sense, the repository's core contrast is simple: standard transformers behave like black boxes, while NGTA is designed to behave like a glass box.

### What is novel here

The main novelty is not just "Transformer + rules." The key idea is that NGTA turns neural uncertainty into explicit symbolic truth values in a NARS-compatible evidential space, revises those values with domain rules, and then feeds the revised confidence back into Transformer attention. In simple terms: the model can use both learned patterns and symbolic evidence to decide how much trust to place in each feature at inference time, while also leaving behind an auditable evidential trace.

This repository is not a full NARS cognitive architecture. It operationalizes selected NAL truth-value functions as an interface layer for a clinical transformer: heuristic neural truth initialization, explicit symbolic deduction from triggered observations, and revision-based fusion before attention reweighting.

The end result is not just another tabular model with a rules layer attached to the side. It is an auditable, human-in-the-loop reasoning engine: instead of emitting an overconfident scalar score on missing data, the system exposes what it does not confidently know and provides a direct insertion point for clinicians to inject overriding physiological rules into the inference path itself. We refer to this uncertainty-conditioned attention update as Dynamic Evidential Routing. The novel outcome of this project is that calibration, clinician steerability, and auditability all appear in the same deployed inference loop.

### How it works

1. The Transformer reads the patient features and predicts risk.
2. Monte Carlo dropout is used to measure how stable that prediction is across repeated passes.
3. That uncertainty is heuristically converted into initial NARS-style truth values: frequency and confidence.
4. If a symbolic rule fires, the rule is first grounded by explicit NAL deduction from an empirical observation and then combined with the neural truth value using NARS revision.
5. The revised confidence is used to reweight attention, so uncertain or weakly supported features matter less.
6. The pipeline then evaluates discrimination, calibration, decision curves, symbolic trigger activity, and baseline comparisons.

### Why there are two datasets

The two benchmarks test different strengths of the architecture:

- `tcga` is the multi-modal proof of concept. It shows that NGTA can fuse clinical variables with a genomic mutation matrix without breaking the mathematical interface.
- `wids` is the primary empirical validation. It shows that the same architecture scales to a much larger ICU dataset with heavy missingness and gives the clearest large-scale view of calibration, uncertainty routing, and symbolic activity.

### What we found

The main result is that NGTA works as intended on both a small multi-modal cancer dataset and a much larger high-missingness ICU dataset, but the two datasets support different claims.

- On `tcga`, the Transformer-based models still beat the random-forest baseline numerically. The best AUC is `0.72605`, tied between `flat_confidence` and `nars_gated`, versus `0.66134` for random forest. This supports the claim that the interface can learn useful signal from combined clinical and genomic inputs, but it does not support a claim that NARS gating is statistically better than the other Transformer variants.
- The flat-confidence control is the strongest TCGA Transformer variant overall in the current run because it pairs that tied-best AUC with the best Brier score (`0.21184`), the best ECE (`0.13897`), and the best accuracy (`0.68116`). TCGA should therefore still be treated as a multi-modal interface proof of concept rather than evidence that dynamic NARS gating dominates a simpler confidence gate on very small cohorts.
- On `wids`, all Transformer variants are extremely close on AUC around `0.8829`. At full precision, the baseline transformer is best on AUC, the flat-confidence control is best on ECE at `0.00490`, and the NARS-gated version is best on Brier score at `0.05618`.
- The WiDS baseline-vs-NARS paired bootstrap intervals now include zero for both Brier difference (`-0.000004` to `0.000061`) and ECE difference (`-0.000355` to `0.001796`). That means the current run does not statistically establish a calibration gain for NARS gating over the baseline transformer.
- The WiDS NARS-gated variant is also not statistically separated from the flat-confidence control on Brier or ECE. The symbolic rules are active at scale, but the current data still do not isolate their marginal calibration effect over generic confidence gating.
- The WiDS result still matters because the transformer family remains stronger than the random forest on the main summaries, and the symbolic path is physically active during inference. But the right interpretation is now narrower: this run supports operational neurosymbolic routing and competitive calibration, not a confirmed within-family superiority claim for NARS gating.
- The symbolic rules were not just decorative. On the held-out WiDS test set, ICU rules fired in `8551` of `13757` cases for `13031` total feature-level revisions, which means the neurosymbolic revision path was active at scale rather than sitting unused.
- Taken together, the results support a narrower and more defensible claim than "always better accuracy": NGTA is competitive on discrimination, operational as a human-auditable instrumentation layer under heavy missingness, and strongest as a framework for explicit uncertainty routing rather than as a proved winner over every control.

Put differently: the main architectural achievement here is safety-oriented behavior, not just ranking performance. NGTA turns the transformer's attention update from an opaque mapping into a transparent and auditable inference path, where uncertainty is explicit, rule interventions are traceable, and probability reliability becomes something the user can inspect rather than simply assume.

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
- `--mc-samples`, `--gamma`, `--seed`
- `--d-model`, `--num-heads`, `--num-layers`, `--dropout`, `--patience`

Notes:

- WiDS uses a dataset-specific batch-size override of `512`
- `--dataset` is used for single-dataset execution; `--run-all` runs both datasets regardless
- outputs are namespaced by dataset so TCGA and WiDS artifacts do not overwrite each other

## Data

TCGA expects the following in [`data/`](data):

- `clinical.tsv`
- `exposure.tsv`
- `family_history.tsv`
- `follow_up.tsv`
- `pathology_detail.tsv`
- one or more `*.maf` files

WiDS expects:

- `wids_icu.csv`

## WiDS Configuration

The WiDS branch uses exactly these 15 core features:

- Continuous numeric: `age`, `bmi`, `d1_heartrate_max`, `d1_sysbp_min`, `d1_temp_max`, `d1_lactate_max`, `d1_bun_max`, `d1_creatinine_max`, `d1_glucose_max`, `d1_wbc_max`, `d1_spo2_min`, `d1_platelets_min`, `apache_4a_hospital_death_prob`
- Binary pass-through: `elective_surgery`
- Categorical: `gender`

Preprocessing rules:

- `pd.read_csv(..., na_values=['NA'])`
- drop rows where `hospital_death` is missing
- stratified `70/15/15` split with the run seed
- `KNNImputer(n_neighbors=5)` on the 13 continuous features, fit on train only
- `SimpleImputer(strategy='most_frequent')` + one-hot encoding for `gender`
- `StandardScaler` on the 13 continuous features only, fit on train only

WiDS symbolic ICU rules are evaluated after KNN imputation and before scaling:

- `d1_lactate_max >= 4.0`
- `d1_sysbp_min <= 90.0`
- `age >= 75.0`
- `d1_creatinine_max >= 2.0`

## Interpretation Caveats

This repository is a first methods implementation, not a clinical validation package.

- The TCGA held-out split has only `69` cases. The transformer variants are close and should not be described as statistically separated from one another.
- The symbolic rule bases are deliberately thin: four thyroid rules and four ICU rules. They demonstrate that the NARS revision path is active, but they are not independently curated clinical ontologies.
- Following feedback from Pei Wang on April 21, 2026, the repository treats the variance-to-confidence map as an application-specific heuristic initializer, not as a claim that model variance directly measures NARS evidence amount.
- The current results do not establish that these exact hand-selected rules are sufficient or optimal. A stronger study would lock a broader expert-curated rule base before evaluation and report sensitivity to rule inclusion and truth-value assignments.
- There is no external validation cohort in this snapshot. Clinical claims would require temporally or institutionally independent test cohorts with locked preprocessing, model settings, and rule definitions.

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

Per-dataset metrics/traces include:

- `metrics.csv`
- `training_history.csv`
- `gamma_ablation.csv`
- `decision_curve.csv`
- `calibration_reliability.csv`
- `run_summary.json`
- `test_predictions.csv`
- ROC, calibration, training-history, gamma-ablation, and decision-curve plots

`metrics.csv` now reports 95% bootstrap confidence intervals for AUC, Brier score, and ECE. `run_summary.json` also includes paired bootstrap Brier/ECE deltas for the main comparisons.

## Latest Full Run

The current default full run was produced with:

```bash
python main.py --run-all
```

Result bundles written by that run:

- [`results/run_all_summary.json`](results/run_all_summary.json)
- [`results/tcga/metrics/run_summary.json`](results/tcga/metrics/run_summary.json)
- [`results/wids/metrics/run_summary.json`](results/wids/metrics/run_summary.json)

TCGA-THCA full-run summary:

Role in the paper: multi-modal proof of concept for clinical-plus-genomic fusion

- Split: `319 / 69 / 69` train/validation/test from `457` labeled cases
- Best AUC: `0.72605`, tied between `flat_confidence` and `nars_gated`
- Best Brier: `0.21184` for `flat_confidence` with 95% CI `[0.17930, 0.24551]`
- Best ECE: `0.13897` for `flat_confidence` with 95% CI `[0.11629, 0.25967]`
- Best accuracy: `0.68116` for `flat_confidence`
- Symbolic activity: `42 / 69` held-out cases with any trigger, `79` total feature-level revisions
- Interpretation: the flat-confidence control is strongest overall among transformer variants, while the NARS-gated model only ties it on AUC. The transformer variants are not statistically separated on this small split.

WiDS ICU full-run summary:

Role in the paper: primary empirical validation for scale, missingness, and calibration

- Split: `64199 / 13757 / 13757` train/validation/test from `91713` labeled rows
- Input width: `16` model features after preprocessing
- Best AUC: `0.88294` for `baseline`
- Best Brier: `0.05618` for `nars_gated` with 95% CI `[0.05327, 0.05945]`
- Best ECE: `0.00490` for `flat_confidence` with 95% CI `[0.00411, 0.00969]`
- Best accuracy: `0.92884`, tied across `flat_confidence` and `nars_gated`
- Symbolic activity: `8551 / 13757` held-out cases with any trigger, `13031` total feature-level revisions
- Paired bootstrap comparisons:
- `baseline -> nars_gated` Brier `0.05621 -> 0.05618`; paired delta CI `[-0.000004, 0.000061]`
- `baseline -> nars_gated` ECE `0.00601 -> 0.00494`; paired delta CI `[-0.000355, 0.001796]`
- `flat_confidence -> nars_gated` Brier `0.05618 -> 0.05618`; paired delta CI `[-0.000009, 0.000015]`
- `flat_confidence -> nars_gated` ECE `0.00490 -> 0.00494`; paired delta CI `[-0.000598, 0.000790]`
- `random_forest -> nars_gated` Brier `0.05821 -> 0.05618`; paired delta CI `[0.001351, 0.002726]`
- `random_forest -> nars_gated` ECE `0.00764 -> 0.00494`; paired delta CI `[-0.001682, 0.005866]`
- AUC confidence intervals overlap across all WiDS variants.
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
- [`paper/main.pdf`](paper/main.pdf): compiled research paper
- [`paper/main.tex`](paper/main.tex): manuscript source

## Acknowledgments

The repository updates in this snapshot were shaped directly by Pei Wang's email feedback on April 21, 2026. In particular, he pointed out that statistical variance is not the same thing as NARS evidence amount and that the manuscript's deduction confidence formula needed to match standard NAL. The current code and paper now reflect those corrections. The project also relies on public TCGA-THCA data from the NCI Genomic Data Commons.
