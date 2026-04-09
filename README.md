# NGTA

**NARS-Guided Transformer Attention**

**Dynamic Evidential Routing for clinical transformers under extreme missingness**

**TL;DR:** NGTA is a clinical transformer that does not just rank patients; it tries to tell the truth about its own uncertainty. It estimates epistemic uncertainty with MC Dropout, converts that uncertainty into NARS truth values, injects explicit human-written medical rules at inference time, and feeds the revised confidence back into attention so brittle evidence is downweighted before the final prediction is made.

NGTA is a neurosymbolic clinical prediction architecture that maps neural uncertainty into NARS truth values and feeds revised confidence back into Transformer attention during inference. The repository now supports two benchmarks in parallel:

- `tcga`: TCGA-THCA lymph node metastasis prediction from merged clinical tables plus a mutation-derived binary gene panel
- `wids`: WiDS Datathon 2020 ICU hospital mortality prediction from a high-missingness ICU tabular cohort

The research paper lives in [`paper/main.pdf`](paper/main.pdf), with source in [`paper/main.tex`](paper/main.tex).

## Key Achievements

- **Inference-Time Logic Injection:** Fuses MC-Dropout epistemic uncertainty with NARS symbolic logic and pushes the revised confidence signal directly into Transformer attention during inference.
- **Scale & Safety:** Benchmarked on `91,713` ICU stays and achieved the best reported Expected Calibration Error in this repository evaluation at `0.00583`, versus `0.00750` for the baseline transformer, with a best Brier score of `0.05647`.
- **Glass-Box Activity:** On held-out WiDS ICU data, explicit symbolic rules fired in `8551` of `13757` stays for `13031` total feature-level revisions, showing that the logic layer is active rather than decorative.
- **Multi-Modal Ready:** Proven to handle fused clinical tabular features and genomic mutation matrices on TCGA-THCA, where the same interface remains operational as a clinical-plus-genomic proof of concept.

## Overview

### What it does

NGTA is a medical prediction system for messy hospital-style tables where many values are missing. It uses a Transformer to make predictions, but it does not stop at producing a single risk score. It estimates epistemic uncertainty, checks a set of human-written medical rules, and then uses both pieces of information to adjust how the model pays attention to the input features before the final output is emitted.

### Why it matters

Many clinical AI systems can give a strong prediction even when the data are incomplete or unreliable. That is dangerous in real settings because missing hospital data can produce overconfident probabilities that look trustworthy when they are not. NGTA is designed to separate "high score" from "high confidence" and to expose a human-readable revision path when symbolic rules intervene. That makes the system more useful in high-missingness environments like ICU data, where safer calibration matters as much as raw accuracy.

In standard clinical prediction, models optimize for point-estimate accuracy but lack native mechanisms to express epistemic doubt, leading to overconfident extrapolation when faced with missing features. NGTA is built around the opposite design goal: instead of a black-box predictor that guesses blindly across data gaps, it calculates feature-level uncertainty and can route attention toward explicit medical heuristics when uncertainty is high. In that sense, the repository's core contrast is simple: standard transformers behave like black boxes, while NGTA is designed to behave like a glass box.

### What is novel here

The main novelty is not just "Transformer + rules." The key idea is that NGTA turns neural uncertainty into explicit symbolic truth values from NARS, revises those values with domain rules, and then feeds the revised confidence back into Transformer attention. In simple terms: the model can use both learned patterns and symbolic evidence to decide how much trust to place in each feature at inference time, while also leaving behind an auditable evidential trace.

The end result is not just another tabular model with a rules layer attached to the side. It is an auditable, human-in-the-loop reasoning engine: instead of emitting an overconfident scalar score on missing data, the system exposes what it does not confidently know and provides a direct insertion point for clinicians to inject overriding physiological rules into the inference path itself. The novel outcome of this project is that calibration, clinician steerability, and auditability all appear in the same deployed inference loop.

### How it works

1. The Transformer reads the patient features and predicts risk.
2. Monte Carlo dropout is used to measure how stable that prediction is across repeated passes.
3. That uncertainty is converted into NARS-style truth values: frequency and confidence.
4. If a symbolic rule fires, its truth value is combined with the neural truth value using NARS revision.
5. The revised confidence is used to reweight attention, so uncertain or weakly supported features matter less.
6. The pipeline then evaluates discrimination, Calibration, decision curves, symbolic trigger activity, and baseline comparisons.

### Why there are two datasets

The two benchmarks test different strengths of the architecture:

- `tcga` is the multi-modal proof of concept. It shows that NGTA can fuse clinical variables with a genomic mutation matrix without breaking the mathematical interface.
- `wids` is the primary empirical validation. It shows that the same architecture scales to a much larger ICU dataset with heavy missingness and delivers its strongest win in calibration and safety-oriented reliability.

### What we found

The main result is that NGTA works as intended on both a small multi-modal cancer dataset and a much larger high-missingness ICU dataset, but the two datasets support different claims.

- On `tcga`, the Transformer-based models clearly beat the random-forest baseline. The best AUC was `0.7328`, compared with `0.6613` for random forest. This supports the claim that the architecture can learn useful signal from combined clinical and genomic inputs.
- On `wids`, all Transformer variants were very close on AUC, with the best AUC at `0.8803`. The key result there was calibration: the NARS-gated version achieved the best ECE at `0.00583` versus `0.00750` for the baseline transformer, and the best Brier score at `0.05647`.
- The WiDS result shows the calibration-versus-discrimination story clearly. NGTA did not need to win AUC by a wide margin to matter; it made the predicted probabilities better behaved while remaining competitive on ranking performance. For clinical deployment, that is arguably the more important result, because calibration determines whether a reported risk can actually be trusted as a probability.
- The symbolic rules were not just decorative. On the held-out WiDS test set, ICU rules fired in `8551` of `13757` cases for `13031` total feature-level revisions, which means the neurosymbolic revision path was active at scale rather than sitting unused.
- Taken together, the results support a narrower and more defensible claim than "always better accuracy": NGTA is competitive on discrimination, strongest on calibration, and valuable as a human-auditable instrumentation layer under heavy missingness.

Put differently: the main architectural achievement here is safety-oriented behavior, not just ranking performance. NGTA turns the transformer's attention update from an opaque mapping into a transparent and auditable inference path, where uncertainty is explicit, rule interventions are traceable, and the final probability is better calibrated for clinical use.

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
- Best AUC: `0.7328` for `flat_confidence`
- Best Brier: `0.2109` for `flat_confidence`
- Best ECE: `0.1178` for `flat_confidence`
- Best accuracy: `0.6812`, tied across `baseline`, `flat_confidence`, and `nars_gated`
- Symbolic activity: `42 / 69` held-out cases with any trigger, `79` total feature-level revisions

WiDS ICU full-run summary:

Role in the paper: primary empirical validation for scale, missingness, and calibration

- Split: `64199 / 13757 / 13757` train/validation/test from `91713` labeled rows
- Input width: `16` model features after preprocessing
- Best AUC: `0.8803` for `baseline`
- Best Brier: `0.05647` for `nars_gated`
- Best ECE: `0.00583` for `nars_gated`
- Best accuracy: `0.92862`, tied across `flat_confidence` and `nars_gated`
- Symbolic activity: `8551 / 13757` held-out cases with any trigger, `13031` total feature-level revisions
- Calibration comparison:
  - vs random forest Brier `0.05817 -> 0.05647`
  - vs random forest ECE `0.00737 -> 0.00583`
  - vs baseline transformer Brier `0.05653 -> 0.05647`
  - vs baseline transformer ECE `0.00750 -> 0.00583`
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
- [`src/nars_interface.py`](src/nars_interface.py): NARS truth-value mapping and revision operators
- [`src/attention_hook.py`](src/attention_hook.py): confidence-based attention gating
- [`src/pipeline.py`](src/pipeline.py): training, baselines, evaluation, plotting, and summary generation
- [`paper/main.tex`](paper/main.tex): manuscript source
