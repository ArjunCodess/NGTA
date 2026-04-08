# NGTA

**NARS-Guided Transformer Attention**

NGTA is a neurosymbolic tabular prediction pipeline that maps neural uncertainty into NARS truth values and feeds revised confidence back into Transformer attention during inference. The repository now supports two benchmarks in parallel:

- `tcga`: TCGA-THCA lymph node metastasis prediction from merged clinical tables plus a mutation-derived binary gene panel
- `wids`: WiDS Datathon 2020 ICU hospital mortality prediction from a high-missingness ICU tabular cohort

The executable manuscript companion lives in [`paper/main.tex`](paper/main.tex).

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

- Split: `319 / 69 / 69` train/validation/test from `457` labeled cases
- Best AUC: `0.7328` for `flat_confidence`
- Best Brier: `0.2109` for `flat_confidence`
- Best ECE: `0.1178` for `flat_confidence`
- Best accuracy: `0.6812`, tied across `baseline`, `flat_confidence`, and `nars_gated`
- Symbolic activity: `42 / 69` held-out cases with any trigger, `79` total feature-level revisions

WiDS ICU full-run summary:

- Split: `64199 / 13757 / 13757` train/validation/test from `91713` labeled rows
- Input width: `16` model features after preprocessing
- Best AUC: `0.8803` for `baseline`
- Best Brier: `0.05647` for `nars_gated`
- Best ECE: `0.00583` for `nars_gated`
- Best accuracy: `0.92862`, tied across `flat_confidence` and `nars_gated`
- Symbolic activity: `8551 / 13757` held-out cases with any trigger, `13031` total feature-level revisions
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
