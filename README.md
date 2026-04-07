# NGTA

**NARS-Guided Transformer Attention**

![AUC](https://img.shields.io/badge/AUC%20(NARS--Gated)-0.7336-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-69.57%25-informational)
![Labeled Cases](https://img.shields.io/badge/Labeled%20Cases-457-informational)
![Dataset](https://img.shields.io/badge/Dataset-TCGA--THCA-success)

NGTA is a research implementation of a neurosymbolic tabular Transformer pipeline for **lymph node metastasis prediction in TCGA-THCA**. The current codebase loads five TCGA clinical TSV tables, merges them at the case level, maps model uncertainty into NARS truth values, and feeds confidence back into feature attention during inference.

The executable manuscript companion lives in [paper/main.tex](paper/main.tex). The public repository is `https://github.com/ArjunCodess/NGTA`.

## Overview

The current benchmark uses the TCGA-THCA tables in [`data/`](data):

- `clinical.tsv`
- `exposure.tsv`
- `family_history.tsv`
- `follow_up.tsv`
- `pathology_detail.tsv`

The loader:

- normalizes TCGA missing-value placeholders such as `--`, `'--`, `Not Reported`, and `Unknown`
- collapses each table to one row per `case_submitter_id`
- left-joins the auxiliary tables onto the clinical base table
- drops columns with more than 70% missingness
- derives the binary target from `diagnoses.ajcc_pathologic_n`

Target mapping:

- `N0 -> 0`
- `N1`, `N1a`, `N1b -> 1`
- `NX`, missing, and unmapped values are removed before splitting

On the saved run in this repository, the merged dataset contains `507` raw TCGA cases and `457` labeled cases after target filtering.

## Current Experiment

The default run uses a leakage-aware feature set with `18` case-level variables:

- Numerical: `diagnoses.age_at_diagnosis`, `diagnoses.year_of_diagnosis`, `pathology_details.tumor_length_measurement`, `pathology_details.tumor_width_measurement`, `pathology_details.tumor_depth_measurement`
- Categorical: `demographic.gender`, `demographic.race`, `demographic.ethnicity`, `diagnoses.ajcc_pathologic_t`, `diagnoses.prior_malignancy`, `diagnoses.synchronous_malignancy`, `diagnoses.prior_treatment`, `diagnoses.primary_diagnosis`, `diagnoses.morphology`, `diagnoses.laterality`, `diagnoses.tumor_focality`, `diagnoses.residual_disease`, `pathology_details.extrathyroid_extension`

After one-hot encoding, the model sees `55` numeric input features.

The split is a stratified `70/15/15` train/validation/test split with `seed=0`:

- Train: `319` cases (`158` positive)
- Validation: `69` cases (`34` positive)
- Test: `69` cases (`34` positive)

## Latest Results

Run command:

```bash
python main.py --run-all
```

Test-set metrics from [`results/metrics/metrics.csv`](results/metrics/metrics.csv):

| Variant | AUC | Brier | ECE | Accuracy | 95% AUC CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.6563 | 0.2309 | 0.1389 | 56.52% | 0.5158 to 0.7816 |
| Baseline Transformer | 0.7311 | 0.2062 | 0.1280 | 69.57% | 0.5991 to 0.8504 |
| Flat-Confidence Transformer | 0.7345 | 0.2060 | 0.1284 | 69.57% | 0.6026 to 0.8511 |
| NARS-Gated Transformer | 0.7336 | 0.2059 | 0.1287 | 69.57% | 0.6025 to 0.8509 |

Interpretation:

- The TCGA-THCA benchmark is materially harder than the old hand-curated cohort and no longer produces near-perfect discrimination.
- On this saved split, the random forest baseline underperforms all three Transformer variants on AUC, Brier score, ECE, and accuracy.
- The flat-confidence control is marginally best on AUC.
- The NARS-gated variant is marginally best on Brier score, so its strongest empirical claim here is slightly improved probabilistic reliability rather than clearly better ranking.
- The baseline Transformer is still best on ECE, so the calibration story is mixed rather than universally favorable to NGTA.
- All four 95% bootstrap AUC intervals overlap, so the observed differences should be treated as uncertain on this test split.

Deployment framing:

- In this repository, NGTA should be read as a reliability-oriented interface, not an AUC-maximization claim.
- The NARS layer makes evidential confidence explicit and exportable, which is the main answer to the clinical black-box problem: the model emits both a risk score and a confidence-like control signal that can be inspected and reused in attention.
- The retained feature set is clinicopathologic, not strictly non-invasive or baseline-only.

Gamma ablation from [`results/metrics/gamma_ablation.csv`](results/metrics/gamma_ablation.csv):

| Gamma | Baseline AUC | NARS-Gated AUC | Baseline Brier | NARS-Gated Brier |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.7311 | 0.7336 | 0.20619 | 0.20601 |
| 0.5 | 0.7311 | 0.7336 | 0.20619 | 0.20599 |
| 1.0 | 0.7311 | 0.7336 | 0.20619 | 0.20596 |
| 2.0 | 0.7311 | 0.7336 | 0.20619 | 0.20590 |
| 4.0 | 0.7311 | 0.7361 | 0.20619 | 0.20584 |

On this run, higher `gamma` values improve the NARS-gated Brier score slightly, and `gamma=4.0` gives the best gated AUC among the tested settings.

## Outputs

Charts:

- [charts/roc_curve.png](charts/roc_curve.png)
- [charts/calibration_curve.png](charts/calibration_curve.png)
- [charts/training_history.png](charts/training_history.png)
- [charts/gamma_ablation_auc.png](charts/gamma_ablation_auc.png)
- [charts/decision_curve.png](charts/decision_curve.png)

Metrics and traces:

- [results/metrics/run_summary.json](results/metrics/run_summary.json)
- [results/metrics/metrics.csv](results/metrics/metrics.csv)
- [results/metrics/training_history.csv](results/metrics/training_history.csv)
- [results/metrics/gamma_ablation.csv](results/metrics/gamma_ablation.csv)
- [results/metrics/decision_curve.csv](results/metrics/decision_curve.csv)
- [results/metrics/calibration_reliability.csv](results/metrics/calibration_reliability.csv)
- [results/traces/split_summary.json](results/traces/split_summary.json)
- [results/traces/preprocessing_metadata.json](results/traces/preprocessing_metadata.json)
- [results/traces/test_predictions.csv](results/traces/test_predictions.csv)

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --run-all
```

Useful flags:

- `--data-dir`: directory containing the TCGA TSV tables
- `--output-dir`: alternate root for charts and results
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`
- `--mc-samples`, `--gamma`, `--seed`
- `--d-model`, `--num-heads`, `--num-layers`, `--dropout`, `--patience`

Example:

```bash
python main.py --run-all --gamma 4.0 --mc-samples 25
```

## Repository Layout

- [main.py](main.py): CLI entry point
- [src/data_loader.py](src/data_loader.py): TCGA ingestion, case-level merging, preprocessing, and stratified splitting
- [src/neural_encoder.py](src/neural_encoder.py): tabular Transformer with MC-dropout inference
- [src/nars_interface.py](src/nars_interface.py): NARS truth-value mapping and revision operators
- [src/attention_hook.py](src/attention_hook.py): confidence-based attention gating
- [src/pipeline.py](src/pipeline.py): training, evaluation, and artifact generation
- [paper/main.tex](paper/main.tex): manuscript source

## Notes

- The current implementation focuses on the uncertainty-to-NARS and confidence-gated attention path on TCGA-THCA. It does not ship a handcrafted THCA symbolic rule base.
- The saved classical baseline is a validation-tuned `RandomForestClassifier`, selected by validation Brier score on the same encoded train/validation split used by the Transformer pipeline.
- The auxiliary TCGA tables are still merged even though most retained model features come from the clinical and pathology-detail tables after missingness filtering.
