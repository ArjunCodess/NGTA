# NGTA

**NARS-Guided Transformer Attention**

![AUC](https://img.shields.io/badge/AUC%20(NARS--Gated)-0.7261-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-68.12%25-informational)
![Labeled Cases](https://img.shields.io/badge/Labeled%20Cases-457-informational)
![Dataset](https://img.shields.io/badge/Dataset-TCGA--THCA-success)

NGTA is a research implementation of a neurosymbolic multi-modal Transformer pipeline for **lymph node metastasis prediction in TCGA-THCA**. The current codebase loads five TCGA clinical TSV tables, fuses them with a somatic-mutation MAF-derived binary gene matrix at the case level, maps model uncertainty into NARS truth values, and feeds confidence back into feature attention during inference.

The executable manuscript companion lives in [paper/main.tex](paper/main.tex). The public repository is `https://github.com/ArjunCodess/NGTA`.

## Overview

The current benchmark uses the TCGA-THCA tables in [`data/`](data):

- `clinical.tsv`
- `exposure.tsv`
- `family_history.tsv`
- `follow_up.tsv`
- `pathology_detail.tsv`
- `*.maf` somatic mutation file(s)

The loader:

- normalizes TCGA missing-value placeholders such as `--`, `'--`, `Not Reported`, and `Unknown`
- collapses each table to one row per `case_submitter_id`
- left-joins the auxiliary tables onto the clinical base table
- drops columns with more than 70% missingness
- derives the binary target from `diagnoses.ajcc_pathologic_n`
- reads MAF files with `comment='#'`, extracts `case_submitter_id` from `Tumor_Sample_Barcode`, filters to functional mutations, selects the top mutated genes, and left-joins a binary mutation matrix onto the clinical frame

Target mapping:

- `N0 -> 0`
- `N1`, `N1a`, `N1b -> 1`
- `NX`, missing, and unmapped values are removed before splitting

On the saved run in this repository, the merged dataset contains `507` raw TCGA cases and `457` labeled cases after target filtering.

## Current Experiment

The current saved run uses a leakage-aware multi-modal feature set with `18` clinical variables plus `50` genomic mutation indicators derived from the currently available MAF files:

- Numerical: `diagnoses.age_at_diagnosis`, `diagnoses.year_of_diagnosis`, `pathology_details.tumor_length_measurement`, `pathology_details.tumor_width_measurement`, `pathology_details.tumor_depth_measurement`
- Categorical: `demographic.gender`, `demographic.race`, `demographic.ethnicity`, `diagnoses.ajcc_pathologic_t`, `diagnoses.prior_malignancy`, `diagnoses.synchronous_malignancy`, `diagnoses.prior_treatment`, `diagnoses.primary_diagnosis`, `diagnoses.morphology`, `diagnoses.laterality`, `diagnoses.tumor_focality`, `diagnoses.residual_disease`, `pathology_details.extrathyroid_extension`
- Genomic binary features: top-50 filtered mutation indicators including `genomic_mutation__BRAF`, `genomic_mutation__ZNF804A`, `genomic_mutation__ST18`, `genomic_mutation__CYP2C9`, `genomic_mutation__OR4D5`, `genomic_mutation__TP73`, `genomic_mutation__RCC1`, `genomic_mutation__PI4KB`, `genomic_mutation__RFX5`, `genomic_mutation__SPDYA`, and 40 additional binary gene columns saved in [`results/traces/preprocessing_metadata.json`](results/traces/preprocessing_metadata.json)

After preprocessing, the model sees `105` numeric input features:

- `5` scaled clinical numeric features
- `50` pass-through genomic binary features
- `50` one-hot encoded clinical categorical features

The current run consumed these two MAF files:

- `1feaf21d-8259-4c70-bcb7-7e3fb0887ea4.wxs.aliquot_ensemble_masked.maf`
- `c2e3ad0c-d449-449d-b3a3-793b90bdc793.wxs.aliquot_ensemble_masked.maf`

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
| Random Forest | 0.6613 | 0.2200 | 0.1597 | 59.42% | 0.5101 to 0.7908 |
| Baseline Transformer | 0.7252 | 0.2118 | 0.1390 | 66.67% | 0.5931 to 0.8454 |
| Flat-Confidence Transformer | 0.7261 | 0.2118 | 0.1390 | 68.12% | 0.5934 to 0.8468 |
| NARS-Gated Transformer | 0.7261 | 0.2122 | 0.1403 | 66.67% | 0.5957 to 0.8460 |

Interpretation:

- The TCGA-THCA benchmark is materially harder than the old hand-curated cohort and no longer produces near-perfect discrimination.
- On this saved split, the random forest baseline underperforms the Transformer family on AUC, Brier score, accuracy, and ECE.
- The flat-confidence and NARS-gated variants tie on AUC to four decimal places shown in the saved summary.
- The flat-confidence control is now marginally best on both Brier score and ECE, and it also reaches the best test accuracy at `68.12%`.
- The NARS-gated variant no longer wins the calibration metrics at `gamma=2.0`; its strongest metric in this saved run is tied AUC rather than Brier or ECE.
- All four 95% bootstrap AUC intervals overlap, so the observed differences should be treated as uncertain on this test split.

Deployment framing:

- In this repository, NGTA should be read as a reliability-oriented interface, not an AUC-maximization claim.
- The NARS layer makes evidential confidence explicit and exportable, which is the main answer to the clinical black-box problem: the model emits both a risk score and a confidence-like control signal that can be inspected and reused in attention.
- The retained feature set is clinicopathologic, not strictly non-invasive or baseline-only.
- The current documentation reflects a broader multi-modal run with the genomic branch expanded to the top-50 gene panel from the available MAF files in `data/`.

Gamma ablation from [`results/metrics/gamma_ablation.csv`](results/metrics/gamma_ablation.csv):

| Gamma | Baseline AUC | NARS-Gated AUC | Baseline Brier | NARS-Gated Brier |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.7252 | 0.7261 | 0.21184 | 0.21187 |
| 0.5 | 0.7252 | 0.7261 | 0.21184 | 0.21192 |
| 1.0 | 0.7252 | 0.7261 | 0.21184 | 0.21200 |
| 2.0 | 0.7252 | 0.7261 | 0.21184 | 0.21219 |
| 4.0 | 0.7252 | 0.7294 | 0.21184 | 0.21264 |

On this run, higher `gamma` values improve the NARS-gated AUC slightly, with `gamma=4.0` giving the best gated AUC. In contrast to the prior run, stronger gating worsens the NARS-gated Brier score here, so the gamma trade-off is now discrimination-versus-calibration rather than a consistent gain on both.

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
- The downloader-integrated pipeline now automatically ensures TCGA-THCA MAF availability before preprocessing. With the current local files, that genomic branch expands to the intended top-50-gene binary panel automatically.
- The auxiliary TCGA tables are still merged even though most retained model features come from the clinical and pathology-detail tables after missingness filtering.
