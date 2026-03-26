# NGTA

**NARS-Guided Transformer Attention**

![AUC](https://img.shields.io/badge/AUC%20(NARS--Gated)-0.9891-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-95.83%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall-100%25-success)
![Samples](https://img.shields.io/badge/Samples-149-informational)

NGTA is a research implementation of a neurosymbolic pipeline for **medullary thyroid carcinoma (MTC) diagnosis**. It combines a tabular Transformer, Monte Carlo dropout uncertainty estimation, and **Non-Axiomatic Reasoning System (NARS)** truth values so that evidential confidence can be fed back into attention during inference.

The project is the executable companion to the manuscript in [paper/main.tex](paper/main.tex), and the public code repository is `https://github.com/ArjunCodess/NGTA`.

## Table Of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [What Was Tested](#what-was-tested)
- [Results](#results)
- [Dataset Provenance](#dataset-provenance)
- [Built With](#built-with)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Output Artifacts](#output-artifacts)
- [Repository Tags](#repository-tags)
- [Contributing](#contributing)
- [License](#license)

## Overview

**NGTA = NARS-Guided Transformer Attention.**

The method combines:

- a tabular Transformer encoder for structured clinical prediction,
- Monte Carlo dropout for epistemic uncertainty estimation,
- a mapping from neural outputs to NARS truth values `(f, c)`,
- and confidence-based attention gating using `c^gamma`.

The goal is not only to classify cases, but to make uncertainty explicit, inspectable, and usable inside the prediction pipeline. In a rare-disease setting, that matters because ranking, triage, and evidential support can be as important as a single hard label.

## Research Question

> How can a transformer expose its uncertainty in a form that supports symbolic evidence aggregation, and how can aggregated evidential confidence control attention during inference?

That question is the core of NGTA. The repository implements that interface and evaluates it on a small structured MTC cohort derived from published hereditary MTC and MEN2 studies.

## What Was Tested

The current saved run evaluates binary **MTC diagnosis** on [data.csv](data.csv) using a study-aware split recorded in [results/metrics/run_summary.json](results/metrics/run_summary.json). The run contains `149` total cases, split into `80` training rows, `21` validation rows, and `48` held-out test rows, with `study_2` and `study_8` reserved for testing. The default configuration uses `50` Monte Carlo dropout passes, a Transformer with `d_model=64`, `num_heads=4`, `num_layers=2`, and `dropout=0.2`, plus a NARS confidence gate with `gamma=2.0`.

The model uses `16` structured variables spanning demographics, RET-related family/genotype context, endocrine comorbidity indicators, and biomarker features such as calcitonin and CEA. The saved results in this repository were generated with the default configuration shown below.

## Results

The latest full run was produced with:

```bash
python main.py --run-all
```

Saved metrics:

- [results/metrics/metrics.csv](results/metrics/metrics.csv)
- [results/metrics/run_summary.json](results/metrics/run_summary.json)

### Test-Set Metrics

| Variant | AUC | Brier Score | Accuracy | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Transformer | 0.9873 | 0.04121 | 95.83% | 100.00% | 90.48% | 0.9500 |
| NARS-Gated Transformer | 0.9891 | 0.04115 | 95.83% | 100.00% | 90.48% | 0.9500 |

### Bootstrap AUC Confidence Intervals

Using `1000` bootstrap resamples of the `48` held-out test cases, the saved run gives the following 95% AUC confidence intervals:

| Variant | AUC | 95% Bootstrap CI |
| --- | ---: | ---: |
| Baseline Transformer | 0.9873 | 0.9590 to 1.0000 |
| NARS-Gated Transformer | 0.9891 | 0.9637 to 1.0000 |

### Interpretation

The main empirical effect is a **small ranking improvement** from the NARS-guided attention layer: AUC rises from `0.9873` to `0.9891`, and the Brier score improves slightly from `0.04121` to `0.04115`. The thresholded predictions on this held-out split remain the same, but the important contribution is that NGTA makes epistemic uncertainty explicit through NARS confidence and uses that confidence to produce slightly better risk ordering.

The bootstrap intervals also matter for interpretation. The baseline interval (`0.9590` to `1.0000`) and the NARS-gated interval (`0.9637` to `1.0000`) **do overlap**, so this run should be described as showing an observed improvement, but not a clearly separated one by the bootstrap-CI check. In other words, the result is promising but still uncertain on a `48`-case test set.

### Default Parameter Setting

The default model configuration is:

- `epochs=60`
- `batch_size=32`
- `learning_rate=0.001`
- `mc_samples=50`
- `gamma=2.0`
- `d_model=64`
- `num_heads=4`
- `num_layers=2`
- `dropout=0.2`
- `patience=12`

These values are the defaults in [main.py](main.py) and [src/pipeline.py](src/pipeline.py), and the saved repository results were generated with this setup.

## Dataset Provenance

The structured dataset in [data.csv](data.csv) was assembled from ten published hereditary MTC and MEN2 case reports, family cohorts, and clinical follow-up studies. Special thanks are due to the authors of these studies for reporting the clinical details that made this small structured cohort possible.

1. *Medullary Thyroid Carcinoma Associated with Germline RETK666N Mutation*. *Thyroid* (2016). DOI: `10.1089/thy.2016.0374`
2. *Long-term outcome in 46 gene carriers of hereditary medullary thyroid carcinoma after prophylactic thyroidectomy: impact of individual RET genotype*. *European Journal of Endocrinology* (2006). DOI: `10.1530/eje.1.02216`
3. Qi et al. *RET mutation p.S891A in a Chinese family with familial medullary thyroid carcinoma and associated cutaneous amyloidosis binding OSMR variant p.G513D*. *Oncotarget* (2015). DOI: `10.18632/oncotarget.4992`
4. Florescu et al. *Endocrine Perspective of Cutaneous Lichen Amyloidosis: RET-C634 Pathogenic Variant in Multiple Endocrine Neoplasia Type 2*. *Clinics and Practice* (2024). DOI: `10.3390/clinpract14060179`
5. La Greca et al. *MEN2 phenotype in a family with germline heterozygous rare RET K666N variant*. *Endocrinology, Diabetes & Metabolism Case Reports* (2024). DOI: `10.1530/EDM-24-0009`
6. Vijayan et al. *A rare RET mutation in an Indian pedigree with familial medullary thyroid carcinoma*. *Indian Journal of Cancer* (2021). DOI: `10.4103/ijc.IJC_639_19`
7. Zhang et al. *C634Y mutation in RET-induced multiple endocrine neoplasia type 2A: A case report*. *World Journal of Clinical Cases* (2024). DOI: `10.12998/wjcc.v12.i15.2627`
8. Shankar et al. *Medullary thyroid cancer in a 9-week-old infant with familial MEN 2B: Implications for timing of prophylactic thyroidectomy*. *International Journal of Pediatric Endocrinology* (2012). DOI: `10.1186/1687-9856-2012-25`
9. Schulte et al. *The Clinical Spectrum of Multiple Endocrine Neoplasia Type 2a Caused by the Rare Intracellular RET Mutation S891A*. *The Journal of Clinical Endocrinology & Metabolism* (2010). DOI: `10.1210/jc.2010-0375`
10. Qi et al. *The rare intracellular RET mutation p.S891A in a Chinese Han family with familial medullary thyroid carcinoma*. *Journal of Biosciences* (2014). DOI: `10.1007/s12038-014-9428-x`

## Built With

The pipeline is built primarily with:

- **Python** for the CLI and orchestration
- **PyTorch** for the tabular Transformer and MC-dropout inference
- **scikit-learn** for splitting, preprocessing helpers, and evaluation metrics
- **Pandas** for tabular data handling and artifact export
- **NumPy** for numerical operations
- **Matplotlib** for ROC, calibration, and training-history plots
- **Seaborn** for plotting support

The exact installable dependencies are listed in [requirements.txt](requirements.txt).

## Repository Layout

- [main.py](main.py): CLI entry point
- [src/data_loader.py](src/data_loader.py): loading, preprocessing, and study-aware splitting
- [src/neural_encoder.py](src/neural_encoder.py): tabular Transformer with MC-dropout inference
- [src/nars_interface.py](src/nars_interface.py): NARS truth-value mapping and revision
- [src/attention_hook.py](src/attention_hook.py): confidence-based attention gating
- [src/pipeline.py](src/pipeline.py): end-to-end training, evaluation, charts, and trace generation
- [paper/main.tex](paper/main.tex): manuscript source
- [paper/references.bib](paper/references.bib): BibTeX bibliography

## Prerequisites

Before running the project, make sure you have:

- Python `3` available on your machine
- `git` installed if you plan to clone the repository
- a virtual environment tool such as `venv`, `virtualenv`, or Conda

## Quick Start

Clone the repository:

```bash
git clone https://github.com/ArjunCodess/NGTA.git
cd NGTA
```

Create and activate a virtual environment with `venv`:

```bash
python -m venv venv
```

On Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py --run-all
```

Conda users can use an equivalent flow:

```bash
conda create -n ngta python=3.10
conda activate ngta
pip install -r requirements.txt
```

## Usage

The main entry point is:

```bash
python main.py --run-all
```

At present, invoking `python main.py` already executes the end-to-end pipeline. The `--run-all` flag is still included in examples because it is the explicit project command used throughout this repository.

### Common Examples

Run with the default configuration:

```bash
python main.py --run-all
```

Increase the strength of confidence gating:

```bash
python main.py --run-all --gamma 4.0
```

Use fewer MC-dropout samples for a faster but noisier estimate:

```bash
python main.py --run-all --mc-samples 15
```

Change batch size or training duration:

```bash
python main.py --run-all --batch-size 16 --epochs 80
```

Write artifacts to a different directory:

```bash
python main.py --run-all --output-dir runs/exp_01
```

### CLI Arguments

| Argument | Meaning | Practical Effect |
| --- | --- | --- |
| `--run-all` | Runs the end-to-end pipeline | Trains the model, evaluates both variants, and writes charts and traces |
| `--data-path` | Path to the input CSV | Lets you point the pipeline at a different dataset file |
| `--output-dir` | Root directory for generated artifacts | Useful for separating experiments |
| `--epochs` | Maximum training epochs | Higher values allow longer training; early stopping may still stop sooner |
| `--batch-size` | Training batch size | Smaller batches can improve stability on small datasets but run slower |
| `--learning-rate` | Optimizer step size | Larger values train faster but can destabilize convergence |
| `--weight-decay` | L2-style regularization strength | Helps limit overfitting |
| `--mc-samples` | Number of stochastic inference passes | Higher values give a smoother mean probability and more stable variance estimate, but increase runtime |
| `--gamma` | Exponent in the attention gate `c^gamma` | Larger values make the gate more selective, amplifying differences between high- and low-confidence features |
| `--seed` | Random seed | Helps reproducibility across runs |
| `--patience` | Early-stopping patience | Controls how long training continues without validation improvement |
| `--validation-size` | Fraction of the training portion reserved for validation | Changes how much data is used for model selection |
| `--d-model` | Transformer hidden width | Larger values increase capacity and computation |
| `--num-heads` | Number of attention heads | Changes how feature interactions are partitioned across attention channels |
| `--num-layers` | Number of Transformer encoder layers | More layers increase model depth |
| `--dropout` | Dropout probability | Affects both regularization during training and stochasticity during MC-dropout inference |

### How To Think About `--gamma` And `--mc-samples`

- Increasing `--gamma` from `2.0` to `4.0` makes the NARS gate more aggressive. High-confidence features keep more attention mass, while lower-confidence features are suppressed more strongly.
- Decreasing `--gamma` toward `1.0` makes the gate gentler and keeps the final attention pattern closer to the baseline Transformer.
- Increasing `--mc-samples` reduces noise in the estimated predictive variance and in the per-feature attention summaries, but it increases inference time.
- Decreasing `--mc-samples` makes experiments faster, but the uncertainty estimates can become less stable, which directly affects the NARS confidence values.

## Output Artifacts

Charts:

- [charts/roc_curve.png](charts/roc_curve.png)
- [charts/calibration_curve.png](charts/calibration_curve.png)
- [charts/training_history.png](charts/training_history.png)

Metrics and traces:

- [results/metrics/metrics.csv](results/metrics/metrics.csv)
- [results/metrics/run_summary.json](results/metrics/run_summary.json)
- [results/metrics/training_history.csv](results/metrics/training_history.csv)
- [results/traces/split_summary.json](results/traces/split_summary.json)
- [results/traces/preprocessing_metadata.json](results/traces/preprocessing_metadata.json)
- [results/traces/test_predictions.csv](results/traces/test_predictions.csv)

## Contributing

Contributions are welcome if they improve correctness, documentation quality, reproducibility, or the experimental pipeline. Useful contributions include:

- clearer evaluation and calibration analysis
- stronger ablations and external validation
- bug fixes in preprocessing, training, or artifact generation
- documentation improvements

If you contribute, keep changes focused, document the motivation clearly, and include any relevant result or artifact changes.
