# NGTA

**NARS-Guided Transformer Attention**

![AUC](https://img.shields.io/badge/AUC%20(NARS--Gated)-0.9891-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-95.83%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall-100%25-success)
![Samples](https://img.shields.io/badge/Samples-149-informational)

NGTA is a research implementation of a neurosymbolic idea: use a Transformer for **medullary thyroid carcinoma (MTC) diagnosis**, estimate uncertainty with Monte Carlo dropout, convert that uncertainty into **NARS** truth values, and then use NARS confidence to guide attention during inference.

The repository contains:

- the manuscript in [paper/main.tex](paper/main.tex),
- the bibliography in [paper/references.bib](paper/references.bib),
- the runnable pipeline in [main.py](main.py) and [src](src),
- and the generated outputs in [charts](charts) and [results](results).

## What NGTA Means

**NGTA = NARS-Guided Transformer Attention.**

The project combines:

- a tabular Transformer encoder,
- Monte Carlo dropout for predictive uncertainty,
- a mapping from neural outputs to NARS truth values `(f, c)`,
- and confidence-based attention gating using `c^gamma`.

The goal is not only to predict, but to make the model's evidential support more explicit and easier to inspect in a rare-disease setting.

## Paper Basis

This repository implements the ideas in the paper:

**A Formal Interface That Maps NARS Truth Values to Uncertainty-Conditioned Transformer Attention for Rare Disease Prediction**

In the current manuscript, the application case is **MTC diagnosis** from structured clinical features, serum markers, and ultrasound-related evidence.

## Research Question

> How can a transformer expose its uncertainty in a form that supports symbolic evidence aggregation, and how can aggregated evidential confidence control attention during inference?

That question is the core of NGTA. The repository is the executable version of that interface.

## What Was Tested

The saved run evaluates a binary **MTC diagnosis** pipeline on [data.csv](data.csv).

Verified setup from [results/metrics/run_summary.json](results/metrics/run_summary.json):

- `149` total rows
- `80` training rows
- `21` validation rows
- `48` test rows
- held-out test studies: `study_2`, `study_8`
- `50` MC-dropout inference passes
- Transformer config: `d_model=64`, `num_heads=4`, `num_layers=2`, `dropout=0.2`
- NARS attention gate: `gamma=2.0`
- `16` features used in the saved run

The current default configuration is the `mc_50` setting:

- `mc_samples=50`
- `gamma=2.0`
- `d_model=64`
- `num_heads=4`
- `num_layers=2`
- `dropout=0.2`
- `epochs=60`
- `batch_size=32`
- `learning_rate=0.001`
- `patience=12`

The saved feature set is:

- `gender`
- `ret_variant`
- `ret_risk_level`
- `calcitonin_elevated`
- `cea_elevated`
- `cea_imputed_flag`
- `thyroid_nodules_present`
- `family_history_mtc`
- `c_cell_disease`
- `men2_syndrome`
- `pheochromocytoma`
- `hyperparathyroidism`
- `age_group`
- `age`
- `calcitonin_level_numeric`
- `cea_level_numeric`

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

## Results

The latest full run was:

```bash
python main.py --run-all
```

The saved metrics are in [results/metrics/metrics.csv](results/metrics/metrics.csv), and the full run summary is in [results/metrics/run_summary.json](results/metrics/run_summary.json).

### Test-set metrics

| Variant | AUC | Brier Score | Accuracy | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Transformer | 0.9873 | 0.04121 | 95.83% | 100.00% | 90.48% | 0.9500 |
| NARS-Gated Transformer | 0.9891 | 0.04115 | 95.83% | 100.00% | 90.48% | 0.9500 |

### What changed with NARS gating

- AUC improved slightly from `0.9873` to `0.9891`.
- Accuracy stayed the same at `95.83%`.
- Recall stayed the same at `100%`.
- Precision stayed the same at `90.48%`.
- F1 stayed the same at `0.95`.
- Brier score improved slightly, from `0.04121` to `0.04115`.

### Practical interpretation

The saved run shows a **small ranking improvement** from NARS-guided attention, but not a threshold-level classification improvement on this test split. At the default threshold of `0.5`, both models produce the same confusion pattern:

- true negatives: `27`
- false positives: `2`
- false negatives: `0`
- true positives: `19`

That means the current evidence supports a narrow claim: in this run, NARS gating slightly improved ordering of cases by score, but it did **not** change the final hard predictions. This is still useful because rare-disease workflows often depend on ranking and triage, but the effect should be interpreted as **modest and preliminary**, not as a dramatic performance jump.

### Default parameter setting

The model defaults are:

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

These values are the defaults in the CLI and pipeline, and the saved results in this repository were generated with this configuration.

## How The Method Works

At a high level, NGTA does the following:

1. Load structured patient data and create a study-aware split.
2. Train a tabular Transformer on the MTC diagnosis label.
3. Keep dropout active at inference and run repeated stochastic forward passes.
4. Estimate a mean probability `p` and epistemic variance `sigma^2`.
5. Map `(p, sigma^2)` to a NARS truth value `(f, c)`.
6. Derive per-feature attention confidence from MC attention variance.
7. Reweight attention using `c^gamma`.
8. Save metrics, traces, and figures.

The main interface equations are:

- neural output to NARS truth value: `(f, c) = (p, (p(1-p)+epsilon)/(p(1-p)+epsilon+sigma^2))`
- attention gate: `alpha_new = alpha * c^gamma / sum(alpha * c^gamma)`

## Repository Layout

- [main.py](main.py): CLI entry point.
- [src/data_loader.py](src/data_loader.py): loading, preprocessing, and study-aware splitting.
- [src/neural_encoder.py](src/neural_encoder.py): tabular Transformer with MC-dropout inference.
- [src/nars_interface.py](src/nars_interface.py): NARS truth-value mapping and revision.
- [src/attention_hook.py](src/attention_hook.py): confidence-based attention gating.
- [src/pipeline.py](src/pipeline.py): end-to-end training, evaluation, charts, and trace generation.
- [paper/main.tex](paper/main.tex): paper source.
- [paper/references.bib](paper/references.bib): BibTeX bibliography.

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

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py --run-all
```

Useful overrides:

```bash
python main.py --run-all --epochs 60 --mc-samples 50 --gamma 2.0
python main.py --run-all --batch-size 16 --seed 0
```

## Why This Project Matters

Rare-disease models are difficult to trust when the data is small, the uncertainty is high, and the clinical stakes are non-trivial. NGTA is built around the idea that uncertainty should be explicit, inspectable, and usable inside the model rather than treated as an afterthought.

NARS provides a useful representation for that:

- `f` captures the balance of evidence,
- `c` captures the strength of evidential support.

By feeding that confidence back into attention, NGTA explores a pragmatic neurosymbolic approach to clinical prediction under data scarcity.
