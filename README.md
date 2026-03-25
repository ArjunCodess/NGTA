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
- `30` MC-dropout inference passes
- Transformer config: `d_model=64`, `num_heads=4`, `num_layers=2`, `dropout=0.2`
- NARS attention gate: `gamma=2.0`
- `16` features used in the saved run

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

## Results

The latest full run was:

```bash
python main.py --run-all
```

The saved metrics are in [results/metrics/metrics.csv](results/metrics/metrics.csv), and the full run summary is in [results/metrics/run_summary.json](results/metrics/run_summary.json).

### Test-set metrics

| Variant | AUC | Brier Score | Accuracy | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Transformer | 0.9873 | 0.04153 | 95.83% | 100.00% | 90.48% | 0.9500 |
| NARS-Gated Transformer | 0.9891 | 0.04187 | 95.83% | 100.00% | 90.48% | 0.9500 |

### What changed with NARS gating

- AUC improved slightly from `0.9873` to `0.9891`.
- Accuracy stayed the same at `95.83%`.
- Recall stayed the same at `100%`.
- Precision stayed the same at `90.48%`.
- F1 stayed the same at `0.95`.
- Brier score changed slightly in the wrong direction, from `0.04153` to `0.04187`.

### Practical interpretation

The saved run shows a **small ranking improvement** from NARS-guided attention, but not a threshold-level classification improvement on this test split. At the default threshold of `0.5`, both models produce the same confusion pattern:

- true negatives: `27`
- false positives: `2`
- false negatives: `0`
- true positives: `19`

That means the current evidence supports a narrow claim: in this run, NARS gating slightly improved ordering of cases by score, but it did **not** change the final hard predictions. This is still useful because rare-disease workflows often depend on ranking and triage, but the effect should be interpreted as **modest and preliminary**, not as a dramatic performance jump.

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
python main.py --run-all --epochs 60 --mc-samples 30 --gamma 2.0
python main.py --run-all --batch-size 16 --seed 0
```

## Why This Project Matters

Rare-disease models are difficult to trust when the data is small, the uncertainty is high, and the clinical stakes are non-trivial. NGTA is built around the idea that uncertainty should be explicit, inspectable, and usable inside the model rather than treated as an afterthought.

NARS provides a useful representation for that:

- `f` captures the balance of evidence,
- `c` captures the strength of evidential support.

By feeding that confidence back into attention, NGTA explores a pragmatic neurosymbolic approach to clinical prediction under data scarcity.
