# WiDS development results

The current WiDS implementation and available-data studies are complete. All conditions use the original data and frozen fitted preprocessing. No additional download or source clarification is an execution prerequisite. Prototype source conventions remain stated limitations, and no independent clinical validation or human-study benefit is claimed.

There are 20 original fits: four conditions with five seeds each. Every condition uses the same 14,225 held-out encounters; repeated seeds and conditions are not additional patients. Original inference has 50 matched passes and three dropout draws. The new encoder extension reuses each checkpoint and its primary 50-pass RNG draw, without retraining.

## Original readout results

| Condition | Ungated AUROC | Ungated Brier | MC-only Brier | NARS Brier | NARS-minus-MC Brier, hierarchical 95% interval |
|---|---:|---:|---:|---:|---|
| KNN | 0.875721 | 0.057830 | 0.057828 | 0.057828 | -2.431e-07 [-6.913e-07, 1.637e-07] |
| Median | 0.874062 | 0.058157 | 0.058158 | 0.058157 | -4.039e-07 [-1.190e-06, 5.787e-07] |
| KNN, no APACHE | 0.841118 | 0.063006 | 0.063015 | 0.063015 | -1.682e-07 [-4.311e-07, 3.115e-07] |
| Median, no APACHE | 0.842176 | 0.063039 | 0.063051 | 0.063050 | -2.350e-07 [-7.851e-07, 6.530e-07] |

The point estimates are descriptive seed averages. Original symbolic intervals resample hospitals, training seeds and dropout repeats. None of the four original conditions establishes the 0.0001 symbolic benefit margin.

## Additional encoder-level intervention

| Condition | Encoder MC Brier | Encoder NARS Brier | Encoder NARS-minus-MC Brier | Family lower | Family upper |
|---|---:|---:|---:|---:|---:|
| KNN | 0.057839 | 0.057837 | -1.871e-06 | -4.420e-06 | 3.536e-07 |
| Median | 0.058190 | 0.058188 | -2.495e-06 | -8.053e-06 | 2.739e-06 |
| KNN, no APACHE | 0.063030 | 0.063028 | -2.058e-06 | -4.929e-06 | 2.863e-06 |
| Median, no APACHE | 0.063070 | 0.063067 | -2.890e-06 | -8.317e-06 | 9.932e-06 |

Uniform encoder gates reproduce the original per-pass probabilities with maximum residual 0. Confidence gates bias feature keys in every encoder layer and recompute contextual representations. This changes the intervention location, so encoder-versus-readout differences cannot be attributed solely to symbolic content. Full contrasts and all five-method metrics are retained in the aggregate JSON.

## Missingness and uncertainty

- knn: 1/15 seed/scenario combinations meet the saved paired degradation/harm criteria. These are repeated development simulations, not independent populations.
- median: 12/15 seed/scenario combinations meet the saved paired degradation/harm criteria. These are repeated development simulations, not independent populations.

Five-model uncertainty comparisons, calibration, selective-risk curves, subgroup metrics, decision curves, rule controls, 100 predicate permutations and the creatinine-rule-off sensitivity remain in the existing study outputs. The original negative symbolic-benefit and general-robustness findings are preserved.

## Reproduce

```powershell
python scripts/evaluate_wids_encoder.py
python scripts/summarize_wids.py
```

These commands use the original permitted local research artifacts; they do not download data. Private probability arrays stay under `results/private/wids_encoder_study/`. All outputs remain development evidence on inspected cases.
