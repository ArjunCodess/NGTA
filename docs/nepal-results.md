# Nepal ICU sensitivity

This completed experiment evaluates frozen WiDS models against a different ICU-mortality endpoint. It is an exploratory sensitivity, without an independent hospital-mortality validation claim.

The checksum-pinned public workbook contains 77,610 daily rows and 18,603 admission identifiers. Selecting one physical row after consistency checks leaves 18,111 recorded adults with 3,316 ICU deaths. 492 identifiers fail the recorded adult-age criterion; no unknown outcome is converted to survival. The selected admission and outcome fields have zero within-identifier conflicts.

Two scenarios were fixed before model scoring: demographics uses age, sex and elective surgery; admission core adds first heart rate and systolic pressure. The latter substitute first measurements for WiDS day-one extrema, so the windows are not equivalent. BMI, temperature, lactate, BUN, creatinine, glucose, WBC, SpO2 and platelets remain unavailable. No APACHE score is supplied. Frozen fitted preprocessing fills unsupported inputs; observed-only rules never treat these imputed values as measured evidence.

The publisher-linked CCAA dictionary specifies age in years, heart rate in beats/minute and systolic pressure in mmHg. Its Fahrenheit temperature and micromol/L creatinine definitions conflict with large parts of the released numeric scales, and the workbook has no per-record unit flags. WBC scales are also unresolved. These fields were excluded rather than converted using outcome performance or an inferred patient-level unit. The source-level scale audit and analysis amendment are retained.

## Results

Each condition reuses five frozen no-APACHE checkpoints and 50 matched dropout passes, without training or recalibration. Seed means reuse the same admissions; they are not additional participants.

| Scenario | Imputer | Method | AUROC mean (seed SD) | Brier mean (seed SD) | Log loss |
|---|---|---|---:|---:|---:|
| demographics | median | baseline | 0.564653 (0.000357) | 0.173390 (0.001230) | 0.719080 |
| demographics | median | mc_confidence_only | 0.564650 (0.000352) | 0.173376 (0.001241) | 0.718836 |
| demographics | median | nars_gated | 0.564649 (0.000351) | 0.173375 (0.001241) | 0.718832 |
| demographics | knn | baseline | 0.559668 (0.002710) | 0.165686 (0.001578) | 0.610251 |
| demographics | knn | mc_confidence_only | 0.559702 (0.002687) | 0.165692 (0.001600) | 0.610336 |
| demographics | knn | nars_gated | 0.559702 (0.002687) | 0.165692 (0.001600) | 0.610335 |
| admission_core | median | baseline | 0.621074 (0.006242) | 0.172782 (0.001368) | 0.714174 |
| admission_core | median | mc_confidence_only | 0.620967 (0.006291) | 0.172772 (0.001379) | 0.713973 |
| admission_core | median | nars_gated | 0.620965 (0.006292) | 0.172771 (0.001380) | 0.713960 |
| admission_core | knn | baseline | 0.610208 (0.001821) | 0.168023 (0.001507) | 0.661195 |
| admission_core | knn | mc_confidence_only | 0.610329 (0.001801) | 0.168036 (0.001523) | 0.661313 |
| admission_core | knn | nars_gated | 0.610327 (0.001801) | 0.168035 (0.001524) | 0.661306 |

| Scenario | Imputer | NARS-minus-MC Brier | Family lower | Family upper |
|---|---|---:|---:|---:|
| demographics | median | -3.799e-07 | -8.715e-07 | -7.319e-08 |
| demographics | knn | -1.620e-07 | -3.496e-07 | -2.228e-08 |
| admission_core | median | -1.356e-06 | -2.338e-06 | -5.816e-07 |
| admission_core | knn | -7.437e-07 | -1.400e-06 | -3.257e-07 |

All four family intervals favor small Brier reductions, with point estimates from -1.36e-06 to -1.62e-07. These conditional numerical effects are far below the 0.0001 benefit margin and do not establish clinical benefit or hospital-mortality compatibility.
The 0.0001 symbolic-benefit criterion remains unestablished by the original studies; these additional inspected outcomes do not replace those negative findings. Undefined precision values remain null when no case is predicted positive, rather than being replaced by zero.

## Verification and limitations

All twenty saved prediction/event bundles pass independent replay and uniform-gate identity. Discrimination, calibration and threshold metrics were separately recomputed from the saved probabilities. Preprocessing cache identities cover the fitted processor, full mapped data and implementation; cached contents are checked on reuse. A duplicate-row optimization failed a real-source equivalence check and was removed; its affected outputs remain in an ignored rejection folder and are excluded. Whole-frame caching preserves the original KNN arithmetic and reduces repeated computation across identical preprocessors.

Paired Brier intervals resample admissions and fitted seeds, with Bonferroni adjustment across the four fixed conditions and a fixed inference draw. Per-fit AUROC contrasts have conditional admission intervals. The export has no hospital identifiers or canonical repeat-person linkage, so these intervals do not estimate institution-level dependence, repeat-person dependence or refitting variability beyond the five fitted seeds. Neither source independence nor the first-day hospital-death endpoint equivalence is established. Plan experiment 7 remains unsatisfied.

Fixed age/sex strata and decision curves at the existing NGTA research thresholds are also exported by `scripts/nepal_diagnostics.py`. The 400 subgroup metric rows remain private; the public [diagnostic summary](../results/research_checks/nepal_diagnostics.json) includes seed-mean whole-cohort decision curves. These are additional descriptive diagnostics, not clinical threshold validation or a separately registered subgroup analysis.

## Reproduce

```powershell
python scripts/prepare_nepal.py
python scripts/evaluate_nepal.py
python scripts/summarize_nepal.py
python scripts/nepal_diagnostics.py
```

Place the publisher workbook under `data/private/open_candidates/nepal_registry/`. Original fitted WiDS artifacts are required. Prepared cohorts, preprocessing caches, pass arrays, event logs and rejected outputs stay under ignored private result directories; only aggregate evidence is committed. The pinned protocol is [nepal-analysis.json](nepal-analysis.json), and the aggregate evidence is [nepal_sensitivity.json](../results/research_checks/nepal_sensitivity.json).

Sources: [Nepal release](https://doi.org/10.6084/m9.figshare.22263847.v1), [study methods](https://wellcomeopenresearch.org/articles/8-180), [CCAA core dictionary](https://doi.org/10.6084/m9.figshare.15167406.v3).
