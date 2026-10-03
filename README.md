# NGTA v2

NGTA is a research interface for uncertainty-conditioned symbolic intervention in a tabular transformer. It uses prototype rules to revise a heuristic confidence signal and change inference. Clinical benefit, independent evidential semantics, robustness, and reviewer usefulness remain unproven.

This branch implements the validity and inference corrections in `plan.md`. `list.md` tracks completion against evidence. A working experiment implementation does not establish its research hypothesis.

## Inference

The model caches every dropout pass's probabilities, attention, token scores, CLS logit, and RNG state. Ungated, uniform, MC-confidence, and NARS gates score those identical passes and average probabilities after the sigmoid. A uniform normalized readout gate must reproduce ungated predictions within 1e-7. Deterministic and mean-logit inference are separately named controls.

The default gate changes the final attention-weighted token readout. `--encoder-intervention` also biases feature keys inside each encoder attention layer and recomputes contextual representations while replaying the same dropout RNG states. These alternatives are evaluated separately.

Neural feature frequency is attention, whereas a symbolic frequency describes a clinical proposition. Their semantics are not established as equivalent, and both pathways use the same clinical measurements. Revised frequency is logged but does not drive the confidence gate. The confidence-boost and frequency-sensitivity controls expose this limitation directly. The interface is not a complete NARS implementation.

## Data validity

TCGA selects one most-complete primary record per case within each source table. It exports conflicting repeated fields instead of silently combining different records column by column. Cross-table chronology is unverified. The prediction landmark is retrospective post-pathology association, so pathological variables do not support a preoperative prediction claim.

Sparse clinical-column filtering, constant filtering, and mutation-panel selection use training data only. Numeric KNN distances use training-standardized features. A recorded functional variant establishes a positive gene call; all other gene calls remain unknown unless explicitly verified by an assay manifest. The transformer receives mutation values and separate missing indicators.

An optional `data/assay_manifest.csv` has columns `case_submitter_id,gene,source,verified`. Each case/gene pair is unique; `verified` is `true` or `false`, and verified rows require source provenance documenting callable coverage for that gene. Only those rows permit an unrecorded variant to become a confirmed negative. MAF presence and file size do not prove negative coverage.

The local TCGA audit finds variants in two training cases and none in validation or test, with no complete verified mutation panel in any partition. This cohort cannot establish held-out genomic fusion or genomic rule intervention.

WiDS uses end-of-first-day measurements for hospital mortality. Patient-disjoint splitting is the default; `--split-mode hospital` produces hospital-disjoint partitions and records patient/hospital/ICU overlap. Group IDs must be present, and each partition must contain both outcomes. Row splitting is available explicitly for legacy comparisons.

APACHE probabilities outside [0,1] are treated as unavailable. The local file contains 2,371 such values; the precise source sentinel semantics remain unverified. After cleaning those scores, the selected features have 10.36% missing cells overall. Feature-level missingness is exported, so this measurement does not justify a broad extreme-missingness claim.

WiDS rules use observed raw values by default. KNN estimates are retained separately, and the audit counts imputed-only triggers that were suppressed. Raw masking experiments compare observed-only gating against an explicitly named imputed-rule alternative.

## Run

Use the existing environment or install `requirements.txt` into a Python environment.

```powershell
.\venv\Scripts\python.exe main.py --run-all --audit-data --split-mode hospital --output-dir results/v2_audit
.\venv\Scripts\python.exe main.py --dataset tcga --skip-paper-figures --output-dir results/v2
.\venv\Scripts\python.exe main.py --dataset wids --split-mode hospital --baseline-set standard --ablation-set submission --skip-paper-figures --output-dir results/v2
```

`--audit-data` computes source hashes, coverage, conflicts, missingness, and exact split lineage without neural training or full WiDS KNN fitting. The full WiDS KNN computation can be expensive because it compares training-standardized rows against the training cohort.

Additional experiment options:

- `--without-apache` excludes the score from feature-based predictors; raw APACHE, APACHE-only logistic regression, and logit recalibration remain separate comparators fitted on training outcomes.
- `--seeds 0 1 2 3 4` trains separate models and exports actual seed variability. One run cannot estimate a standard deviation across seeds. `--split-seed` defaults to 0 and holds cohort partitions and masking draws fixed across those training seeds.
- `--ensemble-size 5` trains and preserves five seeded ensemble checkpoints and member probabilities.
- `--shift-eval` masks originally observed raw values at 0%, 10%, 30%, 50%, and 70%, then reruns frozen preprocessing, rule extraction, and matched MC inference. Random and prespecified feature-dependent masks are saved.
- `--encoder-intervention` compares readout gating with intervention inside the encoder.
- `--evaluation-lock PATH` checks configuration, sources, split IDs, and rules against an existing `evaluation_spec.json` before training. Paths may include `{seed}` and `{dataset}` for per-run locks.
- `--export-case-traces` adds sampled case summaries. Complete intervention events are always exported.

Symbolic controls remove all or individual rules, shuffle cases, randomize truth values, sweep frequency and confidence, apply fixed priors, and compare a closed-form confidence boost. One hundred prevalence-preserving predicate permutations support empirical Brier/log-loss tests with Holm adjustment. These diagnostics do not prove clinical rule value without the locked confirmation study.

## Evaluation and artifacts

Each dataset saves model checkpoints, fitted preprocessing, classical estimators, source hashes, exact split IDs, rule definitions, evaluation specifications, and every MC pass. Metrics include AUROC, PR-AUC, Brier, log loss, fixed and quantile ECE, calibration slope/intercept, and prespecified threshold metrics. Calibration diagnostics fitted on test outcomes are reported only; they never recalibrate test predictions.

WiDS paired comparisons resample whole hospitals, using identical resamples for all methods. Observed effect sizes are separate from bootstrap means. Bonferroni family intervals cover all comparator/metric pairs, including APACHE and stronger baselines. Demographic strata and decision thresholds are fixed in code. A close point estimate does not establish equivalence or noninferiority.

Every rule trigger, including an unmapped trigger, is written to `traces/intervention_events.csv` with its rule identifier/version, source, raw value, observed status, truth values, attention effect, and counterfactual rule-off probability. `raw_test.csv`, `replay_spec.json`, and `inference_cache.npz` preserve the inputs needed for replay. File hashes and `replay_validation.json` record export integrity.

```powershell
.\venv\Scripts\python.exe -m src.trace_replay results/v2/tcga/traces
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.test-tmp/check
```

Replay separately reconstructs predicates, revisions, gates, probabilities, event completeness, and counterfactual effects from persisted files. It does not call production rule extractors or revision/gate operators. This establishes numerical replay, not clinical correctness or human oversight benefit.

## Evidence status

`results/v2_validation` contains audits of the actual local sources. `results/v2_smoke` contains short integration runs and must not be used as clinical performance evidence. The earlier `results/tcga`, `results/wids`, and `results/submission` bundles are legacy v1 artifacts. Their baseline/gate comparisons used different aggregation, their trace audits reused production arithmetic, and WiDS rules could fire on imputed measurements. They remain archived for inspection; their numerical differences cannot establish a gating or symbolic benefit.

New figure exports go to the selected result directory and reject mixed v1/v2 bundles. The manuscript labels its old tables and figures as legacy evidence. Its PDF has not yet been rebuilt: MiKTeX reports incomplete setup, and the bundled Tectonic executable is inaccessible on this host. The edited TeX source is the current manuscript. New clinical conclusions require locked hospital confirmation across five training seeds, verified genomic measurements, documented APACHE provenance, expert rule review, independent cohorts, and a blinded reviewer study. Those tasks remain open in `list.md`.
