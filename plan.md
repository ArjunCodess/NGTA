# TCGA-THCA Migration for NGTA Lymph Node Metastasis

## Summary
Refactor the pipeline from a single `data.csv` MTC task to a case-level TCGA-THCA lymph node metastasis task built from the five TSV files under `data/`. The implementation should produce one row per `cases.submitter_id`, derive the binary target from `diagnoses.ajcc_pathologic_n`, run a leakage-safe stratified 70/15/15 split, preprocess features into numeric tensors for the existing NGTA model, then rerun the pipeline and refresh README plus a focused subset of paper sections/results.

Add a strong tree-based tabular baseline on the exact same split so the final paper can defend the Transformer choice, and strengthen the manuscript framing around calibration, reliability, and black-box mitigation rather than pure AUC.

Upgrade the benchmark again to a multi-modal pipeline by fusing the TCGA-THCA clinical case-level table with a MAF-derived binary mutation matrix, while preserving the existing clinical imputation, scaling, and split logic.

## Implementation Changes
- Replace root `data.csv` usage entirely.
  - Delete `data.csv`.
  - Remove all `data.csv`, `149`-sample, `10 studies`, `study-aware split`, `MTC`, and hereditary MEN2 dataset references from `README.md`, `main.py`, and `src/pipeline.py`.
- Update the entrypoint/config interface.
  - Change `PipelineConfig.data_path` to `data_dir` with default `data`.
  - Change CLI flag from `--data-path` to `--data-dir`.
  - Update CLI/help text and printed summaries to describe TCGA-THCA lymph node metastasis prediction.
- Rebuild `src/data_loader.py` around TCGA tables.
  - Read `clinical.tsv`, `exposure.tsv`, `family_history.tsv`, `follow_up.tsv`, and `pathology_detail.tsv` with `sep="\t"` and string dtype.
  - Normalize the join key by renaming `cases.submitter_id` to `case_submitter_id` in each table.
  - Add a metadata-row guard: inspect row 0 and drop it only if it matches a descriptor/CDE pattern rather than real data. For the current repo snapshot, row 0 appears to be real data, so the implemented heuristic should leave all rows intact.
  - Normalize missing placeholders globally before further processing: `--`, `'--`, `Not Reported`, `not reported`, `Unknown`, `unknown` -> `np.nan`.
  - Build a one-row-per-case clinical base table:
    - Filter `clinical.tsv` to primary-diagnosis rows using `diagnoses.classification_of_tumor == "primary"` (this removes prior-primary / recurrence rows and resolves observed target conflicts).
    - Collapse remaining duplicate case rows by case using first non-null per column for diagnosis-level fields.
    - For repeated yes/no treatment flags in clinical, derive `derived.any_treatment_or_therapy` as `yes` if any row says yes, else `no` if any row says no, else null.
  - Collapse the other tables to one row per `case_submitter_id` with deterministic first-non-null per column, then left-join them onto the clinical base in this order: exposure, family_history, follow_up, pathology_detail.
  - After merging, drop columns with more than 70% missingness.
- Add a genomic mutation-processing branch without deleting the clinical preprocessing path.
  - Inspect `src/data_loader.py` and `src/pipeline.py` before editing and insert the new genomic path around the existing code instead of replacing the clinical imputation/scaling logic.
  - Detect the MAF file from `data/` using a pattern such as `*.maf` or `*tcga_mutations.tsv`.
  - Read the mutation file with `pd.read_csv(..., sep="\t", comment="#", low_memory=False)` so comment-prefixed metadata rows are skipped safely.
  - Derive `case_submitter_id` from `Tumor_Sample_Barcode` by extracting the first 12 characters.
  - Keep only functionally relevant mutations by filtering `Variant_Classification` to:
    - `Missense_Mutation`
    - `Nonsense_Mutation`
    - `Frame_Shift_Del`
    - `Frame_Shift_Ins`
    - `Splice_Site`
    - `In_Frame_Del`
    - `In_Frame_Ins`
  - Exclude synonymous and non-coding noise such as `Silent` and `Intron`.
  - Identify the top 50 most frequently mutated genes by `Hugo_Symbol`.
  - Pivot the filtered mutation table to one row per `case_submitter_id`, one binary column per selected gene, and aggregate duplicate case/gene hits with `groupby().max()`.
  - Prefix genomic columns so they remain distinguishable from clinical columns in the saved metadata and traces.
- Merge the modalities at the case level.
  - Left-join the genomic binary matrix onto the clinical case-level table with the clinical frame on the left.
  - Fill missing gene values with `0` after the merge so cases without sequencing coverage or without selected mutations remain in the cohort.
- Define the target before splitting.
  - Use `diagnoses.ajcc_pathologic_n`.
  - Map `N0 -> 0`, `N1/N1a/N1b -> 1`.
  - Drop any case with `NX`, null, or unmapped target before splitting.
  - On the current data snapshot, this should leave 457 labeled cases.
- Lock a leakage-safe feature set for v1.
  - Numerical features:
    - `diagnoses.age_at_diagnosis`
    - `diagnoses.year_of_diagnosis`
    - `pathology_details.tumor_length_measurement`
    - `pathology_details.tumor_width_measurement`
    - `pathology_details.tumor_depth_measurement`
  - Categorical features:
    - `demographic.gender`
    - `demographic.race`
    - `demographic.ethnicity`
    - `diagnoses.ajcc_pathologic_t`
    - `diagnoses.prior_malignancy`
    - `diagnoses.synchronous_malignancy`
    - `diagnoses.prior_treatment`
    - `diagnoses.primary_diagnosis`
    - `diagnoses.morphology`
    - `diagnoses.laterality`
    - `diagnoses.tumor_focality`
    - `diagnoses.residual_disease`
    - `pathology_details.extrathyroid_extension`
  - Explicitly exclude leakage-prone variables from the model matrix:
    - `diagnoses.ajcc_pathologic_n`
    - `diagnoses.ajcc_pathologic_stage`
    - any lymph-node count / involvement columns
    - recurrence / disease-response / follow-up outcome fields
- Replace the old split/preprocessing path.
  - Use a two-stage `StratifiedShuffleSplit` with the configured seed:
    - stage 1: 70% train / 30% holdout
    - stage 2: split holdout 50/50 into val/test
  - With the current filtered cohort this should yield 319 train, 69 val, 69 test.
  - Fit imputers/encoders/scalers on train only, then transform val/test:
    - `KNNImputer` for numeric columns
    - `SimpleImputer(strategy="most_frequent")` for categorical columns
    - `StandardScaler` for numeric columns
    - `OneHotEncoder(handle_unknown="ignore")` for categorical columns
  - Convert the final processed matrices and labels to `torch.float32` tensors for loaders.
  - Update `DataBundle` / preprocessing metadata so downstream code can access final numeric feature dimension and transformed feature names.
  - Preserve the existing clinical preprocessing behavior:
    - scale only the continuous clinical numeric columns
    - continue categorical imputation and one-hot encoding for clinical categorical variables
    - pass genomic binary columns through without `StandardScaler`
  - Concatenate scaled clinical numeric features, binary genomic features, and encoded categorical clinical features into a single multi-modal tensor.
- Adjust `src/pipeline.py` to the new tabular shape and task wording.
  - Remove all study-based split assumptions and `study_id` logging.
  - Compute model input dimension from the transformed feature matrix; the neural encoder should receive the post-encoding feature width instead of assuming the old mixed categorical-token schema.
  - If the current transformer class is kept, adapt its input path to accept the new fully numeric matrix. If that is more invasive than warranted, replace the mixed categorical/numeric tokenization with a numeric-only tabular encoder wrapper and keep the NGTA uncertainty/attention logic unchanged.
  - Update artifact labels, print statements, summaries, and chart titles to use “Lymph Node Metastasis” as the positive class.
  - Regenerate split summary and preprocessing metadata to describe TCGA case counts and selected features instead of study IDs.
  - Print the final multi-modal dataset shape before training begins, including the clinical/genomic feature breakdown.
- Add at least one classical tabular baseline on the identical split and label definition.
  - Preferred baseline order: `XGBoost`, `LightGBM`, then `RandomForestClassifier` if gradient-boosting dependencies are unavailable.
  - Train the baseline on the exact same train/val/test partition and post-imputation numeric design matrix used by the Transformer.
  - Save baseline AUC, Brier score, ECE, and accuracy into the same metrics artifacts and include it in the manuscript results table.
  - If the tree baseline wins on AUC, keep that result and use it to justify NGTA on uncertainty semantics, calibration behavior, and inspectability rather than ranking alone.
- Update docs with a focused refresh.
  - `README.md`: replace the old task/dataset/results narrative with TCGA-THCA lymph node metastasis wording, new cohort counts, new split description, new feature list, and rerun-derived metrics/artifact paths.
  - `paper/main.tex`: update abstract, task framing, dataset/application description, empirical results text/tables/captions, and conclusion passages to describe TCGA-THCA lymph node metastasis instead of the old MTC cohort.
  - Remove or rewrite the MTC-specific handcrafted rule examples/table so the paper does not claim a disease-specific symbolic rule base that the current THCA codepath does not implement; replace with a brief statement that the present benchmark evaluates the uncertainty-to-NARS attention interface on structured TCGA clinicopathologic variables.
  - Explicitly frame Brier score and ECE as deployment-relevant metrics, and avoid an AUC-first narrative if the results do not support it.
  - Strengthen the introduction, discussion, and conclusion so they state that NGTA addresses the black-box problem by quantifying and propagating uncertainty instead of only emitting a risk score.
  - Do not describe the current v1 benchmark as “strictly non-invasive” or “baseline-only”; the retained predictors are clinicopathologic and include pathologic/post-surgical variables.

## Test Plan
- Loader validation:
  - confirm all five TSVs load from `data/`
  - confirm the join key is normalized to `case_submitter_id`
  - confirm no row-0 drop occurs on the current files
  - confirm the merged frame is one row per case
  - confirm the MAF loader skips `#` metadata lines correctly
  - confirm `Tumor_Sample_Barcode -> case_submitter_id` extraction uses the first 12 characters
  - confirm only the allowed functional mutation classes are retained
  - confirm the genomic matrix is limited to the top 50 genes and is binary after case-level deduplication
  - confirm genomic columns are left-joined and missing values are filled with `0`
- Target validation:
  - confirm only `N0`, `N1`, `N1a`, `N1b` survive into the labeled cohort
  - confirm `NX` and null targets are removed before splitting
  - confirm current labeled count is 457 with near-balanced classes
- Preprocessing validation:
  - confirm columns over 70% missing are removed
  - confirm imputers/scalers/encoders are fit on train only
  - confirm transformed train/val/test matrices contain no NaNs and are strictly numeric
  - confirm genomic binary columns bypass the clinical `StandardScaler`
- Split/model validation:
  - confirm stratified split sizes are 319/69/69 with seed 0 on the current data snapshot
  - confirm model input width matches the transformed feature matrix width at runtime
  - run the full pipeline end-to-end and regenerate metrics CSVs, trace CSV, ROC, calibration, decision-curve, and training-history artifacts
- Baseline validation:
  - confirm the tree baseline uses the exact same split and target filtering as NGTA
  - confirm the baseline appears in the same saved metrics table with AUC, Brier, ECE, and accuracy
  - confirm the manuscript text follows the actual metric winners instead of forcing a Transformer-wins-on-AUC story
- Documentation validation:
  - confirm README and focused paper sections no longer mention the 149-row MTC cohort, 10 studies, or study-aware splitting
  - confirm reported metrics/counts in docs match the rerun outputs

## Assumptions And Defaults
- The current TCGA files use real data in row 0; the metadata-row logic should be heuristic and conservative, not unconditional.
- The actual missing placeholder present in this repo includes `'--`; handle that in addition to the user-listed tokens.
- `diagnoses.classification_of_tumor == "primary"` is the canonical way to resolve multi-diagnosis clinical rows before case-level collapsing.
- The implementation should prioritize leakage-safe predictors, even if some user examples such as `ajcc_pathologic_stage` are excluded.
- The current selected predictors are not a strictly non-invasive or prospective baseline feature set; they are clinicopathologic retrospective benchmark features.
- Paper scope is a focused refresh, not a full theoretical rewrite; generic NARS theory sections can stay, but MTC-specific application/results content must be updated or removed.

## Phase: Symbolic Knowledge Graph Injection

### Summary
- Append symbolic clinical-rule injection as an inference-time extension of the existing MC-dropout NARS pipeline.
- Preserve the current variance-to-NARS mapping, preprocessing, split logic, flat-confidence control, and evaluation metrics.
- Apply symbolic revision to per-feature attention confidence only; keep case-level neural probability truth values unchanged in this phase.

### Tasks
- Add `src/knowledge_base.py` with hardcoded thyroid oncology rules for:
  - `genomic_mutation__BRAF == 1 -> (0.85, 0.75)`
  - `diagnoses.age_at_diagnosis / 365.25 >= 55 -> (0.70, 0.60)`
  - `diagnoses.ajcc_pathologic_t` beginning with `T3` or `T4` -> `(0.90, 0.85)`
  - non-null `pathology_details.extrathyroid_extension` -> `(0.85, 0.80)`
- Expose a helper that converts raw test-case rows plus transformed feature names into symbolic frequency/confidence matrices, a trigger mask, per-rule trigger counts, and per-patient trigger counts.
- Extend `src/nars_interface.py` with tensor-safe `revise_truth_values(f1, c1, f2, c2)` using NARS revision with confidence clamped to `[0.001, 0.999]`.
- Integrate symbolic revision into the attention path without deleting the existing neural attention-to-NARS logic:
  - derive per-feature neural truths from MC-dropout attention summaries
  - revise only triggered features
  - fall back to neural truth values when no rule is triggered
  - gate attention with revised confidence via `A_gated = A ⊙ diag(c_rev^gamma)`
- Update evaluation outputs to log symbolic rule activity while preserving Brier, ECE, and AUC generation:
  - add symbolic trigger summaries to `results/metrics/run_summary.json`
  - add per-case symbolic counts and revised attention confidence reporting to `results/traces/test_predictions.csv`
  - use revised feature confidence for the main NARS-gated path and gamma ablation

### Validation
- Verify `revise_truth_values()` matches the paper equations and remains finite for tensor confidences at `0.0` and `1.0`.
- Confirm raw-rule hit counts on the current `seed=0` test split remain:
  - `69` test cases
  - `0` BRAF hits
  - `25` age >= 55-year hits
  - `29` `T3/T4*` hits
  - `25` non-null extrathyroid-extension hits
- Confirm full pipeline outputs still include metrics, calibration, gamma ablation, decision curve, run summary, and trace CSV artifacts.
