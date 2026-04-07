# Add Parallel WiDS ICU Pipeline Without Disturbing TCGA

## Summary
- Preserve the existing TCGA path as-is and add a second dataset path selected by `--dataset`, defaulting to `tcga` and accepting `wids`.
- Implement a WiDS-specific loader and symbolic rule base for ICU hospital mortality using the requested 15 features from `data/wids_icu.csv`.
- Keep symbolic revision behavior aligned with the current TCGA design: WiDS rule masks are generated in preprocessing, then used in the same post-MC inference attention-revision stage already used by the TCGA pipeline.
- Make `python main.py --run-all` the full-repo orchestration entrypoint: it must run TCGA and WiDS sequentially, calculate every artifact for both datasets, and write all results to disk.
- Isolate outputs by dataset so TCGA and WiDS artifacts never overwrite each other.
- For `seed=0`, the current WiDS file should split to `64199 / 13757 / 13757` rows for train/val/test after target filtering.
- After the pipeline work is complete, update `paper/main.tex` and `README.md` so the documentation matches the dual-dataset implementation and the new `--run-all` behavior.

## Key Changes
- `main.py`
  - Add `--dataset {tcga,wids}` with default `tcga`.
  - Pass `dataset` into `PipelineConfig`.
  - Keep all existing TCGA flags unchanged.
  - Make `--run-all` override single-dataset execution and run the full TCGA pipeline followed by the full WiDS pipeline.
  - Ensure `--run-all` writes both dataset result bundles plus an aggregate top-level run summary.

- `src/wids_loader.py`
  - Add a dedicated WiDS loader entrypoint, e.g. `load_wids_data_bundle(data_dir, batch_size, seed)`.
  - Read `data/wids_icu.csv` with `pd.read_csv(..., na_values=['NA'])`.
  - Use `encounter_id` as the WiDS identifier and `hospital_death` as the binary target; drop rows with missing target, then cast target to integer.
  - Use exactly these feature groups:
    - Continuous numeric, scaled: `age`, `bmi`, `d1_heartrate_max`, `d1_sysbp_min`, `d1_temp_max`, `d1_lactate_max`, `d1_bun_max`, `d1_creatinine_max`, `d1_glucose_max`, `d1_wbc_max`, `d1_spo2_min`, `d1_platelets_min`, `apache_4a_hospital_death_prob`
    - Binary pass-through, unscaled: `elective_surgery`
    - Categorical, imputed + one-hot encoded: `gender`
  - Perform a stratified `70/15/15` split with the provided seed.
  - Fit preprocessing on train only:
    - `KNNImputer(n_neighbors=5)` for the 13 continuous columns
    - `SimpleImputer(strategy='most_frequent')` for `gender`
    - `OneHotEncoder(handle_unknown='ignore')` for `gender`
    - `StandardScaler` on the 13 continuous columns only, after numeric imputation
  - Evaluate ICU rules immediately after numeric KNN imputation and before scaling, using the imputed-but-human-readable numeric values:
    - `rule_lactate`: `d1_lactate_max >= 4.0`
    - `rule_hypotension`: `d1_sysbp_min <= 90.0`
    - `rule_age`: `age >= 75.0`
    - `rule_creatinine`: `d1_creatinine_max >= 2.0`
  - Return split bundles that expose, per sample:
    - scaled continuous tensor `[13]`
    - encoded auxiliary tensor containing `elective_surgery` plus one-hot gender columns
    - combined model feature tensor used by the transformer
    - rule trigger tensor `[4]` in fixed order: lactate, hypotension, age, creatinine
    - target label tensor
  - Store feature names and rule names in the bundle so downstream code can align attention positions with symbolic rules deterministically.

- `src/wids_knowledge_base.py`
  - Define the WiDS rule map:
    - `rule_lactate -> (0.85, 0.80)`
    - `rule_hypotension -> (0.75, 0.70)`
    - `rule_age -> (0.65, 0.60)`
    - `rule_creatinine -> (0.70, 0.65)`
  - Add a helper parallel to the TCGA symbolic builder that converts WiDS rule-trigger tensors into `symbolic_frequency`, `symbolic_confidence`, and `symbolic_trigger_mask` arrays aligned to the transformer input dimension.
  - Apply symbolic truth values only to the matching continuous-feature positions for `d1_lactate_max`, `d1_sysbp_min`, `age`, and `d1_creatinine_max`; all other feature positions remain zero / untriggered.

- `src/pipeline.py`
  - Extend `PipelineConfig` with `dataset`.
  - Branch dataset loading:
    - `tcga` keeps using the existing loader and knowledge base unchanged.
    - `wids` uses the new WiDS loader and WiDS symbolic builder.
  - Ensure each dataset run computes the full artifact set end to end:
    - training history
    - random forest baseline
    - baseline / flat-confidence / NARS-gated metrics
    - bootstrap AUC intervals
    - calibration reliability
    - decision curve
    - gamma ablation
    - trace CSV
    - charts
    - run summary JSON
  - For WiDS, force effective batch size to `512` regardless of the CLI batch-size value.
  - Build the transformer with `input_dim` from the selected bundle, not a hardcoded feature count.
  - Keep training logic unchanged for TCGA.
  - For WiDS batches, train and evaluate the transformer on the combined feature tensor while preserving the rule-trigger tensor for the symbolic gating stage.
  - In the WiDS inference path, run the same NARS revision math already used now:
    - derive neural attention truth values from MC attention mean/variance
    - revise only the triggered feature truths with WiDS symbolic `(f_sym, c_sym)`
    - gate attention with revised confidences
  - Train the random-forest baseline on the exact same WiDS train/val/test splits and the exact same transformed design matrix used for the transformer.
  - Parameterize dataset metadata in summaries and plots:
    - WiDS positive class label: `Hospital Mortality`
    - WiDS target column: `hospital_death`
    - WiDS id column: `encounter_id`
  - Write outputs into dataset-specific directories, e.g. under a `tcga/` or `wids/` subdirectory inside the chosen output root, so artifacts co-exist safely.
  - Print WiDS test-set rule-trigger counts for all 4 ICU rules from the post-imputation, pre-scaling masks.

## Public Interfaces / Contracts
- CLI:
  - `python main.py --dataset tcga`
  - `python main.py --dataset wids`
  - `python main.py --run-all`
- Config:
  - `PipelineConfig.dataset: str`
- WiDS loader contract:
  - deterministic split by `seed`
  - train-only fitted imputers/scaler/encoder
  - fixed rule order: `rule_lactate`, `rule_hypotension`, `rule_age`, `rule_creatinine`
  - combined model feature order:
    - 13 scaled continuous features first
    - `elective_surgery` next
    - one-hot gender columns last

## Test Plan
- CLI smoke checks:
  - `python main.py --dataset tcga --epochs 1 --mc-samples 2`
  - `python main.py --dataset wids --epochs 1 --mc-samples 2`
  - `python main.py --run-all --epochs 1 --mc-samples 2`
- Run-all orchestration checks:
  - confirms `--run-all` executes both datasets even if `--dataset` is omitted
  - confirms every expected metrics/traces/chart artifact is produced for both datasets
  - confirms aggregate summary output is written after both runs finish
- WiDS loader invariants:
  - confirms `na_values=['NA']` parsing
  - confirms target rows with missing `hospital_death` are removed before splitting
  - confirms train/val/test sizes are `64199 / 13757 / 13757` for the current file with `seed=0`
  - confirms continuous tensor width is `13`
  - confirms rule tensor width is `4`
  - confirms transformer input width is derived dynamically from the transformed feature count
- Leakage checks:
  - imputers, scaler, and encoder are fit only on train
  - validation/test use only `.transform(...)`
  - random forest and transformer consume the same transformed split matrices
- Symbolic checks:
  - rule masks are computed from imputed unscaled numeric values, not scaled values
  - only the 4 intended feature positions receive symbolic truth injections
  - WiDS test logging prints per-rule counts for lactate, hypotension, age, and creatinine triggers
- Regression checks:
  - TCGA path still runs without using any WiDS-only code or changing TCGA artifacts/layout except for dataset namespacing
  - `paper/main.tex` and `README.md` are updated at the end to describe both datasets, the new result layout, and the `python main.py --run-all` workflow

## Assumptions / Defaults
- `elective_surgery` is treated as an unscaled binary pass-through feature and concatenated with the one-hot gender block.
- WiDS symbolic revision stays aligned with the current TCGA inference-time gating design rather than moving into the model’s inner training forward pass.
- WiDS uses a forced batch size of `512`.
- Output isolation is mandatory, so dataset-specific artifact directories are part of the implementation.
