# Reproducing NGTA

This is the technical release for the inspected development studies in `plan.md`. External clinical expert review and participant collection are excluded by the owner; further WiDS source clarification is excluded as an execution prerequisite, while eligible independent-cohort validation remains unresolved. Failed symbolic-benefit and robustness criteria remain failed; the release does not change them into completed scientific claims.

## What is included

`study_recipe.json` records all 36 fitted configurations: 20 WiDS fits across imputer/APACHE conditions, 15 TCGA modality fits, and one shuffled-neural-label control. It pins the original source hashes and evaluation-spec hashes. Training seeds, split seed, model dimensions, epochs, early stopping, dropout passes and inference repeats are explicit for every fit.

The source package includes implementation, tests, frozen recipe, aggregate paper assets, selected integrity reports and these instructions. It deliberately excludes source datasets, fitted objects, patient-level traces, private reviewer materials, credentials and Git history. An integrity manifest hashes every packaged file. Source package integrity is separate from clinical validation and from the availability of a fitted research archive.

## Verify this checkout

From the repository root, use the installed environment:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts/verify_local_artifacts.py
.\venv\Scripts\python.exe scripts/verify_studies.py --checkpoint-replay
```

The last command requires all 36 complete local study bundles and their original authorized source files. It restores fitted preprocessors, checkpoints and recorded dropout RNG, compares all 50 saved probability passes, and independently replays persisted events. Missing study folders fail verification. The archive checker separately checks all 297 removed generated files by size and SHA256. Those files remain local and untracked.

## Inspect the frozen training recipe

```powershell
python scripts/reproduce_studies.py
```

This is a dry run: it prints 36 separate destinations and does not train or create result directories. The outputs default to `results/private/reproduction/`, using the same hospital/TCGA subdirectory structure as the original studies. It rejects output roots that would overwrite or nest inside the original studies.

For a new environment, Python 3.14 and `requirements-lock.windows-py314.txt` describe the recorded Windows environment. The recorded Torch version is a CUDA build; installation requires its matching wheel index. On a compatible system the installation command is:

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-lock.windows-py314.txt
```

Other platforms may need a platform-specific Torch build. `requirements.txt` is a portability starting point, not an exact environment lock. Record deviations; cross-platform refitting need not reproduce every floating-point value.

## Regenerate the fitted studies when needed

Acquire the original authorized sources at the recipe's `data/` and `data/acquired_tcga/` locations. Follow [source acquisition instructions](get-needed-data.md); do not substitute the earlier two-file sparse TCGA source. The executor verifies source hashes before any pipeline writes. The WiDS dictionary remains a separate private provenance document and is not used to silently reinterpret the frozen input units.

```powershell
python scripts/reproduce_studies.py --execute
```

To select one condition, append `--conditions hospital_study/knn`. To resume compatible outputs, append `--resume`. Source/configuration checks still apply. The saved 36 fit directories occupy about 10.14 GB before additional analyses; the executor checks estimated remaining space plus a 20% reserve, including on resume. This checkout does not currently have room for another complete copy, so the release reuses the verified fitted artifacts rather than duplicating training. No files need to be deleted to use the current studies.

After a full regeneration, verify its separate root:

```powershell
python scripts/verify_studies.py --results-root results/private/reproduction --checkpoint-replay
```

## Rebuild the analyses

Fit-level evaluation already exports matched baselines, truth-value controls, 100 predicate permutations, individual/all-rule removal, complete events and calibration/subgroup/decision-curve results. To regenerate ensemble, seed-variability, hierarchical and raw-masking analyses, run the following for each WiDS condition:

```powershell
python scripts/evaluate_study.py --root results/private/reproduction/hospital_study/knn --mask-seeds 0 1 2 3 4
python scripts/uncertainty_intervals.py --root results/private/reproduction/hospital_study/knn
```

Repeat with `median`; these two primary imputers have the full masking study. For `knn_no_apache` and `median_no_apache`, omit `--mask-seeds` and the masking uncertainty command, matching the original analysis scope. For each TCGA modality, use:

```powershell
python scripts/evaluate_study.py --root results/private/reproduction/tcga_study/fused --dataset tcga
```

Repeat with `clinical` and `genomic`. Do not pool repeated seeds as independent patients. The separately labeled shuffled-neural-label fit is a negative control, not a five-member ensemble.

Paired condition comparisons use `scripts/compare_controls.py --left-root PATH --right-root PATH --output PATH`, with `--dataset tcga` for modality contrasts. Keep the same five seeds and fixed split. Frozen public cross-source sensitivity has separate preparation/evaluation scripts and remains exploratory; it does not qualify as independent clinical validation.

The unresolved creatinine-unit conflict has a frozen rule-off sensitivity:

```powershell
python scripts/creatinine_sensitivity.py --results-root results/private/reproduction --output results/private/reproduction/research_checks/creatinine_sensitivity.json
```

This disables only the creatinine revision. It does not change input units, retrain, pick a new threshold or claim that a unit has been verified. It checks agreement with the original individual-rule ablation, reports paired hospital/seed intervals and counts threshold changes. Original-run results are retained in `results/research_checks/creatinine_sensitivity.json`.

## Rebuild the paper and source package

The current paper uses current development results. Historical confounded tables and case examples are preserved in `paper/legacy_results.tex`, excluded from the compiled manuscript. NeurIPS formatting files stay inside `paper/`.

To rebuild tables from regenerated analyses, then figures from explicitly selected sources:

```powershell
python scripts/refresh_paper_results.py --results-root results/private/reproduction
python -c "from src.paper_figures import generate_paper_figures; generate_paper_figures(seeds=[0,1,2,3,4], dataset_roots={'tcga':'results/private/reproduction/tcga_study/fused','wids':'results/private/reproduction/hospital_study/knn'})"
```

Compile `paper/main.tex` using the supplied NeurIPS style and a LaTeX installation with bibliography support. With MiKTeX/TeX Live:

```powershell
Set-Location paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Set-Location ..
```

Build a fresh source package and verify its member inventory and hashes:

```powershell
python scripts/package_reproducibility.py --output .cache/NGTA-reproducibility.zip
python scripts/package_reproducibility.py --verify --output .cache/NGTA-reproducibility.zip
```

Choose a new filename if it already exists; the builder does not overwrite an existing package. Exact fitted replay additionally requires a separate permitted full-study archive with source data and trace/checkpoint files. The 297-file preservation manifest alone does not reconstruct all other files in those directories or the new private encoder probability arrays. There is no approved external fitted-archive location yet; see [artifact storage](artifact-storage.md).

No researcher recruitment, additional data download or favorable result is needed to use this technical release. Any future clinical validation should begin with its own reviewed protocol, qualified source data and untouched outcomes.

## Fresh private reviewer materials

The original public demonstration is unsuitable for blinded recruitment. Its predictable case/fault sequence is now replaced in newly generated packages with opaque case IDs and shuffled fault assignments. Use `python scripts/reviewer_study.py --private --output-dir results/reviewer_private/NEW_DIRECTORY` to prepare a new secret-seeded package. The CLI refuses a repository-public destination and does not print the seed or answer key. An existing package is never overwritten.

Only the individual `participant/reviewer_*` folder assigned to a reviewer is intended for delivery after protocol finalization. It contains that reviewer's assigned evaluation format once per case and a free-text correction response sheet. Master cross-format tasks, assignments and the investigator grading template are moved into `investigator/`. Do not distribute the root directory, protocol seed, investigator folder or other reviewers' materials. Empty response sheets cannot produce a human-study result. The prepared private package has 24 reviewer folders and 1,920 balanced assignments for 80 evaluation cases, but zero recruited participants and zero human responses.

The current [scope policy](scope-policy.json) crosses out five expert/participant or WiDS clarification criteria without marking them complete. [Next work](next-work.md) records the remaining official-source and cohort tasks. The source package includes this policy so regenerated checklists retain the owner decision.

## Frozen WiDS encoder extension

Run `python scripts/evaluate_wids_encoder.py` to evaluate the existing 20 WiDS checkpoints with uniform, MC-confidence and NARS-confidence gates inside every encoder layer. The uniform gate must reproduce all original 50 probability passes within 1e-7. This uses the saved processor, weights and primary dropout RNG, with no refitting or downloads. Probability arrays and case-level outputs go only to `results/private/wids_encoder_study/`; the aggregate report is `results/research_checks/wids_encoder.json`. Existing results resume only when their checkpoint/specification/RNG identity and saved probability hash match. A partial result is preserved and rejected instead of overwritten.

Intervals resample hospitals and fitted seeds. The primary original dropout draw is fixed; these new intervals do not include the original study's three-repeat uncertainty. Twelve secondary condition/contrast comparisons receive Bonferroni family intervals. All results remain exploratory on already inspected development cases.


## Nepal changed-endpoint sensitivity and pathology audit

The Nepal extension reuses ten frozen no-APACHE WiDS checkpoints under two explicit mappings, with five seeds per imputer and 50 dropout passes. Its endpoint is ICU death, not hospital death. Place the pinned public workbook under `data/private/open_candidates/nepal_registry/nicrf_encounters.xlsx`, then run:

```powershell
python scripts/prepare_nepal.py
python scripts/evaluate_nepal.py
python scripts/summarize_nepal.py
python scripts/nepal_diagnostics.py
```

Preparation checks the full source SHA256, records the protocol/mapping before loading outcomes, rejects conflicting daily duplicates, restricts to recorded adults and preserves unknowns. The recorded protocol was amended using the obtained CCAA dictionary and predictor-scale audit before model scoring; it is not an independently registered clinical protocol. Unit-conflicting temperature/creatinine/WBC fields stay unavailable. Cached preprocessing requires identical full mapped inputs, fitted processor and implementation identities, plus a saved-content hash. Full-transform batching is preserved because a duplicate-row shortcut changed tied KNN neighbors on real source samples; those rejected outputs stay private and excluded.

Only `results/research_checks/nepal_sensitivity.json` and the source/preprocessing summaries are public evidence. Raw cohorts, cached pass arrays, event logs and rejected outputs stay under ignored `results/private/`. Exact reruns require the separately retained WiDS fitted artifacts; the source package does not include them. No clinical compatibility acceptance flag is produced for the changed endpoint. [Nepal results](nepal-results.md) states the limitations and actual intervals.

Run `python scripts/audit_tcga_pathology.py` to acquire/verify the pinned 515 public PDFs and repeat native extraction using `pdftotext`. Publisher UUIDs, sizes and MD5 are inherited from `source_followup.json`; extracted text stays in `.cache/ngta-pathology/`. The date-candidate review records five private text hashes; rerunning extraction with another Poppler version can change text bytes and requires fresh review rather than silently preserving that review. Numeric date patterns, procurement dates and molecular keywords do not establish selected-input availability or callable negatives.
