# Remaining work after the source search

The owner has removed external clinical expert review and the human participant study from active work. Their four checklist entries are struck through, not counted as completed. The original criteria remain 52 satisfied out of 72; the active scope has 52 satisfied and 16 unresolved entries. Those entries overlap across the work below.

## 1. Inspect the public TCGA pathology reports

The GDC API lists 515 open THCA pathology PDFs, totaling 79,840,335 bytes. The [search inventory](../results/research_checks/source_followup.json) pins their UUIDs, publisher sizes/checksums and case associations. Their contents have not yet been inspected in this search. This is an additional public source beyond the completed biotab/XML audits, so chronology should not be declared exhausted before inspecting it.

The next independent task is to check disk space, acquire the PDFs into the ignored cache, extract their text with OCR where needed, and link only explicitly supported episode/report fields to the selected clinical records. Missing, redacted or administrative dates must stay missing. This may narrow the chronology gap; it cannot be assumed to establish availability of every model input.

The [GDC pipeline documentation](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/DNA_Seq_Variant_Calling_Pipeline/) explicitly explains that public MAF masking can remove true somatic variants. No searched source establishes a complete case/gene callable panel for the frozen THCA inputs. Keep unavailable genes unknown unless a sourced assay/coverage record establishes otherwise. The source request is already sent; no sequencing download is justified merely to make a checklist green.

## 2. Finish the WiDS provenance assessment

The [publisher issue](https://github.com/MIT-LCP/gossis/issues/5) recommends replacing eICU -1 values with NULL. Its [pinned cohort SQL](https://github.com/MIT-LCP/gossis/blob/95b69b99620d322b088262a41cd8098a1aed2f3e/eicu/sql/cohort.sql) also uses an APACHE IVa prediction of -1 with score greater than 1 as a readmission proxy. This supports the existing unavailable-value treatment at the source-family level. It does not identify the exact WiDS extraction revision or prove when its exported score became available.

The creatinine conflict also remains: the inspected dictionary says micromol/L while the source SQL and observed scale are consistent with a different convention. Values alone cannot certify the unit. Record the source-family findings, seek an exact release definition or publisher correction, and preserve the existing no-APACHE and creatinine-rule-off results. No further model fit is needed unless authoritative information changes the interpretation. The GOSSIS request is already sent.

## 3. Decide whether independent-cohort access is feasible

An external dataset is separate from recruiting an external researcher. [SICdb](https://physionet.org/content/sicdb/1.0.8/) is a possible Austrian source with hospital-mortality information. Its publisher requires PhysioNet credentialing, CITI Data or Specimens Only Research training, a signed DUA and contributor approval of the proposed study. These steps do not establish eligibility for NGTA automatically.

If the owner pursues access, the practical order is to complete the [PhysioNet training/credentialing process](https://physionet.org/about/citi-course/), then submit the prepared research question through SICdb's access page. Before inspecting outcomes, freeze the no-APACHE checkpoint choice, first-day measurement mapping, units, hospital-death definition, source-independence checks and analysis. I can prepare and implement that technical work. In particular, SICdb's OffsetOfDeath includes deaths outside hospital and cannot simply be used as the WiDS hospital-death label. Personal training, identity verification and agreement acceptance belong to the owner.

[HiRID](https://physionet.org/content/hirid/1.1.1/) is another gated source; discharge-status meaning and unlinked repeat-admission identities need review before it could meet this plan. Neither dataset was acquired or declared compatible. The existing 12,000-case public sensitivity remains an exploratory result and its outcomes are already inspected.

If access is unavailable too, external confirmation must remain an unavailable result. The project can report its current internal and exploratory evidence without an independent-validation claim.

## 4. Keep failed experiments as completed negative research

The symbolic-benefit, strongest-comparator benefit and general missingness-robustness goals were not established by the executed experiments. There is no remaining command that turns those results into a positive finding. The current paper can retain these negative outcomes as its final findings.

If a stronger method is pursued, it needs a new development hypothesis, training/validation-only model or rule changes, and an untouched evaluation. That would be another research iteration, not unfinished execution of the frozen experiments. Do not tune on the inspected outcomes to satisfy the original margins.

## Removed from active work

- ~~Recruit clinical specialists to validate prototype rules and truth weights.~~
- ~~Recruit participants and run the blinded reviewer-benefit study.~~

Existing private study materials and review forms remain preserved. No clinical expert approval, human benefit or participant evidence is claimed, and no further researcher outreach is requested under this scope decision.
