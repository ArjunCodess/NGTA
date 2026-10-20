# Remaining work after the source search

The owner has removed external clinical expert review, the human participant study and further WiDS source clarification from active work. Five checklist entries are excluded without being counted as satisfied. The original criteria remain 52 satisfied out of 72; active scope has 52 satisfied and 15 unresolved entries out of 67. Existing WiDS data are sufficient for the current development work. No stopped download will be resumed.

## 1. Inspect the public TCGA pathology reports

The GDC API lists 515 open THCA pathology PDFs, totaling 79,840,335 bytes. The [search inventory](../results/research_checks/source_followup.json) pins their UUIDs, publisher sizes/checksums and case associations. Their contents have not yet been inspected in this search. This is an additional public source beyond the completed biotab/XML audits, so chronology should not be declared exhausted before inspecting it.

The next independent task is to check disk space, acquire the PDFs into the ignored cache, extract their text with OCR where needed, and link only explicitly supported episode/report fields to the selected clinical records. Missing, redacted or administrative dates must stay missing. This may narrow the chronology gap; it cannot be assumed to establish availability of every model input.

The [GDC pipeline documentation](https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/DNA_Seq_Variant_Calling_Pipeline/) explicitly explains that public MAF masking can remove true somatic variants. No searched source establishes a complete case/gene callable panel for the frozen THCA inputs. Keep unavailable genes unknown unless a sourced assay/coverage record establishes otherwise. The source request is already sent; no sequencing download is justified merely to make a checklist green.

## 2. WiDS development uses the existing data

Further source clarification is excluded as an execution prerequisite at the owner's instruction. The frozen cleaning and rule conventions are retained, together with the existing with/without-APACHE and creatinine-rule-off analyses. Exact score timing and the dictionary/unit discrepancy remain stated limitations; no new interpretation is asserted because data stay out of Git.

The additional available-data experiment evaluates encoder-level MC and NARS confidence gates across all 20 frozen WiDS checkpoints, matching the saved 50-pass dropout draw and retaining uniform-encoder identity controls. It is an exploratory extension on inspected cases, with no refitting or new data. Its results are separate from the original readout experiment and cannot replace its negative finding.

## 3. Decide whether independent-cohort access is feasible

The practical website-only choices and access steps are in [internet-data.md](internet-data.md). Zigong is the first candidate to assess because the currently listed policy does not require contributor study approval beyond normal credentialing and the DUA. No independent eligibility is claimed. An external dataset is separate from recruiting an external researcher. [SICdb](https://physionet.org/content/sicdb/1.0.8/) is a possible Austrian source with hospital-mortality information. Its publisher requires PhysioNet credentialing, CITI Data or Specimens Only Research training, a signed DUA and contributor approval of the proposed study. These steps do not establish eligibility for NGTA automatically.

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
