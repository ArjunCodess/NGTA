# What to get next

The current implementation and internal studies are checked. The research plan is still 53/72 because the remaining claims need source evidence, new evaluation data or real people. These instructions identify the inputs; they do not promise that every missing field exists or that a new study will produce a positive result.

## 1. Thyroid data: was the gene tested, and when was the result available?

We already acquired 498 public mutation files. Another mutation list will not prove that an unlisted mutation is absent. We need a table explaining whether each selected gene was actually tested well enough in each patient, plus dates linking the clinical, pathology and genomic records to the same episode.

The exact public case IDs and selected genes are in [required-tcga-cases.csv](data-requests/required-tcga-cases.csv) and [required-tcga-genes.csv](data-requests/required-tcga-genes.csv). Send these lists with the GDC email draft in [email templates](email-templates.md).

### Where and how

1. Open the [GDC Data Portal](https://portal.gdc.cancer.gov/). Select project **TCGA-THCA** in the cohort/repository filters. Follow the [official repository guide](https://docs.gdc.cancer.gov/Data_Portal/Users_Guide/Repository/) if the interface differs.
2. In Repository, filter Data Type to **Clinical Supplement**, then **Biospecimen Supplement**. Download the available XML/biotab supplements, their file metadata and the download manifest. These are additional source records, not a replacement for the already pinned mutation files. GDC explicitly identifies these supplement types as sources of cancer-specific clinical fields. [Official supplement instructions](https://gdc.cancer.gov/content/where-can-i-find-clinical-data-elements-specific-my-cancer-research-interest).
3. Email **support@nci-gdc.datacommons.io**, the address on the [GDC Help Desk page](https://gdc.cancer.gov/support/help-desk). Ask whether the selected cases have gene-level assay coverage/quality evidence, specimen-to-case links, pathology dates and assay/report availability dates. Ask for the exact files or fields and their definitions. Ask them to state explicitly if these were never collected or cannot be released.
4. Only if the necessary evidence requires controlled sequencing files, follow [GDC controlled-access instructions](https://gdc.cancer.gov/access-data/obtaining-access-controlled-data). A research PI obtains an eRA Commons account and requests the relevant TCGA dataset in [dbGaP Authorized Access](https://dbgap.ncbi.nlm.nih.gov/aa/wga.cgi?page=login). After approval, the PI can add authorized lab members as downloaders. Log in to GDC with eRA Commons and obtain the permitted files/token. If you lack an institutional PI, seek a research collaborator first; a portal account alone does not grant controlled access.

Do not begin a large BAM download until GDC confirms what it can resolve. Sequencing files may support coverage analysis, but they do not automatically provide the date a pathology or genomic report became available.

### What to bring back

- Coverage evidence with case ID, gene, assay/source file, what was tested, relevant quality/callability criteria and whether a negative call is supported. Our minimal ingest format is `case_submitter_id,gene,source,verified`, but setting `verified=true` requires actual supporting evidence.
- Case/specimen links and dates or relative-day offsets for tissue collection, pathology, genomic testing and result availability, with definitions of each date. A diagnosis date alone is insufficient.
- The publisher's manifest, metadata/dictionary, source version, and the custodian's explanation of unavailable fields.

The currently selected pathology records have no pathology dates. The supplements may add information, but I have not established that they contain the missing availability dates. If the source never recorded them, that part of the audit must remain unresolved or the claim must stay retrospective.

## 2. WiDS: what does the APACHE field mean?

We need documentation for `apache_4a_hospital_death_prob`: the meaning of negative values, which APACHE model produced it, which time window was used, and when the probability became available. We also need the cohort's actual collection period and source-institution provenance to evaluate possible overlap with a new cohort.

### Where and how

1. Create/sign in to a PhysioNet account at the [WiDS 2020 release](https://physionet.org/content/widsdatathon2020/1.0.0/).
2. Complete the access steps shown there and sign its data-use agreement. This release requires registered access; do not assume it has the same credentialing requirements as MIMIC.
3. Download the **WiDS Datathon 2020 Dictionary**, associated release documentation and source/version information. Keep the original files unchanged.
4. If the dictionary does not answer the score/timing questions, email **gossis@mit.edu**, published on the [GOSSIS contact page](https://gossis.mit.edu/contact/). Use the WiDS/GOSSIS draft in [email templates](email-templates.md). Ask specifically whether BIDMC/MIMIC patients or eICU sources overlap this WiDS release and what evidence can establish an independent cohort.

Bring back the dictionary and written clarification. Existing [eICU APACHE result documentation](https://eicu.mit.edu/eicutables/apachepatientresult/) explains the general predictions, but it does not settle the exact exported WiDS field's sentinel or availability time.

## 3. New ICU data: test the frozen model on patients it did not learn from

We need first-24-hour measurements, a hospital-survival outcome, de-identified patient/stay IDs and source/time provenance. We do not need names, addresses or original identifying patient IDs.

### Accessible candidate: MIMIC-IV

MIMIC-IV is a candidate to assess after access and provenance review. It is one institution, so it cannot alone supply between-hospital uncertainty or broad institutional generalization. It also overlaps the broader MIMIC source family used by the public 2012 sensitivity. Obtain source overlap clarification before declaring it independent or untouched.

1. Start with the [MIMIC-IV 3.1 release](https://physionet.org/content/mimiciv/3.1/). This pins a documented version; check the publisher's version list if choosing another release, and keep one version throughout the study.
2. Follow [PhysioNet's CITI instructions](https://physionet.org/about/citi-course/). Create a CITI account, use the listed **Massachusetts Institute of Technology Affiliates** course affiliation as instructed for non-MIT users, and complete **Data or Specimens Only Research** and the required conflict-of-interest modules. This course access does not make you MIT staff. Download the full training report, not just the certificate.
3. Complete identity credentialing at [PhysioNet profile settings](https://physionet.org/settings/profile/) and submit the training report through its training workflow. Return to the MIMIC release and sign the DUA when eligible. See the [PhysioNet access FAQ](https://physionet.org/about/faqs/).
4. Download these tables from the same release: `hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `hosp/labevents.csv.gz`, `hosp/d_labitems.csv.gz`, `icu/icustays.csv.gz`, `icu/chartevents.csv.gz` and `icu/d_items.csv.gz`. Keep their documentation and original timestamps, including record-entry timestamps where present. These tables cover demographics, hospital outcome, stay boundaries, labs and bedside measurements. [MIMIC table documentation](https://mimic.mit.edu/docs/iv/modules/).
5. Ask the source owners about independence from WiDS and the earlier MIMIC-derived sensitivity. Before inspecting evaluation labels, agree and save the cohort inclusion rules, measurement mapping and final evaluation procedure. Download/access alone does not certify independence or eliminate prior inspection.

We will calculate the first-24-hour extrema and check units using the item dictionaries. Do not substitute arterial oxygen saturation for pulse-oximetry SpO2. MIMIC does not automatically supply the WiDS APACHE probability or the same elective-surgery definition; missing inputs must be documented or obtained from a justified source. The existing no-APACHE checkpoints are appropriate candidates for the locked comparison.

Restricted records and derivatives need approved local storage and source-compliant sharing. The [MIMIC publisher's sharing guidance](https://physionet.org/content/mimiciv/3.1/) treats derived data/models as sensitive and directs sharing through the same source agreement. Do not put downloaded rows, tokens or patient-level derivatives into GitHub or email attachments. Use `data/private/` and `results/private/` locally; both are ignored by Git. Keep your passwords and tokens private.

### What completes the broader hospital-generalization task

Ask a collaborating hospital/research network for an authorized independent cohort, preferably spanning multiple institutions, with the fields in [external-icu-data-request.csv](data-requests/external-icu-data-request.csv). We need a custodian statement about source overlap and collection years, plus a reviewed mapping to hospital mortality and the 24-hour prediction point. At least two real institutional clusters are necessary for the existing hospital-bootstrap procedure; the required number of hospitals/patients should be determined with statistical review. Different integer IDs do not prove different patients.

MIMIC is useful to pursue, but no public download has been certified here as a complete solution to all these conditions. A custodian may have to confirm eligibility or provide a different cohort.

## 4. People and decisions still needed

- A thyroid oncology/endocrine surgery/pathology expert and a critical-care expert need to review the eight prototype rules. Send the clinical-review invitation and [rule-review-form.csv](data-requests/rule-review-form.csv); keep dated qualifications, decisions, sources and rationale. An expert opinion alone cannot make numerical truth weights empirically calibrated.
- A statistician needs to review cohort power, hospital/case uncertainty, the reviewer-study sample size and the preregistered comparisons. The existing 24-reviewer/80-case package is a proposed demonstration design, not a completed or approved recruitment study.
- Obtain your institution's ethics/research-review determination and consent process as applicable, run a timing/usability pilot, then recruit real reviewers. Generate a fresh private study package; the published demonstration answer key cannot be used for a blinded study. [Current protocol](../results/reviewer_study/protocol.json).
- Pei Wang can review NARS/NAL methodology. That does not replace clinical rule review or participant evidence. His verified contact and draft are in [email templates](email-templates.md).
- Decide how the paper should handle the negative findings. The current studies do not establish symbolic predictive benefit. We can report that honestly; proving a revised method needs a justified new design and fresh confirmation rather than repeated tuning on inspected labels.
- Keep a separate approved archive of the removed artifacts. All 297 local files are preserved, but the working directory is not a separate backup. [Storage and verification instructions](artifact-storage.md).

## Suggested order

Start the PhysioNet/CITI application, send the GDC and GOSSIS metadata requests, and seek clinical/statistical collaborators. Those can proceed together. Bring back documentation and authorized local data, then finalize the locked cohort/rules/protocol before outcome inspection or formal reviewer recruitment. The emails below are drafts; none has been sent.
