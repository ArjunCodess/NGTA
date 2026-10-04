# Email drafts

Replace bracketed fields and attach only the suggested material. None of these drafts has been sent. Do not send patient records, access tokens or the future blinded-study answer key.

## Clinical expert: review the prototype rules

Subject: Request for clinical review of NGTA's thyroid/ICU rules

Dear Dr. [Surname],

I am [name, role and affiliation], working on NGTA, a research prototype that uses explicit rules to revise a confidence signal in a transformer readout. We evaluate thyroid lymph-node involvement and ICU hospital mortality.

Would you be willing to review the four [thyroid/critical-care] rules relevant to your specialty? I need feedback on the clinical definitions and thresholds, which observations would be available at the prediction point, whether the predicates reuse correlated evidence, and whether the proposed numerical weights have a defensible basis. The rules are currently labeled as prototypes, and the completed comparisons have not established a predictive benefit from symbolic revision.

I can send the rule table, a short methods summary and the current manuscript. The review form allows retaining, revising or rejecting each rule, with reasons and supporting sources. We can agree on the scope and expected effort before you begin.

If you are unavailable, an introduction to a suitable colleague would be helpful. Please let me know whether your institution requires a formal collaboration arrangement for this review.

Best regards,
[Name]
[Role/affiliation]
[Contact]
[Repository/manuscript link]

Suggested attachments: `rule-review-form.csv`, a short methods summary and the manuscript. Send only the specialty-specific rows if preferred. Do not attach participant answer keys.

## Participant: expression of interest before recruitment

Subject: Interest in a research study of model explanation auditability

Dear Dr. [Surname],

I am [name, role and affiliation]. We are preparing a study comparing two formats for explaining research-model outputs. Participants would review synthetic cases, identify possible issues, record their confidence and suggest corrections. This would not involve treating patients or making care decisions.

Would you be interested in receiving the finalized study information? We are currently checking interest and piloting the workload. Formal participation will begin only after the applicable protocol/ethics review and consent process, and the session length and any compensation will be stated in the invitation.

The study is intended to measure whether the explanations help reviewers; no positive result is assumed. We will keep participant identities separate from the coded research responses.

Best regards,
[Name]
[Role/affiliation]
[Contact]

Send the finalized participant information/consent materials later, not the public investigator key or cases intended for the future blinded session.

## Pei Wang: methodological feedback

To: **pei.wang@temple.edu**, verified on [his Temple University page](https://www.cis.temple.edu/~wangp/).

Subject: NGTA follow-up: NAL confidence revision and evidential independence

Dear Professor Wang,

Thank you for your earlier feedback on NGTA. I have revised the implementation and claims to distinguish our application-specific confidence initializer from NARS evidence amount. The system implements NAL-style truth-value operations and feature-level revision; it is not a complete NARS architecture.

The current interface initializes a heuristic confidence from MC-dropout attention variation, combines it with prototype rule evidence, and uses revised confidence in an inference-time readout gate. Revised frequency is recorded but is not used by the gate. The neural and symbolic paths can share the same measurements, so we do not claim independent evidential bases.

We now preserve matched dropout passes, independently replayable traces, rule-removal/randomization controls and five training seeds per main condition. The completed comparisons do not establish the prespecified symbolic Brier benefit. I would value your comments on three questions:

1. Is this description of the confidence initializer and truth-value revision appropriately bounded, or does the formulation still misuse NAL concepts?
2. What explicit evidential-base representation or revision restriction would be needed when neural and symbolic paths reuse the same observation?
3. Should revised frequency influence the decision interface, and what would be a principled way to study that without overstating the current results?

I can send a short formulation note, the eight-rule table, aggregate results and the updated manuscript. If you are willing to comment, I would also like your permission before quoting or attributing any written feedback in a future revision.

Best regards,
[Name]
[Affiliation, if applicable]
[Contact]
[Repository/manuscript link]

## GDC: thyroid coverage and chronology

To: **support@nci-gdc.datacommons.io**, verified on the [GDC Help Desk page](https://gdc.cancer.gov/support/help-desk).

Subject: TCGA-THCA: gene callability and pathology/genomic timing metadata

Dear GDC Help Desk,

I am [name, role and affiliation], working on a retrospective TCGA-THCA analysis. We have downloaded checksum-pinned public mutation files, but we do not treat the absence of a mutation row as a verified negative.

Could you identify source files or fields that establish, for the attached public case/gene lists:

- whether each gene was assayed with sufficient coverage/quality to support a negative call;
- the link between the specimen, diagnosis and pathology episode;
- specimen collection, pathology assessment and genomic test/report availability dates, including the meaning of relative-day offsets?

Our current pathology-detail export has no populated pathology dates/timepoints. We are checking whether clinical/biospecimen supplements provide additional information. Please distinguish measurement or specimen dates from the time a result became available.

If the evidence requires controlled access, please identify the relevant dataset, files and access process. If it was not collected or cannot be released, written confirmation would also help us limit the research claims correctly. We would prefer the smallest sufficient metadata/quality export before requesting large sequencing files.

Best regards,
[Name]
[Role/affiliation]
[Contact]

Suggested attachments: `required-tcga-cases.csv` and `required-tcga-genes.csv`. They contain public TCGA identifiers and gene names, not private patient information.

## GOSSIS/WiDS: APACHE definition and source independence

To: **gossis@mit.edu**, verified on the [GOSSIS contact page](https://gossis.mit.edu/contact/).

Subject: WiDS 2020 APACHE probability provenance and external-cohort overlap

Dear GOSSIS team,

I am [name, role and affiliation], evaluating a research model using the WiDS 2020 ICU mortality data. We are obtaining the versioned dictionary and have completed paired evaluations with and without `apache_4a_hospital_death_prob`.

Could you clarify the negative/sentinel values for that field, the APACHE model/version, the input collection window and when the exported score became available relative to the first-24-hour prediction point?

We also need the actual collection period and source-institution provenance necessary to avoid overlap in external validation. In particular, can you confirm whether the release includes BIDMC/MIMIC-derived patients or eICU data, and what source documentation could establish that a proposed new cohort is independent? We do not need identifying patient information.

Please point us to the appropriate dictionary, release notes, source custodian or access process. If a timing field or source linkage is unavailable, confirmation of that limitation would help us report the evaluation accurately.

Best regards,
[Name]
[Role/affiliation]
[Contact]

## Hospital collaborator: independent ICU cohort

Subject: Collaboration request for independent ICU mortality evaluation

Dear Dr. [Surname]/[Research data team],

I am [name, role and affiliation], working on NGTA. We have completed internal hospital-held-out studies and are seeking an authorized cohort for a frozen evaluation of hospital mortality using first-24-hour ICU information.

Could your team advise whether an eligible de-identified cohort is available, or introduce the appropriate investigator/data custodian? The attached field list covers patient/stay linkage, institutional clusters, hospital outcome, timestamped vital/laboratory measurements and input units. We need documentation of source overlap with WiDS/GOSSIS and previously used MIMIC cohorts, plus a clinically reviewed mapping to the prediction time and endpoint.

We would agree on access, ethics review, analysis rules and the mapping before inspecting evaluation outcomes. A collaborator-run evaluation or approved local workflow would be suitable if records cannot leave your institution. The current model comparisons have not established symbolic benefit, and we will preserve negative findings.

Best regards,
[Name]
[Role/affiliation]
[Contact]

Suggested attachment: `external-icu-data-request.csv`; send code or aggregate results separately as needed.
