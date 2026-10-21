# Internet dataset options for NGTA

No outside researcher is needed to continue the current WiDS development experiments. The original data, preprocessing and fitted models are available. The stopped WiDS download is not needed and will not be resumed.

## Closest practical candidate: Zigong ICU infection data

The [official release](https://physionet.org/content/icu-infection-zigong-fourth/1.1/) describes 2,790 adult infection patients from a hospital in Sichuan, China, collected in 2019-2020. It includes hospital-discharge mortality, laboratory measurements, nursing observations and department transfers. This is a candidate for another source population, not an already eligible or evaluated NGTA cohort.

The listed access requirements are personal PhysioNet credentialing, CITI Data or Specimens Only Research training and signing the project's data-use agreement. Unlike the currently checked SICdb and HiRID policies, this release does not list an additional contributor study-approval step. Account approval is still not guaranteed, and being registered for WiDS does not establish credentialed access.

Practical steps:

1. Sign into [PhysioNet credentialing](https://physionet.org/settings/credentialing/) and check whether the account is already credentialed. If it is not, follow its training and credentialing instructions. The dataset's [required-training page](https://physionet.org/content/icu-infection-zigong-fourth/view-required-training/1.1/) names the exact CITI course. Personal training must be completed by the account owner.
2. Return to the [dataset page](https://physionet.org/content/icu-infection-zigong-fourth/1.1/) and complete its data-use agreement if willing to accept its terms. The prepared [research-purpose text](data-requests/zigong-purpose.txt) describes NGTA without claiming an institutional appointment or clinical approval.
3. Once access is granted, save only the documentation and directory listing under `data/private/zigong/` first. Do not start a recursive download. I can inspect the available columns and sizes and identify the smallest necessary tables before any bulk transfer. There is no need to inspect outcomes or send patient rows in chat.

Technical work after access: identify first ICU entry from transfer records, map measurements within its first 24 hours, preserve unsupported features as unavailable, resolve units from the actual source definitions, verify repeated-patient grouping and construct hospital-death labels only after the mapping/analysis is locked. The publisher's timestamps are offsets from hospital admission, so using zero as ICU admission would be wrong. Infection-only case selection is an explicit population shift. These checks determine whether frozen evaluation is defensible; the data's presence on a website alone does not establish compatibility.

## Immediate open download: SUPPORT2, for a separate benchmark

[UCI SUPPORT2](https://archive.ics.uci.edu/dataset/880/support2) links 9,105 seriously ill hospitalized patients, with a hospital-death outcome. It is accessible through its original [HBiostat source](https://hbiostat.org/data/repo/supportdesc), without recruiting a researcher. The [publisher grants anyone permission to use these datasets](https://hbiostat.org/data/) with the original citation and acknowledgement of Vanderbilt Biostatistics. This is research-use permission, without asserting a CC redistribution license.

This is not a direct first-day WiDS substitute. The original documentation says most assessments are on study day three, and its variables differ. It could support a separately defined additional benchmark with training/validation/test separation. It cannot honestly close the frozen first-day WiDS external-compatibility criterion by renaming columns. The official ZIP was downloaded successfully without credentials on 2026-10-21. It is 782,193 bytes and contains `support2.csv`, with 9,105 records and 47 columns. Only the header and record count were inspected, not outcome values or model performance. The archive stays under ignored `data/private/open_candidates/support2csv.zip`; its SHA256 and access evidence are in [open_dataset_access.json](../results/research_checks/open_dataset_access.json).

A separate benchmark would use `hospdead`, not the longer-term `death` field. Freeze its study-day landmark and predictor list before outcome inspection. Exclude previous prognostic scores (`aps`, `sps`, `surv2m`, `surv6m`), physician survival estimates, DNR timing, outcomes and downstream utilization/cost/disability fields from predictors. Fit preprocessing and models only on training records, keep validation for selection, and reserve test outcomes until analysis is locked. The public CSV has no hospital or patient identifier column, so hospital-disjoint or canonical patient independence cannot be established from this release alone. This preparation does not complete plan experiment 7.

## Public Nepal ICU registry: promising, file access not yet working here

The [official Figshare release](https://figshare.com/articles/dataset/Epidemiology_of_critically_ill_patients_in_intensive_care_units_in_Nepal_a_retrospective_observational_study/22263847) covers adults in 17 Nepal ICUs from September 2019 through September 2022 and lists a CC BY 4.0 license. Its public metadata API works without an account and lists one 22,623,427-byte workbook, `nicrf_encounters.xlsx`, with MD5 `eeb2b9930990fa7ee44fd363dcc1758e`.

The API-listed file endpoint returned HTTP 403 here. The normal page download link returned an empty HTTP 202 response, and clicking Download file in the browser timed out with no completed download. No usable workbook was acquired. Public metadata is not proof of working file access, and the file schema has not been inspected.

The practical owner action is to open the Figshare page, click **Download (21.58 MB)**, and, if it succeeds, save `nicrf_encounters.xlsx` in `data/private/open_candidates/nepal_registry/`. No researcher invitation or credentialing step is listed on this release. Do not download another large database for this attempt. Once the workbook is available, I can inspect its dictionary/headers, check predictor timing and whether it has hospital mortality rather than ICU mortality, and assess source overlap before any outcome evaluation. The same registry network contributing to other international ICU studies means geographic separation alone cannot establish independence from WiDS.

## Other options already checked

[SICdb](https://physionet.org/content/sicdb/1.0.8/) and [HiRID](https://physionet.org/content/hirid/1.1.1/) are internet-hosted candidates, but require credentialing, training, a DUA and contributor review of the study. The [PhysioNet 2012 dataset](https://physionet.org/content/challenge-2012/1.0.0/) is already locally evaluated as a 12,000-case exploratory sensitivity; its missing inputs, selection window and independence limitations remain recorded. Re-downloading it would add no evidence.

The AFICILL and Dresden Impella Mendeley releases were also searched. Their public API requests returned HTTP 403 challenges, so neither is counted as acquired. AFICILL uses a death-or-ICU-transfer composite in a step-down population; Dresden selects cardiogenic-shock patients. Those task differences need explicit handling rather than relabeling their outcomes.

SUPPORT2 is now locally available for a separate benchmark. Trying the single Nepal workbook download is the simplest remaining owner action; Zigong remains the credentialed fallback for a closer hospital-mortality candidate. No candidate has yet established all of the original plan's independent first-day compatibility requirements.
