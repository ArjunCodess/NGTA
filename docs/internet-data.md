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

[UCI SUPPORT2](https://archive.ics.uci.edu/dataset/880/support2) links 9,105 seriously ill hospitalized patients, with a hospital-death outcome. It is accessible through its original [HBiostat source](https://hbiostat.org/data/repo/supportdesc), without recruiting a researcher. Follow the original source's licensing and acknowledgement terms.

This is not a direct first-day WiDS substitute. The original documentation says most assessments are on study day three, and its variables differ. It could support a separately defined additional benchmark with training/validation/test separation. It cannot honestly close the frozen first-day WiDS external-compatibility criterion by renaming columns. No SUPPORT2 patient data or labels have been downloaded in this investigation.

## Other options already checked

[SICdb](https://physionet.org/content/sicdb/1.0.8/) and [HiRID](https://physionet.org/content/hirid/1.1.1/) are internet-hosted candidates, but require credentialing, training, a DUA and contributor review of the study. The [PhysioNet 2012 dataset](https://physionet.org/content/challenge-2012/1.0.0/) is already locally evaluated as a 12,000-case exploratory sensitivity; its missing inputs, selection window and independence limitations remain recorded. Re-downloading it would add no evidence.

No new clinical dataset was downloaded during this search. The next owner action, only if desired, is checking PhysioNet credentialing for the Zigong route. NGTA's current WiDS work does not wait for it.
