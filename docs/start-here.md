# NGTA: optional help with future validation

The full plan is not complete: 52 of 72 criteria are satisfied. External validation was previously deferred, but its criteria remain required by plan.md. The existing implementation and development experiments can be used while the evidence below is unavailable. The steps below are optional preparation for later validation; the current [technical reproduction guide](reproducibility.md) covers the finished implementation and experiments.

I can handle downloads that are openly available, source inspection, coding, data preparation, experiments, analysis, paper updates and repository maintenance. You do not need to design a study or negotiate a hospital data agreement. Start with the small actions below; send back the original replies/files and I will work out the technical next step.

## 1. WiDS dictionary received

Completed on 2026-10-20: the dictionary is present under `data/private/` and parses into 188 unique variable definitions. Its source definition confirms APACHE IVa but does not resolve negative score values or score availability. A creatinine unit conflict also needs publisher clarification. No further download is needed for this step. The rest of the recursive download is incomplete: training_v2.csv is 15,865 of 64,747,327 bytes, unlabeled.csv is 196,608 of 27,733,653 bytes, and solution_template.csv is empty instead of 320,363 bytes. The separate data/wids_icu.csv used for experiments is complete and matches the frozen recipe hash. In particular, WiDSDatathonOverview.pdf is only 212,974 bytes and fails its publisher checksum; it is not usable source evidence. Resume the existing authenticated terminal download, or download that PDF from the signed-in PhysioNet page, and leave the original partial file in place until a complete copy passes verification. This does not prevent running the current fitted studies.

Leave partial files in place; you do not need to sort or rename them. Run this from the repository root to check the download again:

```powershell
.\venv\Scripts\python.exe scripts/verify_wids_download.py
```

Exit code 1 means at least one required check fails; the command does not modify source files. It checks publisher hashes, directory-listing sizes, the inspected dictionary hash and the separately frozen study source. Matching byte counts alone are not proof of full-release integrity.

## 2. Source questions sent

Both prepared requests were sent through the connected Gmail account on 2026-10-20 after checking for duplicates. Their Sent labels, recipients and message bodies were read back and verified. The GDC request includes the public case and gene CSV attachments. Neither request asks for identifiable records or claims institutional authority. Do not send another copy while these requests are pending.

- [GDC request text](action-kit/01-gdc.txt) asks about gene callability and linked report availability after the public biotab/XML audit.
- [GOSSIS request text](action-kit/02-gossis.txt) asks about APACHE sentinels, timing, source overlap and the creatinine-unit conflict.

A sent question is not source evidence, so these checklist criteria remain unsatisfied. Mail identifiers and correspondence records are stored only in the ignored private directory. When a reply arrives, its source claims must be checked against the frozen data before changing scientific conclusions.

## 3. An introduction is optional if you have a suitable contact

Reply to an existing conversation with someone who has already given feedback on NGTA, or contact a researcher you actually know. Use the [introduction message](action-kit/03-introduction.txt). Attach the [short project brief](action-kit/project-brief.md), or paste it beneath the message if their email app does not handle Markdown attachments.

The request is for an introduction to a critical-care or thyroid specialist willing to inspect four rules. It also asks who could advise an independent researcher about a small reviewer study. You do not need to claim that you have ethics approval, hospital access, funding or a university appointment. If you have no suitable contact, tell me that; I can research a small list of publicly listed professional contacts and prepare messages for you to choose from.

Bring back an introduction, a willingness to review, or the reply saying they cannot help. I will prepare the specialty-specific rule sheet, questions and response form. A clinical opinion can validate a predicate's interpretation; it cannot turn an arbitrary numerical weight into an empirically calibrated probability.

## Later, only if a source or collaborator makes it worthwhile

- **MIMIC training/access:** do not download the large ICU tables yet. Independence from WiDS and compatibility are unresolved, and this disk has limited space. If a suitable route is identified, I will give you one account/training step at a time from the [official PhysioNet instructions](https://physionet.org/about/citi-course/). Training and personal identity verification must be completed by you; I can prepare and check the technical workflow.
- **Controlled TCGA data:** a public portal account is not enough for every controlled dataset. If GDC identifies required controlled files, we need an eligible institutional investigator to sponsor that access. Your action will be forwarding the prepared request to a willing collaborator, not filling an institutional role you do not have. See [GDC's access process](https://gdc.cancer.gov/access-data/obtaining-access-controlled-data).
- **Reviewer participants:** do not recruit a full study now. I will finish the technical materials; a qualified collaborator needs to advise the applicable research/ethics process before participant collection. The published demo answers cannot be used for a blinded evaluation.
- **Backup:** if you have a separate external drive with sufficient free space, connect it and tell me its drive letter and which folder I may use. I can check capacity, copy the permitted artifacts and verify hashes. A second folder on the current drive is not an independent backup.
- **GitHub purge:** the existing support ticket is already open. There is no new technical task for you unless GitHub requests owner verification or specific information. Bring back that request and I will prepare the response.

## Things you do not have to solve

You do not need to make the negative results positive, choose a more favorable analysis, or obtain a hospital dataset immediately. The paper can report an auditable implementation and negative development findings. A stronger scientific claim needs new evidence and may remain unavailable. I will keep that boundary clear rather than give you an impossible administrative checklist.

The detailed source requirements remain in [get-needed-data.md](get-needed-data.md). This page is the practical starting point; do only steps 1Ã¢â‚¬â€œ3 for now, and report any step you cannot complete.
