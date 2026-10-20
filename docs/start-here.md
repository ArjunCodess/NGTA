# NGTA: the few things I need you to do

I can handle downloads that are openly available, source inspection, coding, data preparation, experiments, analysis, paper updates and repository maintenance. You do not need to design a study or negotiate a hospital data agreement. Start with the small actions below; send back the original replies/files and I will work out the technical next step.

## 1. Get the WiDS dictionary

1. Open the [official WiDS release](https://physionet.org/content/widsdatathon2020/1.0.0/).
2. Sign in or create your own PhysioNet account and verify your email.
3. Read the access agreement shown on that release. Accept it only if you agree and can meet its terms. This release lists registered access with a signed agreement; do not start the MIMIC/CITI process for this task.
4. After access is granted, look for the WiDS Datathon 2020 Dictionary in the file list and download it. Its exact filename may differ. If the page does not expose it, tell me the wording you see and I will troubleshoot the next step.
5. Put the unchanged downloaded file in `data/private/`, or tell me its local path. Tell me the release URL you used. Do not send your password, session cookie or access token.

That is enough for your part. I will inspect the score definition and determine what it resolves. Access does not guarantee that the dictionary answers the score-timing questions.

## 2. Send two prepared source questions

These ask for documentation, not permission to obtain patient records. No institutional title is claimed. Open each text file, copy its To, Subject and Body into your email app, and send it from your own account. You can add your name at the end. Nothing has been sent automatically.

- [GDC message](action-kit/01-gdc.txt): attach [required-tcga-cases.csv](data-requests/required-tcga-cases.csv) and [required-tcga-genes.csv](data-requests/required-tcga-genes.csv). These are public TCGA IDs and gene names.
- [GOSSIS message](action-kit/02-gossis.txt): no attachment needed.

When a reply arrives, bring back its exact text and any public documentation links. If it includes restricted material or account details, keep the original locally and tell me where it is. You do not need to interpret the answer. If they request technical detail, I will draft the reply. If they say the evidence does not exist or requires institutional access, that is a useful answer too.

Contacts were checked against the [official GDC help desk](https://gdc.cancer.gov/support/help-desk) and [GOSSIS contact page](https://gossis.mit.edu/contact/) on 2026-10-20. We will continue independent work while replies are pending.

## 3. Ask for one introduction, not an entire research team

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

The detailed source requirements remain in [get-needed-data.md](get-needed-data.md). This page is the practical starting point; do only steps 1–3 for now, and report any step you cannot complete.
