"""Update the plan checklist using current artifacts and explicit external blockers."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def update():
    masking_complete=all((ROOT/f"results/hospital_study/{mode}/seed_{seed}/wids/metrics/missingness/robustness_acceptance.json").exists()
        for mode in ("knn","median") for seed in range(5))
    intervals_complete=masking_complete and all((ROOT/f"results/hospital_study/{mode}/seed_{seed}/wids/metrics/missingness/uncertainty_intervals.csv").exists()
        for mode in ("knn","median") for seed in range(5))
    sensitivity_path=ROOT/'results/cross_source_sensitivity/physionet2012/verification.json'
    sensitivity_complete=sensitivity_path.exists() and json.loads(sensitivity_path.read_text()).get('fits')==10
    timing_audit_complete=(ROOT/'results/research_checks/tcga_timing/coverage.json').exists()
    notes={
        "Prove the symbolic component matters":(False,"criterion not met","Full matched clinical comparisons, direct boosts, rule removal, randomized truth, fixed priors and 100 predicate permutations are saved. Every hierarchical symbolic Brier interval spans zero and the 0.0001 minimum gain is not met. A favorable finding cannot be manufactured; a new clinically justified hypothesis needs fresh confirmation data."),
        "Strengthen uncertainty estimation":(True,"completed","Every full condition has five fitted ensemble members, MC variance, predictive entropy, common-reference error diagnostics and three independent dropout repeats. Hierarchical intervals distinguish hospitals/cases, training seeds and dropout draws."),
        "Improve the rule base":(False,"external review required","All eight rules have source/version/prototype status and sensitivity controls, but remain not_reviewed. A thyroid oncology and critical-care expert must review predicate definitions, truth weights, shared evidence and clinically defensible contradiction policies; record signed review evidence before expanding the registry."),
        "Add external validation":(False,"external cohort required","Frozen raw-masking studies and independent-cohort tooling exist. Provide an authorized independent cohort, canonical patient IDs, binary target definition, institutional clusters and verified input units/windows; freeze the mapping before accessing labels. No eligible independent cohort was supplied."),
        "Fix TCGA genomic coverage":(False,"assay evidence required","498 UUID/checksum-pinned public MAF files were acquired; selected-panel positives cover 254/50/50 train/validation/test cases. Sixteen expected clinical cases lack file-associated coverage. Complete callable panels and verified negatives remain absent; supply sourced case/gene assay callability rather than treating missing variant rows as negative."),
        "Audit TCGA record merging":(False,"timing evidence required","The physical-record audit covers all five tables and 68 model/rule inputs. All 507 selected diagnosis records have day zero; none of the 507 pathology records has a pathology date/timepoint. Molecular-test dates are absent. Source-level measurement availability and encounter links are still required; diagnosis offsets cannot certify cross-table chronology."),
        "Check APACHE data":(False,"source dictionary required","2,371 invalid probabilities are masked and twenty full with/without-score fits are saved. Official eICU prediction/APS documentation was reviewed, but does not resolve this WiDS column's negative sentinel or score timestamp. The authoritative WiDS dictionary requires registered source access; provide its versioned definition and calculation/availability provenance."),
        "Use stronger data splits":(False,"partially completed","All 20 WiDS fits use patient/hospital/ICU-disjoint held-out hospitals. TCGA partitions and positive coverage are verified, but the task also requires sourced callable genomic coverage, which remains unavailable."),
        "Test APACHE dependence":(True,"completed","Five paired with/without-APACHE seeds under each imputer, score-only/recalibrated comparators and paired hospital/seed intervals are saved. Removing APACHE lowers KNN mean ungated AUROC from 0.875721 to 0.841118. Execution establishes dependence, not superiority over the best comparator."),
        "Expand rule validation":(False,"external review required","Extraction edges, provenance, collision handling, frequency/confidence sensitivity and randomized controls pass. Clinical correlated-evidence semantics, truth weights and rule correctness still require expert review; software tests cannot certify them."),
        "Fix rule extraction":(True,"completed","The per-predicate missing, unknown, invalid, threshold and category suite covers all eight rules; extension/unknown-stage errors are fixed. Observed float64 predicate values preserve exact boundaries independently of float32 model inputs."),
        "Check multimodal rule behavior":(True,"completed","Each acquired TCGA held-out seed exports 39 real BRAF-positive events, with documented variant bytes and independent replay. All 113 clinical/genomic events are exported; 89 map and 24 are explicitly unmapped. Verified negatives remain a separate unresolved task."),
        "Evaluate human oversight":(False,"participants required","A balanced 24-reviewer/80-case crossover demonstration, separate practice cases, seeded faults, power sensitivity and crossed reviewer/case analysis are built. Recruit qualified reviewers under an approved protocol, generate a fresh private answer key, collect independent responses and grade corrections; no participant results exist."),
        "Test missingness robustness":(masking_complete,"completed" if masking_complete else "running","Both imputers are evaluated at 0/10/30/50/70% random, feature-dependent and outcome-dependent simulated raw masking with fixed models/rules, paired degradation intervals and 0.001 mask-harm bounds. Acceptance is reported per fit/scenario; this does not establish population robustness."),
        "Compare uncertainty methods":(True,"completed","Five-model ensembles, MC-dropout variance and predictive entropy are compared for each full condition using the same ensemble error reference; error-detection AUROC and tie-aware selective risk are saved."),
        "Measure selective prediction":(intervals_complete,"completed" if intervals_complete else "running","Persisted full masking predictions supply error-detection AUROC, tie-aware selective-risk area, Brier degradation and 1,000 hospital-bootstrap intervals. These are per-fit descriptive intervals, not adjusted clinical superiority evidence."),
        "Test institutional generalization":(False,"independent institution required","Twenty full WiDS models are evaluated on internal held-out hospitals without refitting preprocessing or rules. An independent source institution/time cohort is still required; internal benchmark hospitals cannot establish that broader claim."),
        "Check external compatibility":(False,"external cohort required","Frozen mapping, overlap checks, replay and noninferiority tooling exist. A public 12,000-case PhysioNet sensitivity evaluates ten no-APACHE fits with conditional case intervals and explicit clinical_claim_eligible=false. Source identity/window equivalence, original >=48-hour selection and unavailable inputs prevent it from meeting independent clinical eligibility; supply a qualified untouched cohort."),
        "Evaluate subgroups":(True,"completed","Every full hospital seed exports fixed demographic subgroup metrics and decision curves including matched MC-only. These development strata do not establish protected-group safety or prospective clinical utility."),
        "Run multiple seeds":(True,"completed","Twenty hospital fits and 15 acquired TCGA modality fits use five training seeds per condition, 50 dropout passes and three independent repeats. Actual seed means/SDs, seed-mean intervals and hierarchical paired effects are persisted."),
        "Fix stale outputs":(True,"completed","All-seed submission exports and MC decision curves are refreshed from saved predictions. Paper tables and figures explicitly select five verified current sources, record manifests, distinguish mapped/all events and seed/case intervals, and the current 13-page PDF compiles."),
        "Prevent test-set tuning":(False,"fresh confirmation required","Training/validation fit selection, immutable checkpoint provenance and source/split/rule locks are enforced. Already inspected outcomes cannot become unseen; acquire or reserve a genuinely untouched cohort and preregister the final procedure before labels are accessed."),
        "Make data acquisition reproducible":(True,"completed","An immutable 498-file GDC manifest pins publisher IDs, workflow, MD5/size, extracted SHA256 and associated cases. Verification rejects altered or unpinned MAFs and explicitly reports 16 unavailable expected cases; file coverage is never equated with gene callability."),
        "Audit references":(True,"completed","All 28 cited entries have verified primary bibliographic metadata and corrections, with links and explicit source conflicts recorded in results/research_checks/reference_metadata.json. This verifies identity, not every scientific claim attributed to each reference."),
        "Experiment 1: Establish valid data coverage and replay":(False,"assay and timing evidence required","Source pinning, missingness, training-only selection, split disjointness and independent full-study replay pass. Complete callable panels and source-level prediction-time availability remain external evidence requirements."),
        "Experiment 2: Isolate aggregation from gating":(True,"completed","Full five-seed hospital conditions use identical cached passes; uniform and ungated predictions agree within 1e-7. Hierarchical matched comparisons are saved. The additional gating-benefit criterion is not met, so no clinical benefit is claimed."),
        "Experiment 3: Test clinical rule value":(False,"criterion not met","Full rule-content controls and hierarchical comparisons are executed, but no symbolic effect meets the 0.0001 benefit margin with favorable adjusted evidence. A negative study is complete as an experiment; its requested positive scientific goal remains unestablished."),
        "Experiment 4: Validate extraction and trace replay":(True,"completed","All eight predicates have edge/unknown/missing tests; full observed-only exports independently replay including unmapped triggers, closed-form revision, gates, probabilities and rule-off effects. Masking retains a separately labeled imputed-rule control."),
        "Experiment 5: Challenge APACHE dependence":(False,"source dictionary and benefit required","Twenty full hospital fits, APACHE/recalibrated/logistic/boosting comparisons and paired intervals are saved. Authoritative score availability remains unverified, and a clinically significant gain over the strongest locked comparator is not established."),
        "Experiment 6: Test missingness and uncertainty":(masking_complete and intervals_complete,"completed" if masking_complete and intervals_complete else "running","Five-seed/imputer raw-masking studies, five-member ensembles and hospital-bootstrap uncertainty/degradation intervals execute the planned evaluation. Report the per-scenario acceptance results and failure cases; no claim of general clinical robustness follows."),
        "Experiment 7: Test external compatibility":(False,"external cohort required","The public cross-source sensitivity executes frozen predictions and numerical margins, but explicitly cannot establish independent clinical compatibility. A qualified untouched cohort with canonical source identities and reviewed measurement/outcome harmonization remains required."),
        "Experiment 8: Evaluate auditability":(False,"participants required","Protocol, randomized assignments, prospective power sensitivity, fault package and crossed reviewer/case analysis are implemented. The public demo key must be replaced with a private fresh key, and real qualified participant responses are needed to test the 10-point localization and 5-point false-reassurance margins."),
        "First: Fix data validity":(False,"assay and timing evidence required","Train-only selection, observed predicates, missingness handling, acquisition and positive genomic coverage are fixed. Full callability and clinical chronology cannot be inferred from current public variant and clinical records."),
        "Third: Make traces independently replayable":(True,"completed","Independent predicates/revision/gating, complete mapped and unmapped events, integrity hashes and full predicate edge coverage pass on current studies."),
        "Fourth: Strengthen baselines and validation":(False,"external evidence required","Full internal baselines, APACHE dependence, rule controls, ensembles and masking are executed. Independent cohorts, score timing, clinical expert review and favorable research acceptance remain unresolved; the audit specifies the required inputs."),
    }
    replacements={
        "Review imputation":"Training-standardized KNN distances, separate raw/imputed values and observed-only predicates are tested. Full KNN/median, with/without-APACHE paired fits and both five-seed masking studies now evaluate the imputer choices.",
        "Improve artifact preservation":"All fitted artifacts, exact sources/splits, pass arrays, RNG and dependency versions are preserved. Restoring all 36 full-study model/preprocessor/RNG combinations reproduces every 50-pass probability matrix exactly, with maximum residual zero.",
        "Preserve negative results":"Full hierarchical symbolic intervals span zero and fail the benefit margin; masking accepts only 12/15 median and 1/15 KNN conditions. Attention-confidence error intervals span chance. These failed findings remain explicit in the README, audit and manuscript.",
        "Add matched inference baselines":"Ungated, uniform, MC-only and NARS gates use identical cached passes in all 35 full study fits, with independent numerical replay.",
        "Add stronger predictive baselines":"Full studies include deterministic/mean-logit inference, validation recalibration, calibrated logistic regression, ExtraTrees, validation-selected boosting and WiDS score-only comparators.",
        "Test encoder-level intervention":"Acquired TCGA fused runs compare genuine encoder key bias with readout gating under replayed RNG. Identity and contextual changes are tested; clinical advantage is not established.",
        "Correct statistical comparisons":"Observed paired deltas, hospital-cluster bootstrap and multiplicity controls are supplemented by fixed-split training-seed and three-dropout-repeat hierarchical intervals; no cohort-selection uncertainty is claimed.",
        "Test rule necessity":"All-rule and individual removal, 100 prevalence-preserving predicate permutations, shuffled confidence assignments and matched irrelevant predicates are evaluated on full saved clinical caches. These negative controls do not prove clinical necessity.",
        "Separate observed and imputed triggers":"Full raw-masking studies export observed-only rules alongside an explicitly labeled imputed-rule comparator; raw/imputed measurements remain distinguishable.",
        "Improve trace completeness":"Every trigger is exported. Each acquired TCGA fit has 113 events including 24 unmapped; WiDS has 11,457 mapped observed-only events per fit. Numeric completeness applies only to mapped events.",
        "Validate end-to-end replay":"Independent persisted replay validates all full condition seeds; corrupt events, hashes, missing records and nonfinite fields are rejected in regression tests.",
        "Withdraw the held-out genomic demonstration":"The unsupported sparse-source historical claim remains withdrawn. Newly acquired sources now demonstrate recorded positive held-out BRAF events; they still do not establish verified negatives or clinical multimodal benefit.",
        "Qualify confidence-routing claims":"Historical aggregation-confounded results remain qualified; current matched full studies and hierarchical intervals do not meet the planned benefit threshold.",
    }
    content=(ROOT/'list.md').read_text()
    content=content.replace('# NGTA v2 completion checklist','# NGTA completion checklist')
    content=re.sub(r'The full evidence matrix and reasons are in .*?\.',
        'The evidence and exact blockers are in [docs/research-audit.md](docs/research-audit.md) and `results/research_checks/completion_audit.json`.',content,count=1)
    tasks=[]
    def replace_task(match):
        mark,title,description,oldnote=match.groups()
        if title in notes:
            done,status,note=notes[title]
        else:
            done,status=mark=='x','completed' if mark=='x' else 'partial'
            note=replacements.get(title,oldnote.replace('corrected v2','current').replace('v2','current implementation'))
        tasks.append(dict(task=title,completed=done,status=status,evidence_or_action=note))
        return f'- [{"x" if done else " "}] **{title}.** {description} Audit: **{status}**. {note}'
    pattern=r'- \[([ x])\] \*\*(.*?)\.\*\* (.*?) Audit: \*\*.*?\*\*\. ([^\n]+)'
    content=re.sub(pattern,replace_task,content)
    verification='''## Verification evidence

- The current regression suite passes 81 tests, including predicate edges, replay corruption, external mapping, weighted bootstrap equivalence, selected physical-record dates and public-cohort timestamp/sentinel/identity boundaries.
- Full evaluation includes 35 five-seed condition fits plus a shuffled-neural-label control, with 50 matched dropout passes, three repeats, frozen fitted artifacts and recorded dependency versions. Current figures and tables explicitly identify their source seeds; the 13-page PDF compiles without undefined references or overfull boxes.
- `results/research_checks/completion_audit.json` records every task and its exact evidence or required input. Historical `results/v2_validation/` files retain earlier snapshots and are superseded for current completion counts.
- Unchecked items remain because their scientific criterion failed or they require verified assay/timing metadata, qualified expert review, an eligible independent untouched cohort, or real reviewer responses. Those inputs cannot be fabricated by code.
- Public cross-source sensitivity and selected-record timing audits are additional evidence, not substitutes for source chronology or independent clinical eligibility. Source evidence and the separate automatic-review push blocker are documented in `docs/source-evidence.md` and `docs/push-status.md`.
'''
    content=content[:content.index('## Verification evidence')]+verification
    (ROOT/'list.md').write_text(content)
    report=dict(audit_date='2026-10-04',branch='version-2',tasks=tasks,
        completed=sum(t['completed'] for t in tasks),total=len(tasks),
        masking_complete=masking_complete,uncertainty_intervals_complete=intervals_complete,
        cross_source_sensitivity_complete=sensitivity_complete,selected_record_timing_audit_complete=timing_audit_complete,
        historical_audit='results/v2_validation/documentation_audit.json',
        evidence_scope='full development evaluation on inspected fixed cohorts; no prospective confirmation')
    target=ROOT/'results/research_checks/completion_audit.json'
    target.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    report=update()
    print(f"Completed {report['completed']}/{report['total']} tasks; masking={report['masking_complete']}; intervals={report['uncertainty_intervals_complete']}")
