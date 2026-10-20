"""Map every unmet checklist criterion without turning execution into acceptance."""
import json
from pathlib import Path


GROUPS = {
    'scientific_acceptance': {'Prove the symbolic component matters', 'Experiment 3: Test clinical rule value', 'Experiment 6: Test missingness and uncertainty'},
    'qualified_clinical_review': {'Improve the rule base', 'Expand rule validation'},
    'real_participant_study': {'Evaluate human oversight', 'Experiment 8: Evaluate auditability'},
    'eligible_untouched_cohort': {'Add external validation', 'Test institutional generalization', 'Check external compatibility', 'Prevent test-set tuning', 'Experiment 7: Test external compatibility'},
    'source_provenance': {'Fix TCGA genomic coverage', 'Audit TCGA record merging', 'Check APACHE data', 'Use stronger data splits'},
    'combined_acceptance_and_provenance': {'Experiment 1: Establish valid data coverage and replay', 'Experiment 5: Challenge APACHE dependence', 'First: Fix data validity', 'Fourth: Strengthen baselines and validation'},
}


def summarize(root=Path(__file__).resolve().parents[1]):
    root = Path(root)
    audit = json.loads((root/'results/research_checks/completion_audit.json').read_text())
    rows = []
    for task in audit['tasks']:
        if task['completed']:
            continue
        groups = [name for name, names in GROUPS.items() if task['task'] in names]
        if len(groups) != 1:
            raise ValueError(f"Unclassified or multiply classified plan requirement: {task['task']}")
        rows.append(dict(requirement=task['task'], remaining_type=groups[0], scientific_criterion_satisfied=False,
                         required_evidence_or_action=task['evidence_or_action']))
    if len(rows)+audit['completed'] != audit['total']:
        raise ValueError('Checklist counts differ from task records')
    report = dict(plan='plan.md', total=audit['total'], satisfied=audit['completed'], not_satisfied=len(rows),
                  group_counts={key: sum(row['remaining_type']==key for row in rows) for key in GROUPS}, requirements=rows,
                  interpretation='Full-plan status, not a rescope or declaration of completion. Negative experiments and private study preparation do not satisfy unmet scientific or human-evidence criteria.')
    (root/'results/research_checks/full_plan_remaining.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    report = summarize()
    print(f"Plan criteria satisfied: {report['satisfied']}/{report['total']}; unsatisfied: {report['not_satisfied']}")
