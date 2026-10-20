"""Publish aggregate WiDS results from completed studies, without data acquisition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


CONDITIONS = ('knn', 'median', 'knn_no_apache', 'median_no_apache')
LABELS = ('KNN', 'Median', 'KNN, no APACHE', 'Median, no APACHE')


def summarize(results_root='results', markdown='docs/wids-results.md', paper_dir='paper/figures'):
    root = Path(results_root)
    encoder_path = root / 'research_checks/wids_encoder.json'
    encoder = json.loads(encoder_path.read_text())
    if encoder['fits'] != 20 or {row['condition'] for row in encoder['conditions']} != set(CONDITIONS):
        raise ValueError('Require the full four-condition, five-seed encoder experiment')
    encoded = {row['condition']: row for row in encoder['conditions']}
    lines = ['# WiDS development results', '',
        'The current WiDS implementation and available-data studies are complete. All conditions use the original data and frozen fitted preprocessing. No additional download or source clarification is an execution prerequisite. Prototype source conventions remain stated limitations, and no independent clinical validation or human-study benefit is claimed.', '',
        'There are 20 original fits: four conditions with five seeds each. Every condition uses the same 14,225 held-out encounters; repeated seeds and conditions are not additional patients. Original inference has 50 matched passes and three dropout draws. The new encoder extension reuses each checkpoint and its primary 50-pass RNG draw, without retraining.', '',
        '## Original readout results', '',
        '| Condition | Ungated AUROC | Ungated Brier | MC-only Brier | NARS Brier | NARS-minus-MC Brier, hierarchical 95% interval |',
        '|---|---:|---:|---:|---:|---|']
    summary = []
    for name, label in zip(CONDITIONS, LABELS):
        directory = root / 'hospital_study' / name
        metrics = {}
        for seed in range(5):
            with (directory / f'seed_{seed}/wids/metrics/metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                metrics.setdefault(row['variant'], []).append(row)
        if any(len(rows) != 5 for rows in metrics.values()):
            raise ValueError('Original methods differ between fitted seeds')
        aggregate = {variant: {metric: dict(mean=mean(float(row[metric]) for row in rows),
            seed_sd=stdev(float(row[metric]) for row in rows)) for metric in ('auc', 'pr_auc', 'brier', 'log_loss', 'ece')}
            for variant, rows in metrics.items()}
        contrast = json.loads((directory / 'analysis/hierarchical_symbolic_comparison.json').read_text())
        summary.append(dict(condition=name, methods=aggregate, original_symbolic_comparison=contrast))
        value = lambda variant, metric: aggregate[variant][metric]['mean']
        lines.append(f"| {label} | {value('baseline','auc'):.6f} | {value('baseline','brier'):.6f} | {value('mc_confidence_only','brier'):.6f} | {value('nars_gated','brier'):.6f} | {contrast['brier_nars_minus_mc']:.3e} [{contrast['lower_95']:.3e}, {contrast['upper_95']:.3e}] |")
    lines += ['', 'The point estimates are descriptive seed averages. Original symbolic intervals resample hospitals, training seeds and dropout repeats. None of the four original conditions establishes the 0.0001 symbolic benefit margin.', '',
        '## Additional encoder-level intervention', '',
        '| Condition | Encoder MC Brier | Encoder NARS Brier | Encoder NARS-minus-MC Brier | Family lower | Family upper |',
        '|---|---:|---:|---:|---:|---:|']
    tex = [r'\begin{table}[t]',
        r'\caption{Exploratory WiDS encoder intervention on four fixed conditions and five saved checkpoints each. NARS-minus-MC Brier and family intervals resample hospitals and training seeds, with Bonferroni adjustment across twelve secondary contrasts. The primary 50-pass dropout draw is fixed; no external confirmation or repeated-draw uncertainty is claimed. Negative is better.}',
        r'\label{tab:wids_encoder}', r'\centering\scriptsize', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lrrr@{}}', r'\toprule', r'Condition & Difference & Family lower & Family upper \\', r'\midrule']
    for name, label in zip(CONDITIONS, LABELS):
        record = encoded[name]
        if len(record['fits']) != 5 or {fit['seed'] for fit in record['fits']} != set(range(5)):
            raise ValueError('Require exactly five encoder seeds per condition')
        c = record['comparisons']['encoder_nars_minus_encoder_mc']
        brier = lambda variant: mean(fit['metrics'][variant]['brier'] for fit in record['fits'])
        lines.append(f"| {label} | {brier('encoder_mc'):.6f} | {brier('encoder_nars'):.6f} | {c['brier']:.3e} | {c['family_lower_95']:.3e} | {c['family_upper_95']:.3e} |")
        tex.append(f"{label} & {c['brier']:.2e} & {c['family_lower_95']:.2e} & {c['family_upper_95']:.2e} "+r'\\')
    residual = max(fit['uniform_encoder_max_pass_residual'] for row in encoder['conditions'] for fit in row['fits'])
    lines += ['', f'Uniform encoder gates reproduce the original per-pass probabilities with maximum residual {residual:g}. Confidence gates bias feature keys in every encoder layer and recompute contextual representations. This changes the intervention location, so encoder-versus-readout differences cannot be attributed solely to symbolic content. Full contrasts and all five-method metrics are retained in the aggregate JSON.', '',
        '## Missingness and uncertainty', '']
    masking = {}
    for name in ('knn', 'median'):
        reports = [json.loads((root / 'hospital_study' / name / f'seed_{seed}/wids/metrics/missingness/robustness_acceptance.json').read_text()) for seed in range(5)]
        passed = sum(value['accepted'] for report in reports for value in report.values())
        total = sum(len(report) for report in reports)
        masking[name] = dict(passed=passed, total=total)
        lines.append(f'- {name}: {passed}/{total} seed/scenario combinations meet the saved paired degradation/harm criteria. These are repeated development simulations, not independent populations.')
    lines += ['', 'Five-model uncertainty comparisons, calibration, selective-risk curves, subgroup metrics, decision curves, rule controls, 100 predicate permutations and the creatinine-rule-off sensitivity remain in the existing study outputs. The original negative symbolic-benefit and general-robustness findings are preserved.', '',
        '## Reproduce', '', '```powershell', 'python scripts/evaluate_wids_encoder.py', 'python scripts/summarize_wids.py', '```', '',
        'These commands use the original permitted local research artifacts; they do not download data. Private probability arrays stay under `results/private/wids_encoder_study/`. All outputs remain development evidence on inspected cases.']
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    paper = Path(paper_dir)
    paper.mkdir(parents=True, exist_ok=True)
    (paper / 'wids_encoder_results.tex').write_text('\n'.join(tex)+'\n')
    Path(markdown).write_text('\n'.join(lines)+'\n')
    report = dict(original_fits=20, encoder_fits=20, conditions=summary, masking=masking,
        uniform_encoder_max_pass_residual=residual, encoder_report_sha256=hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        source_clarification_execution_blocker=False, new_data_downloaded=False,
        independent_validation=False, negative_original_results_preserved=True)
    (root / 'research_checks/wids_status.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--markdown', default='docs/wids-results.md')
    parser.add_argument('--paper-dir', default='paper/figures')
    summarize(**vars(parser.parse_args()))
