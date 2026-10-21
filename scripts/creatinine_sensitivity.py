"""Frozen rule-off sensitivity for the unresolved WiDS creatinine-unit conflict."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import binary_metrics, paired_bootstrap_indices
from src.matched_inference import score_cached_passes
from src.trace_replay import replay_bundle


def evaluate(iterations=1000, results_root='results', output='results/research_checks/creatinine_sensitivity.json'):
    conditions = []
    for name in ('knn', 'median', 'knn_no_apache', 'median_no_apache'):
        rows, losses, groups, ids, labels = [], [], None, None, None
        for seed in range(5):
            root = Path(results_root) / 'hospital_study' / name / f'seed_{seed}/wids'
            if not replay_bundle(root/'traces')['passed']:
                raise ValueError(f'Independent replay failed: {root}')
            spec_path = root/'traces/replay_spec.json'
            spec = json.loads(spec_path.read_text())
            position = spec['feature_names'].index('d1_creatinine_max')
            raw = pd.read_csv(root/'traces/raw_test.csv', usecols=['encounter_id', 'hospital_id'])
            with np.load(root/'traces/inference_cache.npz') as cache:
                y = cache['labels']
                if ids is None:
                    ids, groups, labels = raw.encounter_id.to_numpy(), raw.hospital_id.to_numpy(), y.copy()
                elif not np.array_equal(ids, raw.encounter_id) or not np.array_equal(groups, raw.hospital_id) or not np.array_equal(labels, y):
                    raise ValueError('Sensitivity cases/outcomes/clusters are not aligned')
                confidence = cache['revised_confidence'].copy()
                confidence[:, position] = cache['neural_confidence'][:, position]
                off, _ = score_cached_passes(cache['attention_passes'], cache['token_score_passes'],
                                             cache['cls_logit_passes'], confidence, spec['gamma'])
                nars, mc = cache['probability__nars_gated'], cache['probability__mc_confidence_only']
                metrics = binary_metrics(y, off)
                control = pd.read_csv(root/'metrics/symbolic_isolation.csv').set_index('variant').loc['rule_removed:rule_creatinine']
                if abs(metrics['brier']-control.brier) > 1e-12:
                    raise ValueError('Rule-off result differs from original individual-rule ablation')
                losses.append(np.stack(((off-y)**2-(mc-y)**2, (off-y)**2-(nars-y)**2)))
                rows.append(dict(seed=seed, metrics=metrics,
                                 max_probability_change=float(np.max(np.abs(off-nars))),
                                 threshold_changes=int(np.sum((off>=.5)!=(nars>=.5))),
                                 original_control_brier_residual=float(abs(metrics['brier']-control.brier)),
                                 replay_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest()))
            print(f'Checked creatinine rule-off: {name}/{seed}', flush=True)
        losses = np.array(losses)
        rng = np.random.default_rng(0)
        samples = []
        for index in paired_bootstrap_indices(labels, iterations, rng, groups):
            seeds = rng.integers(0, 5, 5)
            samples.append(losses[seeds][:, :, index].mean(axis=(0, 2)))
        samples = np.array(samples)
        contrasts = {}
        for i, label in enumerate(('rule_off_minus_mc', 'rule_off_minus_original_nars')):
            contrasts[label] = dict(brier=float(losses[:, i].mean()),
                                   lower_95=float(np.percentile(samples[:, i], 2.5)),
                                   upper_95=float(np.percentile(samples[:, i], 97.5)))
        conditions.append(dict(condition=name, seeds=rows, contrasts=contrasts))
    report = dict(conditions=conditions, training_seeds=5, mc_samples=50, bootstrap_iterations=iterations,
                  scope='Exploratory rule-off sensitivity; fixed development cases and original dropout draw, resampled hospitals and training seeds. No retraining, input conversion, threshold selection, clinical unit resolution or prospective confirmation.',
                  unit_conflict_resolved=False)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--output', default='results/research_checks/creatinine_sensitivity.json')
    parser.add_argument('--iterations', type=int, default=1000)
    evaluate(**vars(parser.parse_args()))
