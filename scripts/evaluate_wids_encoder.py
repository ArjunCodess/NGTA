"""Evaluate frozen WiDS encoder gates without refitting or replacing original studies."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bundle_cache import load_cached_bundle
from src.evaluation import binary_metrics, paired_bootstrap_indices
from src.neural_encoder import TabularTransformerClassifier
from src.pipeline import DATASET_METADATA, PipelineConfig


CONDITIONS = ('knn', 'median', 'knn_no_apache', 'median_no_apache')
PAIRS = (('encoder_nars', 'encoder_mc'), ('encoder_nars', 'readout_nars'), ('encoder_mc', 'readout_mc'))


def brier_differences(predictions, labels):
    labels = np.asarray(labels, dtype=np.float64)
    losses = {name: (np.asarray(values, dtype=np.float64)-labels)**2 for name, values in predictions.items()}
    return np.stack([losses[left]-losses[right] for left, right in PAIRS])


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_intervals(losses, labels, groups, iterations=1000, seed=0):
    """Resample hospitals and fitted seeds; never treat repeated fits as patients."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 3 or losses.shape[1:] != (len(PAIRS), len(labels)) or not np.isfinite(losses).all():
        raise ValueError('Expected finite seed/contrast/case loss differences')
    rng = np.random.default_rng(seed)
    draws = []
    for index in paired_bootstrap_indices(labels, iterations, rng, groups):
        selected = rng.integers(0, losses.shape[0], losses.shape[0])
        draws.append(losses[selected][:, :, index].mean(axis=(0, 2)))
    draws = np.asarray(draws)
    # Four fixed conditions, each with three exploratory contrasts.
    tail = 100 * .05 / (2 * len(CONDITIONS) * len(PAIRS))
    return {f'{left}_minus_{right}': dict(brier=float(losses[:, i].mean()),
        lower_95=float(np.percentile(draws[:, i], 2.5)), upper_95=float(np.percentile(draws[:, i], 97.5)),
        family_lower_95=float(np.percentile(draws[:, i], tail)),
        family_upper_95=float(np.percentile(draws[:, i], 100-tail)))
        for i, (left, right) in enumerate(PAIRS)}


def evaluate(results_root='results', output_root='results/private/wids_encoder_study',
             report_path='results/research_checks/wids_encoder.json', iterations=1000):
    original = (Path(results_root) / 'hospital_study').resolve()
    destination = Path(output_root).resolve()
    if destination.is_relative_to(original) or original.is_relative_to(destination):
        raise ValueError('Encoder outputs must be separate from original studies')
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < 750 * 1024**2:
        raise ValueError('Need 750 MB free for encoder probabilities and reserve')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conditions = []
    for name in CONDITIONS:
        bundle = None
        records, losses = [], []
        case_ids = labels = groups = None
        for seed in range(5):
            source = original / name / f'seed_{seed}' / 'wids'
            spec = json.loads((source / 'evaluation_spec.json').read_text())
            saved = torch.load(source / 'model.pt', map_location='cpu', weights_only=True)
            if saved['training_spec'] != spec:
                raise ValueError('Frozen checkpoint and evaluation specification differ')
            config = PipelineConfig(**{key: value for key, value in saved['config'].items()
                if key in {field.name for field in fields(PipelineConfig)}})
            if config.dataset != 'wids' or config.seed != seed or config.split_mode != 'hospital' or config.mc_samples != 50:
                raise ValueError('Unexpected frozen WiDS configuration')
            if bundle is None:
                bundle = load_cached_bundle(DATASET_METADATA['wids']['loader'], dataset='wids', data_dir=config.data_dir,
                    batch_size=config.batch_size, seed=config.split_seed, cache_dir='.cache/ngta',
                    split_mode=config.split_mode, include_apache=config.include_apache, imputation=config.imputation)
            if joblib.hash(bundle.preprocessor) != joblib.hash(joblib.load(source / 'preprocessor.joblib')):
                raise ValueError('Frozen fitted preprocessing differs from the reused bundle')
            raw = pd.read_csv(source / 'traces/raw_test.csv')
            with np.load(source / 'traces/inference_cache.npz') as cache:
                y = cache['labels'].copy()
                ids = raw.encounter_id.to_numpy()
                if labels is None:
                    labels, case_ids, groups = y, ids, raw.hospital_id.to_numpy()
                elif not np.array_equal(labels, y) or not np.array_equal(case_ids, ids):
                    raise ValueError('Frozen seed cases or labels are not aligned')
                neural, revised = cache['neural_confidence'].copy(), cache['revised_confidence'].copy()
                expected = cache['probability_passes'].copy()
                predictions = dict(ungated=expected.mean(0), readout_mc=cache['probability__mc_confidence_only'].copy(),
                                   readout_nars=cache['probability__nars_gated'].copy())
            rng = torch.load(source / 'traces/mc_rng.pt', map_location='cpu', weights_only=True)
            identity = dict(checkpoint_sha256=sha256(source / 'model.pt'),
                evaluation_spec_sha256=sha256(source / 'evaluation_spec.json'), rng_sha256=sha256(source / 'traces/mc_rng.pt'))
            path = destination / name / f'seed_{seed}.npz'
            metadata = path.with_suffix('.json')
            if path.exists() != metadata.exists():
                raise ValueError('Incomplete encoder result; preserve it and select a new output root')
            if path.exists():
                record = json.loads(metadata.read_text())
                if record['identity'] != identity or sha256(path) != record['probability_sha256']:
                    raise ValueError('Existing encoder result does not match its frozen source or hash')
                with np.load(path) as cache:
                    if not np.array_equal(cache['case_ids'], case_ids) or not np.array_equal(cache['labels'], labels):
                        raise ValueError('Existing encoder cases differ')
                    predictions.update({variant: cache[variant].mean(0) for variant in ('encoder_mc', 'encoder_nars')})
            else:
                model = TabularTransformerClassifier(saved['input_dim'], config.d_model, config.num_heads, config.num_layers, config.dropout).to(device)
                model.load_state_dict(saved['state_dict'])
                passes, residual = {}, None
                for variant, confidence in (('uniform_encoder', np.ones_like(neural)), ('encoder_mc', neural), ('encoder_nars', revised)):
                    summary = model.predict_with_mc_dropout(bundle.test_loader, device, 50,
                        feature_confidence=confidence, gamma=config.gamma, replay_rng=rng)
                    if not np.array_equal(summary.labels, labels):
                        raise ValueError('Frozen loader case labels differ')
                    if variant == 'uniform_encoder':
                        residual = float(np.max(np.abs(summary.probability_passes - expected)))
                        if residual > 1e-7:
                            raise ValueError(f'Uniform encoder identity residual {residual}')
                    else:
                        passes[variant] = summary.probability_passes.copy()
                        predictions[variant] = summary.probabilities_mean.copy()
                    del summary
                    print(f'{name}/{seed}: {variant}', flush=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, case_ids=case_ids, labels=labels, **passes)
                record = dict(seed=seed, identity=identity, uniform_encoder_max_pass_residual=residual,
                    probability_sha256=sha256(path), metrics={variant: binary_metrics(labels, p) for variant, p in predictions.items()},
                    threshold_changes_encoder_nars_vs_mc=int(np.sum((predictions['encoder_nars'] >= .5) != (predictions['encoder_mc'] >= .5))))
                metadata.write_text(json.dumps(record, indent=2))
                del model, passes
            records.append(record)
            losses.append(brier_differences(predictions, labels))
        conditions.append(dict(condition=name, fits=records, comparisons=paired_intervals(losses, labels, groups, iterations),
            cases=len(labels), hospitals=len(np.unique(groups))))
        del bundle
    report = dict(conditions=conditions, fits=20, mc_passes=50, bootstrap_iterations=iterations,
        intervention='Fixed saved feature confidence biases feature keys inside every encoder layer; contextual representations are recomputed.',
        sampling_units=['hospital', 'training_seed'], secondary_interval_policy='Bonferroni across 12 fixed condition/contrast comparisons',
        scope='Exploratory extension on already inspected development cases, using each original checkpoint and primary saved dropout draw. No retraining, source reinterpretation, threshold tuning or independent confirmation.',
        new_training=False, new_data=False, clinical_benefit_claim=False)
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--output-root', default='results/private/wids_encoder_study')
    parser.add_argument('--report-path', default='results/research_checks/wids_encoder.json')
    parser.add_argument('--iterations', type=int, default=1000)
    evaluate(**vars(parser.parse_args()))
