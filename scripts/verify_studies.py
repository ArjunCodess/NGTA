"""Check full-study artifacts and optionally restore every checkpoint/RNG draw."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.bundle_cache import load_cached_bundle
from src.data_quality import source_manifest
from src.modality import select_tcga_modality
from src.neural_encoder import TabularTransformerClassifier
from src.pipeline import DATASET_METADATA,PipelineConfig
from src.trace_replay import replay_bundle


def verify(checkpoint_replay=False):
    roots=[Path('results/hospital_study')/name for name in ('knn','median','knn_no_apache','median_no_apache','shuffled_labels')]
    roots += [Path('results/tcga_study')/name for name in ('fused','clinical','genomic')]
    bundles={}
    records=[]
    for root in roots:
        for path in sorted(root.glob('seed_*/*/metrics/run_summary.json')):
            directory=path.parent.parent
            saved=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
            config=PipelineConfig(**{key:value for key,value in saved['config'].items() if key in {field.name for field in fields(PipelineConfig)}})
            spec=json.loads((directory/'evaluation_spec.json').read_text())
            if saved['training_spec']!=spec:
                raise ValueError(f'Checkpoint provenance differs: {directory}')
            if source_manifest(config.data_dir,config.dataset)!=spec['sources']:
                raise ValueError(f'Source bytes changed: {directory}')
            processor=joblib.load(directory/'preprocessor.joblib')
            if processor.feature_names!=saved['feature_names'] or processor.input_dim!=saved['input_dim']:
                raise ValueError(f'Checkpoint/preprocessor features differ: {directory}')
            if not all(torch.isfinite(value).all().item() for value in saved['state_dict'].values()):
                raise ValueError(f'Nonfinite model parameters: {directory}')
            replay=replay_bundle(directory/'traces')
            if not replay['passed']:
                raise ValueError(f'Independent replay failed: {directory}')
            with np.load(directory/'traces/inference_cache.npz') as cache:
                prediction_csv=pd.read_csv(directory/'traces/test_predictions.csv')
                # Individual probability-column spelling is recorded by the exporter.
                if len(prediction_csv)!=len(cache['labels']):
                    raise ValueError(f'Prediction case count differs: {directory}')
                raw=pd.read_csv(directory/'traces/raw_test.csv')
                if not np.array_equal(prediction_csv[processor.id_column].astype(str),raw[processor.id_column].astype(str)) or not np.array_equal(prediction_csv.target,cache['labels']):
                    raise ValueError(f'Prediction case order or outcomes differ: {directory}')
                for variant in ('baseline','flat_confidence','mc_confidence_only','nars_gated'):
                    if np.max(np.abs(prediction_csv[variant+'_probability']-cache['probability__'+variant]))>1e-7:
                        raise ValueError(f'Prediction CSV differs from pass cache: {directory}/{variant}')
                passes=int(cache['probability_passes'].shape[0])
                if passes!=config.mc_samples:
                    raise ValueError(f'MC pass count differs: {directory}')
                expected=cache['probability_passes'].copy() if checkpoint_replay else None
            record=dict(directory=str(directory),independent_replay=True,events=replay['events'],
                source_identity=True,feature_identity=True,mc_samples=passes,checkpoint_restored=False)
            if checkpoint_replay:
                key=(config.dataset,config.data_dir,config.split_seed,config.imputation,config.include_apache,config.feature_mode)
                if key not in bundles:
                    options=dict(split_mode=config.split_mode,include_apache=config.include_apache,imputation=config.imputation) if config.dataset=='wids' else {}
                    bundle=load_cached_bundle(DATASET_METADATA[config.dataset]['loader'],dataset=config.dataset,
                        data_dir=config.data_dir,batch_size=config.batch_size,seed=config.split_seed,cache_dir='.cache/ngta',**options)
                    if config.dataset=='tcga':
                        bundle=select_tcga_modality(bundle,config.feature_mode,config.batch_size)
                    bundles.clear() # Retain only one fitted input condition, bounding memory.
                    bundles[key]=bundle
                bundle=bundles[key]
                if joblib.hash(bundle.preprocessor)!=joblib.hash(processor):
                    raise ValueError(f'Cached fitted preprocessor differs: {directory}')
                model=TabularTransformerClassifier(saved['input_dim'],config.d_model,config.num_heads,config.num_layers,config.dropout)
                model.load_state_dict(saved['state_dict'])
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                rng=torch.load(directory/'traces/mc_rng.pt',map_location='cpu',weights_only=True)
                restored=model.predict_with_mc_dropout(bundle.test_loader,device,config.mc_samples,replay_rng=rng)
                residual=float(np.max(np.abs(expected-restored.probability_passes)))
                if residual>1e-7:
                    raise ValueError(f'Restored checkpoint predictions differ by {residual}: {directory}')
                record.update(checkpoint_restored=True,max_pass_probability_residual=residual)
                del model,restored
            hashes={}
            for artifact in sorted(directory.glob('*.joblib')):
                with artifact.open('rb') as stream:
                    hashes[artifact.name]=hashlib.file_digest(stream,'sha256').hexdigest()
            record['fitted_artifact_sha256']=hashes
            records.append(record)
            print(f'Verified {directory}',flush=True)
    report=dict(passed=True,bundles=len(records),checkpoint_replay=checkpoint_replay,records=records,
        scope='full development studies; synthetic external integration is tested separately')
    output=Path('results/research_checks')/('checkpoint_verification.json' if checkpoint_replay else 'artifact_verification.json')
    output.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-replay',action='store_true')
    verify(**vars(parser.parse_args()))
