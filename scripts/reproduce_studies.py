"""Print or execute the frozen 36-fit development recipe, without overwriting originals."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reproduce(output_dir, execute=False, resume=False, conditions=None):
    from src.data_quality import source_manifest
    from src.pipeline import PipelineConfig, run_pipeline

    recipe = json.loads((ROOT/'study_recipe.json').read_text())
    selected = [row for row in recipe['fits'] if conditions is None or row['condition'] in conditions]
    if not selected or conditions and set(conditions)-{row['condition'] for row in selected}:
        raise ValueError('Unknown or empty study condition selection')
    output = Path(output_dir).resolve()
    for row in selected:
        original = (ROOT/row['original_directory']).resolve()
        if output == original or original.is_relative_to(output) or output.is_relative_to(original):
            raise ValueError('Choose a separate output root, outside the original studies')
    plan = [dict(condition=row['condition'], seed=row['config']['seed'],
                 output=str(output/Path(row['original_directory']).parent.relative_to('results'))) for row in selected]
    if not execute:
        return dict(executed=False, fits=len(plan), plan=plan,
                    instruction='Add --execute after source acquisition and disk-space review; --resume reuses only compatible checkpoints.')
    # Validate every source before any pipeline is allowed to write outputs.
    checked = set()
    for row in selected:
        key = (row['dataset'], row['data_dir'])
        if key not in checked:
            if source_manifest(ROOT/row['data_dir'], row['dataset']) != row['sources']:
                raise ValueError(f'Source hashes differ for {key}')
            checked.add(key)
    existing = output if output.exists() else next(parent for parent in output.parents if parent.exists())
    estimated_bytes = sum(max(row['artifact_bytes'] - sum(path.stat().st_size for path in Path(item['output']).rglob('*') if path.is_file()), 0)
                          for row, item in zip(selected, plan))
    if shutil.disk_usage(existing).free < estimated_bytes * 1.2:
        raise ValueError('Insufficient free disk for the saved-study size estimate plus 20% reserve')
    for row, destination in zip(selected, plan):
        target = Path(destination['output'])
        if target.exists() and any(target.iterdir()) and not resume:
            raise ValueError('Nonempty reproduction output requires explicit --resume')
        config = PipelineConfig(**row['config'], data_dir=str(ROOT/row['data_dir']), output_dir=str(target),
                                cache_dir=str(ROOT/'.cache/ngta'), resume=resume)
        print(f"Reproducing {row['condition']} seed {config.seed}", flush=True)
        run_pipeline(config)
    return dict(executed=True, fits=len(plan), plan=plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/private/reproduction')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--conditions', nargs='+')
    print(json.dumps(reproduce(**vars(parser.parse_args())), indent=2))
