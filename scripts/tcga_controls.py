"""Run full clinical/genomic input controls on the newly pinned TCGA source."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import _write_submission_outputs
from src.pipeline import PipelineConfig, run_pipeline

if __name__ == "__main__":
    for mode in ("clinical", "genomic"):
        root = Path("results/tcga_study") / mode
        summaries = []
        for seed in range(5):
            print(f"Running TCGA {mode} input control, seed {seed}", flush=True)
            config = PipelineConfig(data_dir="data/acquired_tcga", output_dir=str(root / f"seed_{seed}"),
                cache_dir=".cache/ngta", dataset="tcga", feature_mode=mode, seed=seed, split_seed=0,
                epochs=60, mc_samples=50, mc_repeats=3, baseline_set="standard", ablation_set="submission",
                resume=True)
            summaries.append(run_pipeline(config))
        _write_submission_outputs(summaries, root, write_paper_tables=False)
        (root / "run_all_summary.json").write_text(json.dumps({"mode":mode,"seeds":list(range(5)),"datasets":summaries}, indent=2, default=str))
