"""Rebuild decision curves, grouping manifests and orchestration from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import _write_submission_outputs
from src.pipeline import _build_decision_curve_frame, _save_decision_curve_plot, DATASET_METADATA


def refresh(root,dataset,seeds=(0,1,2,3,4)):
    root=Path(root)
    summaries=[]
    for seed in seeds:
        directory=root/f"seed_{seed}"/dataset
        summary_path=directory/"metrics"/"run_summary.json"
        summary=json.loads(summary_path.read_text())
        with np.load(directory/"traces"/"inference_cache.npz") as cache:
            curve=_build_decision_curve_frame(cache["labels"],cache["probability__random_forest"],
                cache["probability__baseline"],cache["probability__flat_confidence"],cache["probability__nars_gated"],
                cache["probability__mc_confidence_only"])
        curve.to_csv(directory/"metrics"/"decision_curve.csv",index=False)
        _save_decision_curve_plot(curve,directory/"charts"/"decision_curve.png",DATASET_METADATA[dataset]["positive_class"])
        summary["decision_curve"]=curve.to_dict(orient="records")
        summary_path.write_text(json.dumps(summary,indent=2))
        if dataset=="wids" and not (directory/"traces"/"development_groups.csv").exists():
            source_path=Path(summary["config"]["data_dir"])/"wids_icu.csv"
            spec=json.loads((directory/"evaluation_spec.json").read_text())
            sources=spec["sources"]
            expected=next(item["sha256"] for item in sources if item["file"]=="wids_icu.csv")
            with source_path.open("rb") as source:
                digest=hashlib.file_digest(source,"sha256").hexdigest()
            if digest!=expected:
                raise ValueError("Development patient mapping requires the original verified source")
            groups=pd.read_csv(source_path,usecols=["encounter_id","patient_id","hospital_id","icu_id"])
            manifest=pd.read_csv(directory/"traces"/"split_ids.csv")
            merged=manifest.merge(groups,on="encounter_id",validate="one_to_one")
            if len(merged)!=len(manifest):
                raise ValueError("Source grouping fields do not cover every development case")
            merged.to_csv(directory/"traces"/"development_groups.csv",index=False)
        summaries.append(summary)
    _write_submission_outputs(summaries,root,write_paper_tables=False)
    (root/"run_all_summary.json").write_text(json.dumps({"mode":"multi_seed","seeds":list(seeds),
        "datasets":{f"{dataset}_seed_{seed}":summary for seed,summary in zip(seeds,summaries)}},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--dataset",choices=("wids","tcga"),default="wids")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    refresh(**vars(parser.parse_args()))
