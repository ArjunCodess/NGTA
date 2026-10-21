"""Paired contrasts across prespecified training conditions, with seed variability."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import paired_bootstrap_indices


def load_condition(root, dataset, seeds):
    matrices, ids, labels, groups, identity = [], None, None, None, None
    for seed in seeds:
        directory = Path(root) / f"seed_{seed}" / dataset
        spec=json.loads((directory/"evaluation_spec.json").read_text())
        current_identity={key:spec[key] for key in ("sources","split_ids_sha256","rules")}
        if identity is None:
            identity=current_identity
        elif identity!=current_identity:
            raise ValueError("Condition seeds differ in source, split or rule identity")
        with np.load(directory / "traces" / "inference_cache.npz") as cache:
            probabilities = {key.removeprefix("probability__"): cache[key].copy() for key in cache.files if key.startswith("probability__")}
            raw = pd.read_csv(directory / "traces" / "raw_test.csv")
            id_column = "encounter_id" if dataset == "wids" else "case_submitter_id"
            current_ids = raw[id_column].to_numpy()
            if ids is None:
                ids, labels = current_ids, cache["labels"].copy()
                groups = raw.hospital_id.to_numpy() if dataset == "wids" else None
            elif not np.array_equal(ids,current_ids) or not np.array_equal(labels,cache["labels"]):
                raise ValueError("Condition seeds are not case/outcome aligned")
            matrices.append(probabilities)
    return ids,labels,groups,matrices,identity


def compare(left_root,right_root,output,dataset="wids",seeds=(0,1,2,3,4),iterations=1000):
    left=load_condition(left_root,dataset,seeds)
    right=load_condition(right_root,dataset,seeds)
    if not np.array_equal(left[0],right[0]) or not np.array_equal(left[1],right[1]):
        raise ValueError("Paired conditions have different held-out cases or outcomes")
    if left[4]!=right[4] or not np.array_equal(left[2],right[2]):
        raise ValueError("Paired conditions differ in source, partitions, rules or clusters")
    y,groups=left[1],left[2]
    rng=np.random.default_rng(0)
    indices=paired_bootstrap_indices(y,iterations,rng,groups)
    comparisons=[]
    names=sorted(set.intersection(*(set(row) for row in left[3]+right[3])))
    tail=2.5/max(len(names),1)
    for name in names:
        delta=np.array([(a[name]-y)**2-(b[name]-y)**2 for a,b in zip(left[3],right[3])])
        values=[]
        for index in indices:
            positions=rng.integers(0,len(seeds),len(seeds))
            values.append(delta[positions][:,index].mean())
        comparisons.append(dict(variant=name,brier_left_minus_right=float(delta.mean()),
            lower_95=float(np.percentile(values,2.5)),upper_95=float(np.percentile(values,97.5)),
            lower_family_95=float(np.percentile(values,tail)),upper_family_95=float(np.percentile(values,100-tail))))
    report=dict(left=str(left_root),right=str(right_root),dataset=dataset,seeds=list(seeds),
        comparisons=comparisons,bootstrap_units=["training_seed","hospital" if groups is not None else "case"],
        bootstrap_iterations=iterations,scope="paired development conditions on one fixed split; not prospective confirmation")
    destination=Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    pd.DataFrame(comparisons).to_csv(destination.with_suffix(".csv"),index=False)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root",required=True)
    parser.add_argument("--right-root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--dataset",choices=("wids","tcga"),default="wids")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--iterations",type=int,default=1000)
    print(json.dumps(compare(**vars(parser.parse_args())),indent=2))
