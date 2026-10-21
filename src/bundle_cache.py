"""Reuse trusted local preprocessing only for identical sources and loader code."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from .data_quality import source_manifest


def load_cached_bundle(loader, *, dataset, data_dir, batch_size, seed, cache_dir=None, **options):
    arguments = dict(data_dir=data_dir, batch_size=batch_size, seed=seed, **options)
    if cache_dir is None:
        return loader(**arguments)
    code = {}
    for name in ("data_loader.py", "wids_loader.py", "bundle_cache.py"):
        code[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    spec = {"sources": source_manifest(data_dir, dataset), "dataset": dataset,
            "batch_size": batch_size, "split_seed": seed, "options": options, "code": code,
            "versions": [np.__version__, pd.__version__, sklearn.__version__, torch.__version__, joblib.__version__]}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{dataset}_{digest}.joblib"
    if destination.exists():
        saved = joblib.load(destination)
        if saved["spec"] != spec:
            raise ValueError("Preprocessing cache specification mismatch")
        print(f"Reusing source-verified preprocessing: {destination}", flush=True)
        return saved["bundle"]
    print(f"Fitting {dataset} preprocessing ({options}); cache key {digest[:12]}", flush=True)
    bundle = loader(**arguments)
    temporary = destination.with_suffix(".tmp")
    joblib.dump({"spec": spec, "bundle": bundle}, temporary, compress=3)
    temporary.replace(destination)
    print(f"Saved preprocessing: {destination}", flush=True)
    return bundle
