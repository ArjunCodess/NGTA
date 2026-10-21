"""Train-only TCGA clinical/genomic ablations with unchanged case partitions."""
from dataclasses import replace
from torch.utils.data import DataLoader

from .data_loader import TabularPreprocessor, TabularDataset


def select_tcga_modality(bundle, mode, batch_size):
    if mode == "all":
        return bundle
    if mode not in {"clinical", "genomic"}:
        raise ValueError("TCGA feature mode must be all, clinical or genomic")
    source = bundle.preprocessor
    if mode == "genomic" and not source.binary_columns:
        raise ValueError("No train-selected genomic inputs are available")
    processor = TabularPreprocessor(id_column=source.id_column, target_column=source.target_column,
        numeric_columns=tuple(source.numeric_columns) if mode == "clinical" else (),
        categorical_columns=tuple(source.categorical_columns) if mode == "clinical" else (),
        binary_columns=tuple(source.binary_columns) if mode == "genomic" else ()).fit(bundle.train_frame)
    loaders = {}
    for split in ("train", "val", "test"):
        encoded = processor.transform(getattr(bundle, split+"_frame"))
        loaders[split+"_loader"] = DataLoader(TabularDataset(encoded), batch_size=batch_size, shuffle=split == "train")
    summary = dict(bundle.split_summary, feature_mode=mode, input_dim=processor.input_dim,
                   numeric_columns=processor.numeric_columns, binary_columns=processor.binary_columns,
                   categorical_columns=processor.categorical_columns,
                   genomic_feature_count=len(processor.binary_columns),
                   clinical_feature_count=len(processor.numeric_columns)+len(processor.categorical_columns),
                   modality_interpretation="positive variant records and unknown indicators; no unverified mutation-negative calls")
    return replace(bundle, preprocessor=processor, split_summary=summary, **loaders)
