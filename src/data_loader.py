from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .gdc_downloader import ensure_tcga_thca_maf

pd.set_option("future.no_silent_downcasting", True)

DEFAULT_TARGET_COLUMN = "diagnoses.ajcc_pathologic_n"
DEFAULT_ID_COLUMN = "case_submitter_id"
DEFAULT_NUMERIC_COLUMNS = (
    "diagnoses.age_at_diagnosis",
    "diagnoses.year_of_diagnosis",
    "pathology_details.tumor_length_measurement",
    "pathology_details.tumor_width_measurement",
    "pathology_details.tumor_depth_measurement",
)
DEFAULT_CATEGORICAL_COLUMNS = (
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.prior_malignancy",
    "diagnoses.synchronous_malignancy",
    "diagnoses.prior_treatment",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.laterality",
    "diagnoses.tumor_focality",
    "diagnoses.residual_disease",
    "pathology_details.extrathyroid_extension",
)
DEFAULT_BINARY_COLUMNS: tuple[str, ...] = ()
TARGET_MAP = {"N0": 0, "N1": 1, "N1a": 1, "N1b": 1}
MISSING_PLACEHOLDERS = {
    "--": np.nan,
    "'--": np.nan,
    "Not Reported": np.nan,
    "not reported": np.nan,
    "Unknown": np.nan,
    "unknown": np.nan,
}
TABLE_FILES = (
    "clinical.tsv",
    "exposure.tsv",
    "family_history.tsv",
    "follow_up.tsv",
    "pathology_detail.tsv",
)
GENOMIC_FEATURE_PREFIX = "genomic_mutation__"
FUNCTIONAL_VARIANT_CLASSES = (
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
)
MAX_GENOMIC_GENES = 50
MAF_GLOB_PATTERNS = ("*.maf", "*tcga_mutations.tsv", "*mutations*.tsv")


@dataclass
class EncodedFrame:
    features: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PreprocessorMetadata:
    target_column: str
    id_column: str
    numeric_columns: list[str]
    binary_columns: list[str]
    categorical_columns: list[str]
    dropped_missing_columns: list[str]
    dropped_constant_columns: list[str]
    output_feature_names: list[str]
    target_mapping: dict[str, int]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    merged_frame: pd.DataFrame
    labeled_frame: pd.DataFrame
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    preprocessor: "TabularPreprocessor"
    split_summary: dict[str, Any]


class TabularDataset(Dataset):
    def __init__(self, encoded: EncodedFrame) -> None:
        self.features = torch.tensor(encoded.features, dtype=torch.float32)
        self.target = torch.tensor(encoded.target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.target[index]


class TabularPreprocessor:
    def __init__(
        self,
        target_column: str = DEFAULT_TARGET_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        numeric_columns: tuple[str, ...] = DEFAULT_NUMERIC_COLUMNS,
        binary_columns: tuple[str, ...] = DEFAULT_BINARY_COLUMNS,
        categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS,
    ) -> None:
        self.target_column = target_column
        self.id_column = id_column
        self.numeric_columns = list(numeric_columns)
        self.binary_columns = list(binary_columns)
        self.categorical_columns = list(categorical_columns)
        self.dropped_missing_columns: list[str] = []
        self.dropped_constant_columns: list[str] = []
        self.numeric_imputer: KNNImputer | None = None
        self.categorical_imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.output_feature_names_: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(self.output_feature_names_)

    @property
    def input_dim(self) -> int:
        return len(self.output_feature_names_)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        required = {self.id_column, self.target_column}
        required.update(self.numeric_columns)
        required.update(self.binary_columns)
        required.update(self.categorical_columns)
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        feature_columns = self.numeric_columns + self.binary_columns + self.categorical_columns
        self.dropped_constant_columns = [
            column for column in feature_columns if frame[column].nunique(dropna=False) <= 1
        ]
        self.numeric_columns = [
            column for column in self.numeric_columns if column not in self.dropped_constant_columns
        ]
        self.binary_columns = [
            column for column in self.binary_columns if column not in self.dropped_constant_columns
        ]
        self.categorical_columns = [
            column for column in self.categorical_columns if column not in self.dropped_constant_columns
        ]

        numeric_frame = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        categorical_frame = frame[self.categorical_columns].astype("object")

        self.numeric_imputer = KNNImputer(n_neighbors=5)
        numeric_imputed = self.numeric_imputer.fit_transform(numeric_frame)
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_imputed)

        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_imputed = self.categorical_imputer.fit_transform(categorical_frame)
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder.fit(categorical_imputed)

        numeric_feature_names = list(self.numeric_columns)
        categorical_feature_names = list(
            self.encoder.get_feature_names_out(self.categorical_columns)
        )
        self.output_feature_names_ = numeric_feature_names + list(self.binary_columns) + categorical_feature_names
        return self

    def transform(self, frame: pd.DataFrame) -> EncodedFrame:
        if (
            self.numeric_imputer is None
            or self.categorical_imputer is None
            or self.scaler is None
            or self.encoder is None
        ):
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        metadata_columns = [self.id_column, self.target_column]
        metadata = frame[metadata_columns].reset_index(drop=True).copy()
        target = metadata[self.target_column].astype(np.float32).to_numpy()

        numeric_frame = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        binary_frame = (
            frame[self.binary_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if self.binary_columns
            else pd.DataFrame(index=frame.index)
        )
        categorical_frame = frame[self.categorical_columns].astype("object")

        numeric_transformed = self.scaler.transform(self.numeric_imputer.transform(numeric_frame))
        categorical_transformed = self.encoder.transform(
            self.categorical_imputer.transform(categorical_frame)
        )
        feature_blocks = [numeric_transformed.astype(np.float32)]
        if self.binary_columns:
            feature_blocks.append(binary_frame.to_numpy(dtype=np.float32, copy=True))
        feature_blocks.append(categorical_transformed.astype(np.float32))
        features = np.concatenate(feature_blocks, axis=1)

        return EncodedFrame(features=features, target=target, metadata=metadata)

    def metadata(self) -> PreprocessorMetadata:
        return PreprocessorMetadata(
            target_column=self.target_column,
            id_column=self.id_column,
            numeric_columns=self.numeric_columns,
            binary_columns=self.binary_columns,
            categorical_columns=self.categorical_columns,
            dropped_missing_columns=self.dropped_missing_columns,
            dropped_constant_columns=self.dropped_constant_columns,
            output_feature_names=self.output_feature_names_,
            target_mapping=TARGET_MAP,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self.metadata()), indent=2), encoding="utf-8")


def _looks_like_metadata_row(frame: pd.DataFrame) -> bool:
    if len(frame) < 1:
        return False

    row = frame.iloc[0].fillna("").astype(str).str.strip().str.lower()
    if row.empty:
        return False

    header_like_ratio = float(
        np.mean(row.to_numpy() == pd.Index(frame.columns).astype(str).str.strip().str.lower().to_numpy())
    )
    descriptor_hits = row.str.contains(r"(?:cde|descriptor|definition|data element)", regex=True).mean()
    repeated_values_ratio = row.nunique(dropna=False) <= 2
    return header_like_ratio >= 0.5 or descriptor_hits >= 0.5 or (
        repeated_values_ratio and descriptor_hits > 0.0
    )


def _read_tcga_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if _looks_like_metadata_row(frame):
        frame = frame.iloc[1:].reset_index(drop=True)
    frame = frame.replace(list(MISSING_PLACEHOLDERS.keys()), np.nan)
    if "cases.submitter_id" not in frame.columns:
        raise ValueError(f"Missing cases.submitter_id in {path.name}")
    frame = frame.rename(columns={"cases.submitter_id": DEFAULT_ID_COLUMN})
    return frame


def _first_non_null(series: pd.Series) -> Any:
    for value in series:
        if pd.notna(value):
            return value
    return np.nan


def _collapse_case_table(frame: pd.DataFrame, prefer_primary: bool = False) -> pd.DataFrame:
    if prefer_primary and "diagnoses.classification_of_tumor" in frame.columns:
        primary_mask = frame["diagnoses.classification_of_tumor"].fillna("").str.lower().eq("primary")
        if primary_mask.any():
            frame = frame.loc[primary_mask].copy()

    collapsed = frame.groupby(DEFAULT_ID_COLUMN, dropna=False).agg(_first_non_null)

    if "treatments.treatment_or_therapy" in frame.columns:
        treatment_flag = frame.groupby(DEFAULT_ID_COLUMN)["treatments.treatment_or_therapy"].agg(
            lambda series: (
                "yes"
                if series.fillna("").astype(str).str.lower().eq("yes").any()
                else (
                    "no"
                    if series.fillna("").astype(str).str.lower().eq("no").any()
                    else np.nan
                )
            )
        )
        collapsed = collapsed.join(
            treatment_flag.rename("derived.any_treatment_or_therapy"),
            how="left",
        )

    # Defragment after the wide groupby aggregation so downstream resets/joins do not
    # emit the pandas fragmentation warning during the TCGA pipeline.
    return collapsed.copy().reset_index()


def _drop_sparse_columns(frame: pd.DataFrame, threshold: float = 0.70) -> tuple[pd.DataFrame, list[str]]:
    missing_ratio = frame.isna().mean()
    dropped_columns = sorted(missing_ratio[missing_ratio > threshold].index.tolist())
    return frame.drop(columns=dropped_columns), dropped_columns


def _find_genomic_maf_files(data_dir: Path) -> list[Path]:
    matches: list[Path] = []
    for pattern in MAF_GLOB_PATTERNS:
        matches.extend(sorted(data_dir.glob(pattern)))
    deduplicated = sorted({path.resolve(): path for path in matches}.values(), key=lambda path: path.name)
    if deduplicated:
        return deduplicated

    ensured_path = ensure_tcga_thca_maf(download_dir=data_dir)
    matches = [ensured_path]
    return matches


def _load_genomic_binary_matrix(
    data_dir: str | Path,
    max_genes: int = MAX_GENOMIC_GENES,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    maf_paths = _find_genomic_maf_files(Path(data_dir))
    maf_frames = [pd.read_csv(path, sep="\t", comment="#", low_memory=False) for path in maf_paths]
    maf_frame = pd.concat(maf_frames, ignore_index=True)
    required_columns = {"Tumor_Sample_Barcode", "Variant_Classification", "Hugo_Symbol"}
    missing_columns = sorted(required_columns - set(maf_frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required MAF columns in genomic input: {missing_columns}")

    maf_frame["case_submitter_id"] = (
        maf_frame["Tumor_Sample_Barcode"].astype(str).str.slice(0, 12)
    )
    filtered = maf_frame.loc[
        maf_frame["Variant_Classification"].isin(FUNCTIONAL_VARIANT_CLASSES)
        & maf_frame["Hugo_Symbol"].notna()
        & maf_frame["case_submitter_id"].notna()
    ].copy()
    top_genes = (
        filtered["Hugo_Symbol"].value_counts().head(max_genes).index.astype(str).tolist()
    )
    filtered = filtered.loc[filtered["Hugo_Symbol"].isin(top_genes)].copy()
    filtered["mutated"] = 1

    genomic_matrix = (
        filtered.groupby([DEFAULT_ID_COLUMN, "Hugo_Symbol"], dropna=False)["mutated"]
        .max()
        .unstack(fill_value=0)
        .reindex(columns=top_genes, fill_value=0)
        .astype(np.int8)
        .reset_index()
    )
    renamed_columns = {
        gene: f"{GENOMIC_FEATURE_PREFIX}{gene}" for gene in genomic_matrix.columns if gene != DEFAULT_ID_COLUMN
    }
    genomic_matrix = genomic_matrix.rename(columns=renamed_columns)
    genomic_columns = [column for column in genomic_matrix.columns if column != DEFAULT_ID_COLUMN]
    return genomic_matrix, genomic_columns, [path.name for path in maf_paths]


def load_merged_tcga_frame(data_dir: str | Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    base_dir = Path(data_dir)
    missing_files = [name for name in TABLE_FILES if not (base_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing TCGA table(s): {missing_files}")

    clinical = _collapse_case_table(_read_tcga_table(base_dir / "clinical.tsv"), prefer_primary=True)
    merged = clinical
    for table_name in TABLE_FILES[1:]:
        collapsed = _collapse_case_table(_read_tcga_table(base_dir / table_name))
        merged = merged.merge(collapsed, on=DEFAULT_ID_COLUMN, how="left", suffixes=("", f"__{table_name}"))

    merged, dropped_columns = _drop_sparse_columns(merged, threshold=0.70)
    genomic_matrix, genomic_columns, maf_names = _load_genomic_binary_matrix(base_dir)
    merged = merged.merge(genomic_matrix, on=DEFAULT_ID_COLUMN, how="left")
    if genomic_columns:
        merged[genomic_columns] = (
            merged[genomic_columns].fillna(0).astype(np.int8)
        )
    return merged, dropped_columns, genomic_columns, maf_names


def _prepare_labeled_frame(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    frame[DEFAULT_TARGET_COLUMN] = frame[DEFAULT_TARGET_COLUMN].map(TARGET_MAP)
    frame = frame.dropna(subset=[DEFAULT_TARGET_COLUMN]).reset_index(drop=True)
    frame[DEFAULT_TARGET_COLUMN] = frame[DEFAULT_TARGET_COLUMN].astype(np.int64)
    return frame


def _stratified_split(
    frame: pd.DataFrame,
    target_column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = frame[target_column].to_numpy()
    indices = np.arange(len(frame))
    train_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_indices, holdout_indices = next(train_splitter.split(indices.reshape(-1, 1), stratify))

    holdout_frame = frame.iloc[holdout_indices].reset_index(drop=True)
    holdout_targets = holdout_frame[target_column].to_numpy()
    val_test_indices = np.arange(len(holdout_frame))
    val_test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_indices, test_indices = next(
        val_test_splitter.split(val_test_indices.reshape(-1, 1), holdout_targets)
    )

    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    val_frame = holdout_frame.iloc[val_indices].reset_index(drop=True)
    test_frame = holdout_frame.iloc[test_indices].reset_index(drop=True)
    return train_frame, val_frame, test_frame


def load_data_bundle(
    data_dir: str | Path,
    batch_size: int,
    seed: int = 0,
) -> DataBundle:
    merged_frame, dropped_missing_columns, genomic_columns, maf_file_names = load_merged_tcga_frame(data_dir)
    labeled_frame = _prepare_labeled_frame(merged_frame)
    train_frame, val_frame, test_frame = _stratified_split(
        labeled_frame,
        target_column=DEFAULT_TARGET_COLUMN,
        seed=seed,
    )

    preprocessor = TabularPreprocessor(binary_columns=tuple(genomic_columns)).fit(train_frame)
    preprocessor.dropped_missing_columns = dropped_missing_columns
    encoded_train = preprocessor.transform(train_frame)
    encoded_val = preprocessor.transform(val_frame)
    encoded_test = preprocessor.transform(test_frame)

    train_loader = DataLoader(TabularDataset(encoded_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(encoded_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(encoded_test), batch_size=batch_size, shuffle=False)

    split_summary = {
        "raw_case_rows": int(len(merged_frame)),
        "labeled_case_rows": int(len(labeled_frame)),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive": int(train_frame[DEFAULT_TARGET_COLUMN].sum()),
        "val_positive": int(val_frame[DEFAULT_TARGET_COLUMN].sum()),
        "test_positive": int(test_frame[DEFAULT_TARGET_COLUMN].sum()),
        "positive_label": "Lymph Node Metastasis",
        "target_column": DEFAULT_TARGET_COLUMN,
        "id_column": DEFAULT_ID_COLUMN,
        "maf_files": maf_file_names,
        "numeric_columns": preprocessor.numeric_columns,
        "binary_columns": preprocessor.binary_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "clinical_feature_count": int(len(preprocessor.numeric_columns) + len(preprocessor.categorical_columns)),
        "genomic_feature_count": int(len(preprocessor.binary_columns)),
        "input_dim": preprocessor.input_dim,
        "output_feature_names": preprocessor.feature_names,
        "dropped_missing_columns": dropped_missing_columns,
        "dropped_constant_columns": preprocessor.dropped_constant_columns,
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        merged_frame=merged_frame,
        labeled_frame=labeled_frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        preprocessor=preprocessor,
        split_summary=split_summary,
    )
