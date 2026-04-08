from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .data_loader import DataBundle, EncodedFrame
from .wids_knowledge_base import RULE_ORDER

WIDS_ID_COLUMN = "encounter_id"
WIDS_TARGET_COLUMN = "hospital_death"
WIDS_CONTINUOUS_COLUMNS = (
    "age",
    "bmi",
    "d1_heartrate_max",
    "d1_sysbp_min",
    "d1_temp_max",
    "d1_lactate_max",
    "d1_bun_max",
    "d1_creatinine_max",
    "d1_glucose_max",
    "d1_wbc_max",
    "d1_spo2_min",
    "d1_platelets_min",
    "apache_4a_hospital_death_prob",
)
WIDS_BINARY_COLUMNS = ("elective_surgery",)
WIDS_CATEGORICAL_COLUMNS = ("gender",)
WIDS_REQUIRED_COLUMNS = (
    WIDS_ID_COLUMN,
    WIDS_TARGET_COLUMN,
    *WIDS_CONTINUOUS_COLUMNS,
    *WIDS_BINARY_COLUMNS,
    *WIDS_CATEGORICAL_COLUMNS,
)
WIDS_RULE_COLUMN_ORDER = RULE_ORDER


@dataclass
class WIDSEncodedFrame(EncodedFrame):
    continuous_features: np.ndarray
    auxiliary_features: np.ndarray
    rule_triggers: np.ndarray


class WIDSDataset(Dataset):
    def __init__(self, encoded: WIDSEncodedFrame) -> None:
        self.features = torch.tensor(encoded.features, dtype=torch.float32)
        self.target = torch.tensor(encoded.target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.target[index]


class WIDSPreprocessor:
    def __init__(self) -> None:
        self.id_column = WIDS_ID_COLUMN
        self.target_column = WIDS_TARGET_COLUMN
        self.numeric_columns = list(WIDS_CONTINUOUS_COLUMNS)
        self.binary_columns = list(WIDS_BINARY_COLUMNS)
        self.categorical_columns = list(WIDS_CATEGORICAL_COLUMNS)
        self.numeric_imputer: KNNImputer | None = None
        self.binary_imputer: SimpleImputer | None = None
        self.categorical_imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.output_feature_names_: list[str] = []
        self.rule_names = list(WIDS_RULE_COLUMN_ORDER)
        self.dropped_missing_columns: list[str] = []
        self.dropped_constant_columns: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(self.output_feature_names_)

    @property
    def input_dim(self) -> int:
        return len(self.output_feature_names_)

    def fit(self, frame: pd.DataFrame) -> "WIDSPreprocessor":
        missing_columns = sorted(set(WIDS_REQUIRED_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Missing required WiDS columns: {missing_columns}")

        numeric_frame = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        binary_frame = frame[self.binary_columns].apply(pd.to_numeric, errors="coerce")
        categorical_frame = frame[self.categorical_columns].astype("object")

        self.numeric_imputer = KNNImputer(n_neighbors=5)
        numeric_imputed = self.numeric_imputer.fit_transform(numeric_frame)
        self.scaler = StandardScaler()
        self.scaler.fit(numeric_imputed)

        self.binary_imputer = SimpleImputer(strategy="most_frequent")
        self.binary_imputer.fit(binary_frame)

        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_imputed = self.categorical_imputer.fit_transform(categorical_frame)
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder.fit(categorical_imputed)

        categorical_feature_names = list(self.encoder.get_feature_names_out(self.categorical_columns))
        self.output_feature_names_ = list(self.numeric_columns) + list(self.binary_columns) + categorical_feature_names
        return self

    def transform_components(self, frame: pd.DataFrame) -> WIDSEncodedFrame:
        if (
            self.numeric_imputer is None
            or self.binary_imputer is None
            or self.categorical_imputer is None
            or self.scaler is None
            or self.encoder is None
        ):
            raise RuntimeError("Preprocessor must be fitted before calling transform_components().")

        metadata = frame[[self.id_column, self.target_column]].reset_index(drop=True).copy()
        target = metadata[self.target_column].astype(np.float32).to_numpy()

        numeric_frame = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        binary_frame = frame[self.binary_columns].apply(pd.to_numeric, errors="coerce")
        categorical_frame = frame[self.categorical_columns].astype("object")

        numeric_imputed = self.numeric_imputer.transform(numeric_frame)
        numeric_imputed_frame = pd.DataFrame(numeric_imputed, columns=self.numeric_columns, index=frame.index)
        binary_imputed = self.binary_imputer.transform(binary_frame).astype(np.float32)
        categorical_imputed = self.categorical_imputer.transform(categorical_frame)
        categorical_encoded = self.encoder.transform(categorical_imputed).astype(np.float32)
        numeric_scaled = self.scaler.transform(numeric_imputed).astype(np.float32)

        rule_triggers = np.column_stack(
            [
                numeric_imputed_frame["d1_lactate_max"].to_numpy(dtype=np.float32) >= 4.0,
                numeric_imputed_frame["d1_sysbp_min"].to_numpy(dtype=np.float32) <= 90.0,
                numeric_imputed_frame["age"].to_numpy(dtype=np.float32) >= 75.0,
                numeric_imputed_frame["d1_creatinine_max"].to_numpy(dtype=np.float32) >= 2.0,
            ]
        ).astype(np.float32)
        auxiliary_features = np.concatenate([binary_imputed, categorical_encoded], axis=1).astype(np.float32)
        features = np.concatenate([numeric_scaled, auxiliary_features], axis=1).astype(np.float32)

        return WIDSEncodedFrame(
            features=features,
            target=target,
            metadata=metadata,
            continuous_features=numeric_scaled,
            auxiliary_features=auxiliary_features,
            rule_triggers=rule_triggers,
        )

    def transform(self, frame: pd.DataFrame) -> EncodedFrame:
        encoded = self.transform_components(frame)
        return EncodedFrame(features=encoded.features, target=encoded.target, metadata=encoded.metadata)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "target_column": self.target_column,
            "id_column": self.id_column,
            "numeric_columns": self.numeric_columns,
            "binary_columns": self.binary_columns,
            "categorical_columns": self.categorical_columns,
            "rule_names": self.rule_names,
            "output_feature_names": self.output_feature_names_,
        }
        output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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


def load_wids_data_bundle(
    data_dir: str | Path,
    batch_size: int,
    seed: int = 0,
) -> DataBundle:
    csv_path = Path(data_dir) / "wids_icu.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing WiDS CSV: {csv_path}")

    frame = pd.read_csv(csv_path, na_values=["NA"], usecols=list(WIDS_REQUIRED_COLUMNS))
    frame = frame.dropna(subset=[WIDS_TARGET_COLUMN]).reset_index(drop=True)
    frame[WIDS_TARGET_COLUMN] = frame[WIDS_TARGET_COLUMN].astype(np.int64)

    train_frame, val_frame, test_frame = _stratified_split(
        frame,
        target_column=WIDS_TARGET_COLUMN,
        seed=seed,
    )

    preprocessor = WIDSPreprocessor().fit(train_frame)
    encoded_train = preprocessor.transform_components(train_frame)
    encoded_val = preprocessor.transform_components(val_frame)
    encoded_test = preprocessor.transform_components(test_frame)

    train_loader = DataLoader(WIDSDataset(encoded_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WIDSDataset(encoded_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WIDSDataset(encoded_test), batch_size=batch_size, shuffle=False)

    split_summary: dict[str, Any] = {
        "raw_case_rows": int(len(frame)),
        "labeled_case_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive": int(train_frame[WIDS_TARGET_COLUMN].sum()),
        "val_positive": int(val_frame[WIDS_TARGET_COLUMN].sum()),
        "test_positive": int(test_frame[WIDS_TARGET_COLUMN].sum()),
        "positive_label": "Hospital Mortality",
        "target_column": WIDS_TARGET_COLUMN,
        "id_column": WIDS_ID_COLUMN,
        "numeric_columns": preprocessor.numeric_columns,
        "binary_columns": preprocessor.binary_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "rule_names": preprocessor.rule_names,
        "clinical_feature_count": int(
            len(preprocessor.numeric_columns) + len(preprocessor.binary_columns) + len(preprocessor.categorical_columns)
        ),
        "genomic_feature_count": 0,
        "input_dim": preprocessor.input_dim,
        "output_feature_names": preprocessor.feature_names,
        "continuous_feature_count": int(encoded_train.continuous_features.shape[1]),
        "auxiliary_feature_count": int(encoded_train.auxiliary_features.shape[1]),
        "rule_feature_count": int(encoded_train.rule_triggers.shape[1]),
    }

    bundle = DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        merged_frame=frame,
        labeled_frame=frame,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        preprocessor=preprocessor,
        split_summary=split_summary,
    )
    setattr(bundle, "encoded_train", encoded_train)
    setattr(bundle, "encoded_val", encoded_val)
    setattr(bundle, "encoded_test", encoded_test)
    return bundle
