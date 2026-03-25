from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

DEFAULT_TARGET_COLUMN = "mtc_diagnosis"
DEFAULT_ID_COLUMNS = ("source_id", "study_id")
DEFAULT_NUMERIC_COLUMNS = (
    "age",
    "calcitonin_level_numeric",
    "cea_level_numeric",
)
DEFAULT_CATEGORICAL_COLUMNS = (
    "gender",
    "ret_variant",
    "ret_risk_level",
    "calcitonin_elevated",
    "cea_elevated",
    "cea_imputed_flag",
    "thyroid_nodules_present",
    "family_history_mtc",
    "c_cell_disease",
    "men2_syndrome",
    "pheochromocytoma",
    "hyperparathyroidism",
    "age_group",
)


@dataclass
class EncodedFrame:
    categorical: np.ndarray
    numerical: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PreprocessorMetadata:
    target_column: str
    id_columns: list[str]
    dropped_constant_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_mappings: dict[str, dict[str, int]]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    preprocessor: "TabularPreprocessor"
    split_summary: dict[str, Any]


class TabularDataset(Dataset):
    def __init__(self, encoded: EncodedFrame) -> None:
        self.categorical = torch.tensor(encoded.categorical, dtype=torch.long)
        self.numerical = torch.tensor(encoded.numerical, dtype=torch.float32)
        self.target = torch.tensor(encoded.target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.categorical[index], self.numerical[index], self.target[index]


class TabularPreprocessor:
    def __init__(
        self,
        target_column: str = DEFAULT_TARGET_COLUMN,
        id_columns: tuple[str, ...] = DEFAULT_ID_COLUMNS,
        numeric_columns: tuple[str, ...] = DEFAULT_NUMERIC_COLUMNS,
        categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_COLUMNS,
    ) -> None:
        self.target_column = target_column
        self.id_columns = list(id_columns)
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.dropped_constant_columns: list[str] = []
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.categorical_mappings: dict[str, dict[str, int]] = {}

    @property
    def feature_names(self) -> list[str]:
        return self.categorical_columns + self.numeric_columns

    @property
    def categorical_cardinalities(self) -> list[int]:
        return [len(self.categorical_mappings[column]) for column in self.categorical_columns]

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        required_columns = set(self.id_columns + [self.target_column])
        required_columns.update(self.numeric_columns)
        required_columns.update(self.categorical_columns)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        candidate_features = self.numeric_columns + self.categorical_columns
        self.dropped_constant_columns = [
            column for column in candidate_features if frame[column].nunique(dropna=False) <= 1
        ]
        self.numeric_columns = [
            column for column in self.numeric_columns if column not in self.dropped_constant_columns
        ]
        self.categorical_columns = [
            column for column in self.categorical_columns if column not in self.dropped_constant_columns
        ]

        for column in self.numeric_columns:
            series = frame[column].astype(float)
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            self.numeric_means[column] = mean
            self.numeric_stds[column] = std if std > 0 else 1.0

        for column in self.categorical_columns:
            values = frame[column].astype(str).fillna("__MISSING__")
            unique_values = sorted(values.unique().tolist())
            self.categorical_mappings[column] = {
                value: index + 1 for index, value in enumerate(unique_values)
            }

        return self

    def transform(self, frame: pd.DataFrame) -> EncodedFrame:
        if not self.categorical_mappings and not self.numeric_means:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        metadata_columns = [column for column in self.id_columns + [self.target_column] if column in frame]
        metadata = frame[metadata_columns].reset_index(drop=True).copy()
        target = metadata[self.target_column].astype(np.float32).to_numpy()

        categorical_matrices = []
        for column in self.categorical_columns:
            mapping = self.categorical_mappings[column]
            values = (
                frame[column]
                .astype(str)
                .fillna("__MISSING__")
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )
            categorical_matrices.append(values)

        if categorical_matrices:
            categorical = np.stack(categorical_matrices, axis=1)
        else:
            categorical = np.zeros((len(frame), 0), dtype=np.int64)

        numerical_matrices = []
        for column in self.numeric_columns:
            mean = self.numeric_means[column]
            std = self.numeric_stds[column]
            values = ((frame[column].astype(float) - mean) / std).astype(np.float32).to_numpy()
            numerical_matrices.append(values)

        if numerical_matrices:
            numerical = np.stack(numerical_matrices, axis=1)
        else:
            numerical = np.zeros((len(frame), 0), dtype=np.float32)

        return EncodedFrame(
            categorical=categorical,
            numerical=numerical,
            target=target,
            metadata=metadata,
        )

    def metadata(self) -> PreprocessorMetadata:
        return PreprocessorMetadata(
            target_column=self.target_column,
            id_columns=self.id_columns,
            dropped_constant_columns=self.dropped_constant_columns,
            categorical_columns=self.categorical_columns,
            numeric_columns=self.numeric_columns,
            numeric_means=self.numeric_means,
            numeric_stds=self.numeric_stds,
            categorical_mappings=self.categorical_mappings,
        )

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self.metadata()), indent=2), encoding="utf-8")


def _read_frame(data_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    if frame.isna().any().any():
        missing_counts = frame.isna().sum()
        missing_columns = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Dataset contains missing values: {missing_columns}")
    return frame


def _group_aware_split(
    frame: pd.DataFrame,
    target_column: str,
    seed: int,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(frame, frame[target_column], frame["study_id"])
    )
    train_full = frame.iloc[train_indices].reset_index(drop=True)
    test_frame = frame.iloc[test_indices].reset_index(drop=True)

    if train_full[target_column].nunique() < 2 or test_frame[target_column].nunique() < 2:
        raise ValueError("Group-aware split failed to retain both target classes.")

    train_frame, val_frame = train_test_split(
        train_full,
        test_size=validation_size,
        random_state=seed,
        stratify=train_full[target_column],
    )
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame,
    )


def load_data_bundle(
    data_path: str | Path,
    batch_size: int,
    seed: int = 0,
    validation_size: float = 0.2,
) -> DataBundle:
    frame = _read_frame(data_path)
    train_frame, val_frame, test_frame = _group_aware_split(
        frame,
        target_column=DEFAULT_TARGET_COLUMN,
        seed=seed,
        validation_size=validation_size,
    )

    preprocessor = TabularPreprocessor().fit(train_frame)
    encoded_train = preprocessor.transform(train_frame)
    encoded_val = preprocessor.transform(val_frame)
    encoded_test = preprocessor.transform(test_frame)

    train_loader = DataLoader(TabularDataset(encoded_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(encoded_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(encoded_test), batch_size=batch_size, shuffle=False)

    split_summary = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive": int(train_frame[DEFAULT_TARGET_COLUMN].sum()),
        "val_positive": int(val_frame[DEFAULT_TARGET_COLUMN].sum()),
        "test_positive": int(test_frame[DEFAULT_TARGET_COLUMN].sum()),
        "test_studies": sorted(test_frame["study_id"].astype(str).unique().tolist()),
        "feature_names": preprocessor.feature_names,
    }

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        preprocessor=preprocessor,
        split_summary=split_summary,
    )
