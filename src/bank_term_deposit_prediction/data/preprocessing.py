from collections.abc import Mapping, Sequence
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_term_deposit_prediction.config import RANDOM_STATE


class DataPartition(NamedTuple):
    """Features and encoded target for one dataset partition."""

    features: pd.DataFrame
    target: "pd.Series[int]"


class DatasetSplits(NamedTuple):
    """Train, validation, and test partitions."""

    train: DataPartition
    validation: DataPartition
    test: DataPartition


def prepare_dataset_splits(
    data: pd.DataFrame,
    *,
    target: str,
    target_mapping: Mapping[str, int],
    drop_columns: Sequence[str] = (),
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """Split, encode, and prepare the dataset for modeling."""
    features = data.drop(columns=[target, *drop_columns])
    target_values = data[target]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        target_values,
        test_size=test_size,
        stratify=target_values,
        random_state=random_state,
    )
    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return DatasetSplits(
        train=_prepare_partition(X_train, y_train, target_mapping),
        validation=_prepare_partition(
            X_validation,
            y_validation,
            target_mapping,
        ),
        test=_prepare_partition(X_test, y_test, target_mapping),
    )


def _prepare_partition(
    features: pd.DataFrame,
    target: "pd.Series[str]",
    target_mapping: Mapping[str, int],
) -> DataPartition:
    return DataPartition(
        features=handle_pdays(features),
        target=target.map(target_mapping).astype("int8"),
    )


def handle_pdays(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare `pdays` and add an indicator of its availability."""
    result = df.copy()

    result = result.drop(columns=["pdays_known"], errors="ignore")

    pdays_available = result["pdays"].notna() & result["pdays"].ne(999)
    result["pdays_available"] = pdays_available.astype("int8")
    result["pdays"] = result["pdays"].mask(result["pdays"].eq(999))

    return result


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of numerical columns in the DataFrame."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of categorical columns in the DataFrame."""
    return df.select_dtypes(include="object").columns.tolist()


def get_numerical_and_categorical_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return a tuple of numerical and categorical columns in the DataFrame."""
    return get_numerical_columns(df), get_categorical_columns(df)
