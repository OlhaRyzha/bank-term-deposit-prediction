from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def get_columns_summary(columns: Iterable[str], data: pd.DataFrame) -> pd.DataFrame:
    """Return unique counts and non-null unique values for selected columns."""
    names = list(columns)
    return pd.DataFrame(
        {
            "column": names,
            "nunique": [data[column].nunique() for column in names],
            "unique_values": [
                data[column].dropna().unique().tolist() for column in names
            ],
        }
    )


def get_category_target_summary(
    data: pd.DataFrame,
    columns: Iterable[str],
    *,
    target: str,
) -> pd.DataFrame:
    """Summarize category frequency and positive-target rate."""
    summaries = []
    for column in columns:
        column_summary = (
            data.groupby(column, dropna=False)[target]
            .agg(count="size", yes_rate="mean")
            .reset_index()
            .rename(columns={column: "category"})
        )
        column_summary.insert(0, "column", column)
        column_summary["share"] = column_summary["count"] / len(data)
        summaries.append(column_summary)

    return pd.concat(summaries, ignore_index=True)


def get_iqr_outlier_summary(
    data: pd.DataFrame,
    columns: Iterable[str],
    *,
    target: str,
) -> pd.DataFrame:
    """Flag potential numeric outliers with IQR and summarize their target rate."""
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        values = data[column].dropna().to_numpy(dtype=np.float64)
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[column].lt(lower_bound) | data[column].gt(upper_bound)

        rows.append(
            {
                "column": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outliers.sum()),
                "outlier_share": float(outliers.mean()),
                "yes_rate_outliers": (
                    float(data.loc[outliers, target].mean()) if outliers.any() else 0.0
                ),
                "skew": _calculate_skew(values),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_share", ascending=False)


def _calculate_skew(values: NDArray[np.float64]) -> float:
    centered = values - values.mean()
    standard_deviation = values.std()
    if standard_deviation == 0:
        return 0.0
    return float(np.mean(centered**3) / standard_deviation**3)
