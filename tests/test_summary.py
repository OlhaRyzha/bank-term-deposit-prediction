import pandas as pd

from bank_term_deposit_prediction.data.summary import (
    get_category_target_summary,
    get_columns_summary,
    get_iqr_outlier_summary,
)


def test_get_columns_summary_excludes_missing_unique_values() -> None:
    data = pd.DataFrame({"job": ["admin", "admin", None, "student"]})

    result = get_columns_summary(["job"], data).iloc[0]

    assert result["column"] == "job"
    assert result["nunique"] == 2
    assert result["unique_values"] == ["admin", "student"]


def test_get_category_target_summary_calculates_rates_and_shares() -> None:
    data = pd.DataFrame({"job": ["admin", "admin", "student"], "y": [0, 1, 1]})

    result = get_category_target_summary(data, ["job"], target="y")

    admin = result.loc[result["category"].eq("admin")].iloc[0]
    assert admin["count"] == 2
    assert admin["yes_rate"] == 0.5
    assert admin["share"] == 2 / 3


def test_get_iqr_outlier_summary_flags_extreme_value() -> None:
    data = pd.DataFrame({"age": [20, 21, 22, 23, 100], "y": [0, 0, 0, 0, 1]})

    result = get_iqr_outlier_summary(data, ["age"], target="y").iloc[0]

    assert result["outlier_count"] == 1
    assert result["yes_rate_outliers"] == 1.0
