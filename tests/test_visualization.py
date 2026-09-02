import matplotlib.pyplot as plt
import pandas as pd
import pytest

from bank_term_deposit_prediction.visualization import (
    plot_categorical_target_rate,
    plot_feature_importance,
    plot_numeric_target_rate,
)


def test_plot_categorical_target_rate_summarizes_categories() -> None:
    df = pd.DataFrame(
        {
            "job": ["admin", "admin", "student", "student"],
            "y": [0, 1, 1, 1],
        }
    )

    stats, ax = plot_categorical_target_rate(df, "job", show=False)

    assert stats.loc["admin", "yes_rate"] == pytest.approx(50.0)
    assert stats.loc["student", "yes_rate"] == pytest.approx(100.0)
    assert stats.loc["student", "yes_count"] == 2
    assert stats["count"].sum() == len(df)
    assert ax.get_xlabel() == "Частка клієнтів із депозитом, %"
    assert ax.get_legend() is None
    assert len(ax.lines) == 0
    assert any("2/2" in text.get_text() for text in ax.texts)
    plt.close(ax.figure)


def test_plot_numeric_target_rate_bins_values_and_keeps_missing_group() -> None:
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, None],
            "y": [0, 0, 1, 1, 1],
        }
    )

    stats, ax = plot_numeric_target_rate(
        df,
        "age",
        bins=2,
        show=False,
    )

    assert stats["count"].sum() == len(df)
    assert stats.loc["Missing", "yes_rate"] == pytest.approx(100.0)
    assert ax.get_ylabel() == "Yes rate, %"
    plt.close(ax.figure)


def test_plot_feature_importance_limits_number_of_features() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["age", "job", "campaign"],
            "importance": [0.3, 0.2, 0.1],
            "std": [0.01, 0.02, 0.03],
        }
    )

    ax = plot_feature_importance(importance, top_n=2, show=False)

    assert len(ax.patches) == 2
    assert ax.get_xlabel() == "Зниження ROC AUC після перемішування ознаки"
    plt.close(ax.figure)
