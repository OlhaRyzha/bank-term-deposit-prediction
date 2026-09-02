"""Feature-importance charts."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from bank_term_deposit_prediction.visualization.target_rate import BAR_COLOR


def plot_feature_importance(
    importance: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Важливість ознак для ROC AUC",
    show: bool = True,
) -> Axes:
    """Plot the most important features from a permutation-importance table."""
    required_columns = {"feature", "importance"}
    missing_columns = required_columns.difference(importance.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing feature-importance columns: {missing}")

    plot_data = importance.nlargest(top_n, "importance").sort_values("importance")
    figure, ax = plt.subplots(figsize=(10, max(5, len(plot_data) * 0.4)))
    sns.barplot(
        data=plot_data,
        x="importance",
        y="feature",
        color=BAR_COLOR,
        ax=ax,
    )
    ax.axvline(0, color="#6B7280", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Зниження ROC AUC після перемішування ознаки")
    ax.set_ylabel("Ознака")
    ax.grid(axis="x", alpha=0.15)
    sns.despine(ax=ax)
    figure.tight_layout()

    if show:
        plt.show()

    return ax
