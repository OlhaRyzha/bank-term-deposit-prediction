import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_correlation_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: list[str],
    target_name: str,
) -> None:
    heatmap_df = X[num_cols].copy()
    heatmap_df[target_name] = y.reindex(X.index)

    correlation_matrix = heatmap_df.corr()

    mask = np.triu(
        np.ones_like(correlation_matrix, dtype=bool),
        k=1,
    )

    cmap = sns.diverging_palette(
        250,
        10,
        s=55,
        l=70,
        as_cmap=True,
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Correlation between numerical features")
    plt.tight_layout()
    plt.show()
