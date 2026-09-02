"""Target-rate charts for categorical and numeric features."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

BAR_COLOR = "#D8A7B1"
BAR_EDGE_COLOR = "#B5838D"
BASELINE_COLOR = "#6B7280"


def plot_categorical_target_rate(
    df: pd.DataFrame,
    column: str,
    *,
    target: str = "y",
    title: str | None = None,
    show: bool = True,
) -> tuple[pd.DataFrame, Axes]:
    """Plot ranked positive-target rates with category sample sizes."""
    target_encoded = df[target].eq(1)
    groups = df[column].fillna("Missing")
    plot_data = pd.DataFrame({"group": groups, "target": target_encoded})
    stats = (
        plot_data.groupby("group")
        .agg(
            yes_rate=("target", "mean"),
            yes_count=("target", "sum"),
            count=("target", "size"),
        )
        .sort_values("yes_rate")
    )
    stats.index.name = column
    stats["yes_rate"] *= 100
    stats[["yes_count", "count"]] = stats[["yes_count", "count"]].astype(int)

    fig, ax = plt.subplots(figsize=(10, max(5, len(stats) * 0.45)))
    bars = ax.barh(
        stats.index.astype(str),
        stats["yes_rate"],
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    for bar, rate, yes_count, count in zip(
        bars,
        stats["yes_rate"],
        stats["yes_count"],
        stats["count"],
        strict=True,
    ):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}% · {yes_count}/{count}",
            va="center",
            fontsize=9,
        )

    max_rate = float(stats["yes_rate"].max())
    ax.set_xlabel("Частка клієнтів із депозитом, %")
    ax.set_ylabel("")
    ax.set_xlim(0, max(max_rate * 1.35, 5))
    ax.set_title(
        title or f"Частка депозитів за категоріями: {column}",
        loc="left",
        pad=26,
    )
    ax.text(
        0,
        1.01,
        "Підпис: частка yes · кількість yes / кількість клієнтів",
        transform=ax.transAxes,
        color=BASELINE_COLOR,
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.15)
    sns.despine(ax=ax)
    fig.tight_layout()

    if show:
        plt.show()

    return stats, ax


def plot_numeric_target_rate(
    df: pd.DataFrame,
    column: str,
    *,
    target: str = "y",
    bins: int = 10,
    title: str | None = None,
    show: bool = True,
) -> tuple[pd.DataFrame, Axes]:
    """Plot the positive-target rate across numeric quantile bins."""
    target_encoded = df[target].eq(1)
    groups = pd.qcut(df[column], q=bins, duplicates="drop")
    plot_data = pd.DataFrame({"group": groups, "target": target_encoded})
    stats = plot_data.groupby("group", observed=True).agg(
        yes_rate=("target", "mean"),
        count=("target", "size"),
    )
    stats.index = stats.index.astype(str)

    missing = target_encoded[df[column].isna()]
    if not missing.empty:
        stats.loc["Missing"] = [missing.mean(), missing.size]

    stats.index.name = column
    stats["yes_rate"] *= 100
    stats["count"] = stats["count"].astype(int)

    baseline = float(target_encoded.mean() * 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        stats.index,
        stats["yes_rate"],
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    ax.axhline(
        baseline,
        color=BASELINE_COLOR,
        linestyle="--",
        label=f"Train average: {baseline:.2f}%",
    )

    for bar, rate, count in zip(bars, stats["yes_rate"], stats["count"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{rate:.1f}%\nn={count}",
            ha="center",
        )

    ax.set_xlabel(column)
    ax.set_ylabel("Yes rate, %")
    ax.set_ylim(0, max(stats["yes_rate"].max(), baseline) * 1.2)
    ax.set_title(title or f"Deposit subscription rate by {column}")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    sns.despine(ax=ax)
    fig.tight_layout()

    if show:
        plt.show()

    return stats, ax
