"""Consistent evaluation for the imbalanced deposit-subscription target."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

PRIMARY_METRIC = "ROC AUC"
METRIC_DESCRIPTIONS = {
    "ROC AUC": "Primary threshold-independent client-ranking metric.",
    "PR AUC": "Positive-class ranking metric used as an imbalance guardrail.",
    "Recall": "Share of actual subscribers found by the model.",
    "Precision": "Share of contacted positive predictions that subscribe.",
    "F1": "Balance between positive-class precision and recall.",
    "Accuracy": "Context only; it can be misleading for the imbalanced target.",
}


class BinaryProbabilisticClassifier(Protocol):
    """Interface required by the project evaluation helper."""

    @property
    def classes_(self) -> Sequence[int]: ...

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]: ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]: ...


def evaluate_model(
    model: BinaryProbabilisticClassifier,
    X_train: pd.DataFrame,
    y_train: "pd.Series[int]",
    X_eval: pd.DataFrame,
    y_eval: "pd.Series[int]",
    *,
    model_name: str = "Model",
    eval_name: str = "Validation",
    positive_label: int = 1,
    digits: int = 4,
) -> pd.DataFrame:
    """Print the evaluation report and return comparable train/eval metrics."""
    train_metrics, _ = _calculate_metrics(
        model,
        X_train,
        y_train,
        positive_label,
    )
    eval_metrics, eval_predictions = _calculate_metrics(
        model,
        X_eval,
        y_eval,
        positive_label,
    )

    print(f"{model_name} — {eval_name} classification report:")
    print(
        classification_report(
            y_eval,
            eval_predictions,
            digits=digits,
            zero_division=0,
        )
    )

    summary = pd.DataFrame(
        [train_metrics, eval_metrics],
        index=["Train", eval_name],
    ).round(digits)
    summary.index.name = model_name
    return summary


def _calculate_metrics(
    model: BinaryProbabilisticClassifier,
    X: pd.DataFrame,
    y: "pd.Series[int]",
    positive_label: int,
) -> tuple[dict[str, float], NDArray[np.int_]]:
    predictions = model.predict(X)
    positive_index = list(model.classes_).index(positive_label)
    probabilities = model.predict_proba(X)[:, positive_index]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predictions,
        average="binary",
        pos_label=positive_label,
        zero_division=0,
    )

    return (
        {
            "ROC AUC": float(roc_auc_score(y, probabilities)),
            "PR AUC": float(
                average_precision_score(
                    y,
                    probabilities,
                    pos_label=positive_label,
                )
            ),
            "Recall": float(recall),
            "Precision": float(precision),
            "F1": float(f1),
            "Accuracy": float(accuracy_score(y, predictions)),
        },
        predictions,
    )
