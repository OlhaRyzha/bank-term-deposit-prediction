"""Model-agnostic feature importance for fitted classification pipelines."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from bank_term_deposit_prediction.config import RANDOM_STATE
from bank_term_deposit_prediction.models.evaluation import (
    BinaryProbabilisticClassifier,
)


def get_permutation_feature_importance(
    model: BinaryProbabilisticClassifier,
    X: pd.DataFrame,
    y: "pd.Series[int]",
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Return the validation-score decrease caused by shuffling each feature."""
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": np.asarray(result.importances_mean, dtype=np.float64),
                "std": np.asarray(result.importances_std, dtype=np.float64),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
