"""Candidate domain features for model experiments."""

import numpy as np
import pandas as pd


def add_candidate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add nonlinear and contact-history features suggested by EDA."""
    result = data.copy()
    result["campaign_log"] = np.log1p(result["campaign"])
    result["has_previous_contact"] = result["previous"].gt(0).astype("int8")
    result["age_group"] = pd.cut(
        result["age"],
        bins=[0, 25, 35, 50, 65, np.inf],
        labels=["under_25", "25_34", "35_49", "50_64", "65_plus"],
    ).astype("object")
    return result
