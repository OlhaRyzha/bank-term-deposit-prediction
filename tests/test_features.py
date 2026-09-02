import numpy as np
import pandas as pd

from bank_term_deposit_prediction.features import add_candidate_features


def test_add_candidate_features_preserves_input() -> None:
    data = pd.DataFrame(
        {
            "age": [20, 40, 70],
            "campaign": [1, 3, 9],
            "previous": [0, 2, 0],
        }
    )

    result = add_candidate_features(data)

    assert list(data.columns) == ["age", "campaign", "previous"]
    assert np.allclose(result["campaign_log"], np.log1p(data["campaign"]))
    assert result["has_previous_contact"].tolist() == [0, 1, 0]
    assert result["age_group"].tolist() == ["under_25", "35_49", "65_plus"]
