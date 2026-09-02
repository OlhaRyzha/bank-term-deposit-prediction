import pandas as pd

from bank_term_deposit_prediction.data.preprocessing import (
    handle_pdays,
    prepare_dataset_splits,
)


def test_handle_pdays_replaces_sentinel_and_old_indicator() -> None:
    data = pd.DataFrame(
        {
            "pdays": [999.0, 5.0, None],
            "pdays_known": [1, 0, 1],
        }
    )

    result = handle_pdays(data)

    assert "pdays_known" not in result
    assert result["pdays_available"].tolist() == [0, 1, 0]
    assert result["pdays"].isna().tolist() == [True, False, True]
    assert "pdays_available" not in data


def test_prepare_dataset_splits_returns_model_ready_partitions() -> None:
    data = pd.DataFrame(
        {
            "age": range(100),
            "pdays": [999, 5] * 50,
            "duration": range(100, 200),
            "y": ["no"] * 80 + ["yes"] * 20,
        }
    )

    splits = prepare_dataset_splits(
        data,
        target="y",
        target_mapping={"no": 0, "yes": 1},
        drop_columns=["duration"],
        random_state=42,
    )

    assert len(splits.train.features) == 60
    assert len(splits.validation.features) == 20
    assert len(splits.test.features) == 20

    for partition in splits:
        assert "y" not in partition.features
        assert "duration" not in partition.features
        assert "pdays_available" in partition.features
        assert not partition.features["pdays"].eq(999).any()
        assert set(partition.target.unique()) == {0, 1}
        assert partition.target.index.equals(partition.features.index)

    assert "pdays_available" not in data
    assert data["pdays"].eq(999).any()
