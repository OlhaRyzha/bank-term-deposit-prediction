import pandas as pd
import pytest

from bank_term_deposit_prediction.models import (
    build_decision_tree_pipeline,
    build_lightgbm_pipeline,
    build_logistic_pipeline,
    build_polynomial_logistic_pipeline,
    build_preprocessor,
    build_smotenc_logistic_pipeline,
    build_smotetomek_logistic_pipeline,
    build_xgboost_pipeline,
    evaluate_model,
    get_permutation_feature_importance,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "age": range(40),
            "job": ["admin", "student"] * 20,
        }
    )
    y = pd.Series([0] * 28 + [1] * 12)
    return X, y


@pytest.mark.parametrize(
    "kind",
    [
        "logistic",
        "polynomial",
        "smotenc",
        "smotetomek",
        "decision_tree",
        "xgboost",
        "lightgbm",
    ],
)
def test_model_pipeline_fits_raw_features(
    classification_data: tuple[pd.DataFrame, pd.Series],
    kind: str,
) -> None:
    X, y = classification_data
    preprocessor = build_preprocessor(["age"], ["job"])

    if kind == "logistic":
        model = build_logistic_pipeline(
            preprocessor,
            class_weight="balanced",
        )
    elif kind == "polynomial":
        model = build_polynomial_logistic_pipeline(["age"], ["job"])
    elif kind == "smotenc":
        model = build_smotenc_logistic_pipeline(["age"], ["job"])
    elif kind == "smotetomek":
        model = build_smotetomek_logistic_pipeline(
            preprocessor,
            ovr=True,
        )
    elif kind == "decision_tree":
        model = build_decision_tree_pipeline(["age"], ["job"])
    elif kind == "xgboost":
        model = build_xgboost_pipeline(
            ["age"],
            ["job"],
            n_estimators=10,
        )
    else:
        model = build_lightgbm_pipeline(
            ["age"],
            ["job"],
            n_estimators=10,
        )

    model.fit(X, y)

    assert len(model.predict(X)) == len(X)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_evaluate_model_returns_summary_and_prints_report(
    classification_data: tuple[pd.DataFrame, pd.Series],
    capsys: pytest.CaptureFixture[str],
) -> None:
    X, y = classification_data
    model = build_logistic_pipeline(
        build_preprocessor(["age"], ["job"]),
        class_weight="balanced",
    ).fit(X, y)

    summary = evaluate_model(
        model,
        X,
        y,
        X,
        y,
        model_name="Balanced Logistic Regression",
    )

    assert list(summary.index) == ["Train", "Validation"]
    assert summary.index.name == "Balanced Logistic Regression"
    assert list(summary.columns) == [
        "ROC AUC",
        "PR AUC",
        "Recall",
        "Precision",
        "F1",
        "Accuracy",
    ]
    assert (
        "Balanced Logistic Regression — Validation classification report"
        in capsys.readouterr().out
    )


def test_get_permutation_feature_importance_returns_original_features(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_data
    model = build_logistic_pipeline(
        build_preprocessor(["age"], ["job"]),
        class_weight="balanced",
    ).fit(X, y)

    importance = get_permutation_feature_importance(
        model,
        X,
        y,
        n_repeats=2,
        n_jobs=1,
    )

    assert list(importance.columns) == ["feature", "importance", "std"]
    assert set(importance["feature"]) == set(X.columns)
    assert importance["importance"].is_monotonic_decreasing


def test_build_logistic_pipeline_accepts_elasticnet_parameters() -> None:
    model = build_logistic_pipeline(
        build_preprocessor(["age"], ["job"]),
        class_weight="balanced",
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5_000,
    )
    classifier = model.named_steps["classifier"]

    assert classifier.solver == "saga"
    assert classifier.l1_ratio == 0.5
    assert classifier.C == 0.1
    assert classifier.max_iter == 5_000
