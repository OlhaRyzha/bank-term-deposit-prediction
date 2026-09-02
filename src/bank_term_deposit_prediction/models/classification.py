"""Factories for the classification pipelines used in the project."""

from collections.abc import Sequence
from typing import Literal

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from bank_term_deposit_prediction.config import RANDOM_STATE

ClassWeight = Literal["balanced"] | dict[int, float] | None
Solver = Literal[
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
]


def build_preprocessor(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    impute: bool = True,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build numeric scaling and categorical one-hot encoding."""
    numeric_steps = []
    categorical_steps = []

    if impute:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), list(numerical_columns)),
            ("cat", Pipeline(categorical_steps), list(categorical_columns)),
        ]
    )


def build_logistic_pipeline(
    preprocessor: ColumnTransformer,
    *,
    class_weight: ClassWeight = None,
    ovr: bool = False,
    solver: Solver = "lbfgs",
    l1_ratio: float = 0.0,
    C: float = 1.0,
    max_iter: int = 2_000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a logistic regression pipeline for the original training data."""
    return Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                _build_classifier(
                    class_weight=class_weight,
                    ovr=ovr,
                    solver=solver,
                    l1_ratio=l1_ratio,
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_decision_tree_pipeline(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    class_weight: ClassWeight = "balanced",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a decision-tree pipeline for the original feature columns."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numerical_columns,
                    categorical_columns,
                    scale_numeric=False,
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    min_child_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build an XGBoost pipeline for the original feature columns."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numerical_columns,
                    categorical_columns,
                    scale_numeric=False,
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a LightGBM pipeline for the original feature columns."""
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numerical_columns,
                    categorical_columns,
                    scale_numeric=False,
                ),
            ),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    subsample_freq=1,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary",
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_polynomial_logistic_pipeline(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    degree: int = 2,
    class_weight: ClassWeight = "balanced",
    max_iter: int = 2_000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build logistic regression with numeric polynomial features."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "polynomial",
                            PolynomialFeatures(degree=degree, include_bias=False),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numerical_columns),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_columns),
            ),
        ]
    )
    return build_logistic_pipeline(
        preprocessor,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )


def build_smotenc_logistic_pipeline(
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    sampling_strategy: float | Literal["auto"] = "auto",
    ovr: bool = False,
    max_iter: int = 2_000,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Build an imputation, SMOTENC, encoding, and logistic pipeline."""
    numerical_columns = list(numerical_columns)
    categorical_columns = list(categorical_columns)

    imputer = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numerical_columns),
            (
                "cat",
                SimpleImputer(strategy="most_frequent"),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return ImbPipeline(
        [
            ("imputer", imputer),
            (
                "sampler",
                SMOTENC(
                    categorical_features=categorical_columns,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(
                    numerical_columns,
                    categorical_columns,
                    impute=False,
                ),
            ),
            (
                "classifier",
                _build_classifier(
                    ovr=ovr,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_smotetomek_logistic_pipeline(
    preprocessor: ColumnTransformer,
    *,
    sampling_strategy: float | Literal["auto"] = "auto",
    ovr: bool = False,
    max_iter: int = 2_000,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Build a preprocessing, SMOTE-Tomek, and logistic pipeline."""
    return ImbPipeline(
        [
            ("preprocessor", clone(preprocessor)),
            (
                "sampler",
                SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                ),
            ),
            (
                "classifier",
                _build_classifier(
                    ovr=ovr,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_classifier(
    *,
    max_iter: int,
    random_state: int,
    class_weight: ClassWeight = None,
    ovr: bool = False,
    solver: Solver = "lbfgs",
    l1_ratio: float = 0.0,
    C: float = 1.0,
) -> LogisticRegression | OneVsRestClassifier:
    classifier = LogisticRegression(
        class_weight=class_weight,
        solver=solver,
        l1_ratio=l1_ratio,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )
    return OneVsRestClassifier(classifier) if ovr else classifier
