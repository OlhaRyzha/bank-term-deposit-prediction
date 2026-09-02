"""Reusable model pipelines and evaluation helpers."""

from bank_term_deposit_prediction.models.classification import (
    build_decision_tree_pipeline,
    build_lightgbm_pipeline,
    build_logistic_pipeline,
    build_polynomial_logistic_pipeline,
    build_preprocessor,
    build_smotenc_logistic_pipeline,
    build_smotetomek_logistic_pipeline,
    build_xgboost_pipeline,
)
from bank_term_deposit_prediction.models.evaluation import (
    METRIC_DESCRIPTIONS,
    PRIMARY_METRIC,
    evaluate_model,
)
from bank_term_deposit_prediction.models.importance import (
    get_permutation_feature_importance,
)

__all__ = [
    "METRIC_DESCRIPTIONS",
    "PRIMARY_METRIC",
    "build_decision_tree_pipeline",
    "build_lightgbm_pipeline",
    "build_logistic_pipeline",
    "build_polynomial_logistic_pipeline",
    "build_preprocessor",
    "build_smotenc_logistic_pipeline",
    "build_smotetomek_logistic_pipeline",
    "build_xgboost_pipeline",
    "evaluate_model",
    "get_permutation_feature_importance",
]
