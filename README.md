# Bank Term Deposit Prediction

Binary classification project that predicts whether a bank client will
subscribe to a term deposit (`y = yes`). The positive class is imbalanced, and
the call duration feature is excluded because it is unavailable before the
call ends and would cause target leakage in a real campaign.

## Workflow

- exploratory data analysis and feature hypotheses;
- stratified train, validation, and test split;
- preprocessing of numeric and categorical features;
- comparison of linear models, KNN, decision trees, XGBoost, and LightGBM;
- imbalance experiments with class weights, SMOTENC, and SMOTE-Tomek;
- hyperparameter tuning and model interpretation with feature importance and
  SHAP.

The complete analysis and conclusions are in
[`notebooks/Mid_term_Project.ipynb`](notebooks/Mid_term_Project.ipynb).

## Results

Models were selected by validation ROC-AUC because the campaign needs a useful
ranking of clients before choosing a decision threshold. PR-AUC, recall,
precision, and F1 were monitored because the positive class is relatively
rare. Accuracy was not used for model selection.

The table contains the main validation results. Recall, precision, and F1 use
the default threshold of `0.5`.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| **LightGBM + Hyperopt** | **0.8153** | 0.4613 | 0.6358 | 0.4033 | 0.4935 |
| XGBoost + Hyperopt | 0.8135 | 0.4614 | 0.5647 | 0.4422 | **0.4960** |
| XGBoost | 0.8093 | **0.4637** | 0.6304 | 0.3887 | 0.4809 |
| LightGBM | 0.8062 | 0.4535 | 0.6261 | 0.3881 | 0.4792 |
| Logistic regression (L1) | 0.8028 | 0.4483 | 0.6282 | 0.3635 | 0.4605 |
| Logistic regression | 0.8024 | 0.4480 | 0.6293 | 0.3645 | 0.4617 |
| Tuned decision tree | 0.7944 | 0.4236 | 0.5797 | 0.4277 | 0.4922 |
| KNN | 0.7526 | 0.3563 | 0.2640 | 0.5396 | 0.3546 |

LightGBM tuned with Hyperopt achieved the highest validation ROC-AUC and was
selected as the final model. After retraining on the combined training and
validation data, it produced the following results on the untouched test set:

| ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|
| **0.8170** | **0.4939** | **0.6509** | **0.4054** | **0.4996** |

## Run the project

The project requires Python 3.13 and
[`uv`](https://docs.astral.sh/uv/).

```bash
make setup   # install locked dependencies and Git hooks
make lab     # start JupyterLab
make check   # run linting, type checking, and tests
```

## Structure

```text
data/raw/                              source dataset
notebooks/Mid_term_Project.ipynb       EDA and model experiments
src/bank_term_deposit_prediction/      reusable project code
tests/                                 automated tests
models/                                generated model artifacts
reports/figures/                       exported charts
```
