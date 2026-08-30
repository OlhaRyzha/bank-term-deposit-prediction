# Bank Term Deposit Prediction

Machine learning project for predicting whether a bank client will subscribe
to a term deposit.

## Setup

```bash
make setup
```

This creates `.venv`, installs the locked dependencies, and enables the
pre-commit and pre-push hooks.

`make setup` creates `.venv` but does not activate it. Project commands use
`uv run`, so activation is optional. To activate the environment manually:

```bash
source .venv/bin/activate
```

## Project structure

```text
.
|-- data/
|   |-- raw/          # original, immutable input data
|   `-- processed/    # generated cleaned and transformed data
|-- models/           # generated model artifacts
|-- notebooks/        # exploration and experiments
|-- reports/
|   `-- figures/      # charts exported for reports
|-- src/              # reusable Python code
|   `-- bank_term_deposit_prediction/
|       |-- data/          # loading and preprocessing
|       |-- features/      # feature engineering
|       |-- models/        # training and prediction
|       |-- evaluation/    # metrics and evaluation
|       |-- pipelines/     # end-to-end workflows
|       `-- visualization/ # reusable plots
|-- tests/            # pytest tests for src/
|-- tools/            # utilities specific to this project only
|-- Makefile
|-- pyproject.toml
`-- uv.lock
```

Do not modify files in `data/raw/` from notebooks. Write transformed datasets
to `data/processed/`. Reusable logic should move from notebooks to `src/` and
receive a matching test in `tests/`.

Keep only project-specific utilities in `tools/`. Reusable editor extensions
and developer tools should live separately and be installed globally.

## Common commands

```bash
make sync     # sync dependencies after pyproject.toml changes
make format   # autofix lint issues and format code/notebooks
make lint     # check lint and formatting
make test     # run tests when tests/ contains test_*.py files
make check    # run lint, mypy, and tests
make hooks    # run all Git hooks manually
```

Add a runtime dependency with `uv add <package>` and a development dependency
with `uv add --dev <package>`.

## Optional VS Code notebook tool

`Notebook Import Runner` is a reusable VS Code extension and should be installed
globally rather than copied into every project. Build it into a `.vsix` from its
own source directory, then install it with:

```bash
code --install-extension notebook-import-runner.vsix
```

After that, it is available in every VS Code project for the current user.
