# 🧪 Classification Pipeline for Exceedance Analysis

**Author:** Roger Erismann
**Purpose:** Estimate and evaluate exceedance probabilities using flexible, configurable classification models. Originally developed to model plastic shotgun-wadding encounters on Lake Geneva's beaches, this pipeline supports any binary classification task involving thresholds.

## 🚀 Overview

This repository provides a modular, extensible framework for evaluating the probability of a numeric target variable exceeding various thresholds (e.g., `P(quantity ≥ X)`).

### Core Capabilities:

* Binary classification over cumulative thresholds
* Model tuning and comparison using any scikit-learn-compatible classifier
* Evaluation across full threshold ranges
* Region-level or group-based summarization
* Automated saving of results in `.csv` and `.json` formats

## 📁 Modules

### `classifiers.py`

Contains the core logic and `ClassifierPipeline` class which handles:

* Data preprocessing
* Model tuning
* Best-model selection
* Threshold-based evaluation
* Output summarization and export

The pipeline is driven by a configuration dictionary. See **Usage** below.

## ⚙️ Configuration Example

The pipeline is configured using a dictionary. Here's a simplified example:

```python
config = {
    "task": "classification",
    "target_column": "quantity",
    "columns": ["region"],
    "categorical_cols": ["region"],
    "numeric_cols": [],
    "summary_column": "region",
    "split": {
        "method": "date",
        "date_column": "date",
        "date_split": "2022-01-01"
    },
    "split_name": "date_split",
    "thresholds": [1, 2, 3],
    "threshold_step": 1.0,
    "model_defs": classifiers,
    "model_classes": model_classes,
    "selection_metric": {
        "method": "mean",
        "columns": ["1", "2", "3"],
        "maximize": True
    },
    "output_dir": "data/test_results"
}
```



## 💠 Usage

### Running a Custom Task

```python
from classifiers import csv_to_dataframe, ClassifierPipeline
from my_config import config  # Define your config

df = csv_to_dataframe("path/to/data.csv")
pipeline = ClassifierPipeline(df, config)
pipeline.run()
```
### Running All Predefined Evaluations

#### `evaluate_encounters.py`

Defines and runs multiple classification tasks using different targets and split strategies (e.g., by date or randomly). It demonstrates:

* How to build and run multiple `ClassifierPipeline` configurations
* How to evaluate models on both raw count and rate targets

The original use case for this pipeline

```python
from evaluate_encounters import evaluate_encounters

evaluate_encounters()
```

## 📄 Output

The pipeline produces:

* `*_summary.csv` and `.json` — Threshold performance summaries
* `*_test_predictions.csv` — Full prediction outputs with probabilities
* Logs in `logs/classifiers.log` (configurable)

All filenames include:

* the split strategy (`split_name`)
* the target column
* the output type (e.g. `summary`, `tuning_summary`)

## 📦 Dependencies

* Python 3.10+
* pandas
* numpy
* scikit-learn
* xgboost

## Utilities

### `error_utilities.py`

Provides utilities for error handling:

* `handle_error`: Logs and returns structured error messages
* `handle_errors`: A decorator to apply error handling consistently across pipeline functions

### `logging_config.py`

Initializes and configures loggers for clean, centralized logging across all modules:

* Supports both file and console logging
* Prevents duplicate handlers
* Automatically ensures log directories exist

contact: roger@hammerdirt.ch
