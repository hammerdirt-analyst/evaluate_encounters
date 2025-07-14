"""
evaluate_encounters.py
Author: Roger Erismann

Purpose:
---------
Runs a series of classification evaluations to estimate the probability of plastic encounters
on Lake Geneva’s beaches, using different targets and data split strategies.

Key Features:
-------------
- Loads observation data and applies four classification scenarios:
    1. Count data split by date
    2. Count data split randomly
    3. Rate data split randomly
    4. Rate data split by date
- Each scenario is configured via a `ClassifierPipeline` and its corresponding config dictionary
- Supports automated evaluation, model selection, and results export

Configuration:
--------------
Each config dictionary defines the classification task, including:
- target column and thresholds (e.g. "quantity" or "pcs/m")
- categorical/numeric features
- data split method ("random" or "date")
- model definitions and evaluation logic
- output destination

Usage:
------
from evaluate_encounters import evaluate_encounters

evaluate_encounters()
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import classifiers as clsf
from logging_config import get_logger
import warnings
from sklearn.exceptions import UndefinedMetricWarning

logger = get_logger("Run script", to_file='logs/classifiers.log')

# the config parameters for each test -
# splitting by date and randomly and using a column for raw counts and one for rates
# the default config is for evaluating raw counts from the observations and splitting
# the data by date
default_config = {
    "task": "classification",
    "target_column": "quantity",
    "columns": ["region"],  # feature columns

    # preprocessing
    "categorical_cols": ["region"],
    "numeric_cols": [],  # can be populated if numeric variables are used

    "summary_column": "region",  # used for summarizing probabilities

    "split": {
        "method": "date",
        "date_column": "date",
        "date_split": "2022-01-01"
    },
    "split_name": "date_split",

    "thresholds": [1, 2, 3],
    "model_defs": clsf.classifiers,
    "model_classes": clsf.model_classes,
    "evaluation_fn": clsf.evaluate_single_model,
    "summary_fn": clsf.summarize_region_probabilities,

    "selection_metric": {
        "method": "mean",
        "columns": ["1", "2", "3"],
        "maximize": True
    },
    "threshold_step": 1,

    "output_dir": "data/test_results"
}

random_split_count_config = {
    "task": "classification",
    "target_column": "quantity",
    "columns": ["region"],  # feature columns

    # preprocessing
    "categorical_cols": ["region"],
    "numeric_cols": [],  # can be populated if numeric variables are used

    "summary_column": "region",  # used for summarizing probabilities

    "split": {
        "method": "random",
        "test_size": 0.2
    },
    "split_name": "random_split",

    "thresholds": [1, 2, 3],
    "model_defs": clsf.classifiers,
    "model_classes": clsf.model_classes,
    "evaluation_fn": clsf.evaluate_single_model,
    "summary_fn": clsf.summarize_region_probabilities,

    "selection_metric": {
        "method": "mean",
        "columns": ["1", "2", "3"],
        "maximize": True
    },
    "threshold_step": 1,

    "output_dir": "data/test_results"
}


random_split_rate_config = {
    "task": "classification",
    "target_column": "pcs/m",
    "columns": ["region"],  # feature columns

    # preprocessing
    "categorical_cols": ["region"],
    "numeric_cols": [],  # can be populated if numeric variables are used

    "summary_column": "region",  # used for summarizing probabilities

    "split": {
        "method": "random",
        "test_size": 0.2
    },
    "split_name": "random_split",

    "thresholds": [0.01, 0.02, 0.03],
    "model_defs": clsf.classifiers,
    "model_classes": clsf.model_classes,
    "evaluation_fn": clsf.evaluate_single_model,
    "summary_fn": clsf.summarize_region_probabilities,

    "selection_metric": {
        "method": "mean",
        "columns": ["0.01", "0.02", "0.03"],
        "maximize": True
    },
    "threshold_step": 0.01,

    "output_dir": "data/test_results"
}

date_split_rate_config = {
    "task": "classification",
    "target_column": "pcs/m",
    "columns": ["region"],  # feature columns

    # preprocessing
    "categorical_cols": ["region"],
    "numeric_cols": [],  # can be populated if numeric variables are used

    "summary_column": "region",  # used for summarizing probabilities

    "split": {
        "method": "date",
        "date_column": "date",
        "date_split": "2022-01-01"
    },
    "split_name": "date_split",

    "thresholds": [0.01, 0.02, 0.03],
    "model_defs": clsf.classifiers,
    "model_classes": clsf.model_classes,
    "evaluation_fn": clsf.evaluate_single_model,
    "summary_fn": clsf.summarize_region_probabilities,

    "selection_metric": {
        "method": "mean",
        "columns": ["0.01", "0.02", "0.03"],
        "maximize": True
    },
    "threshold_step": 0.01,

    "output_dir": "data/test_results"
}
def evaluate_encounters():
    logger.info(f"New test: {default_config['task']}")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    df = clsf.csv_to_dataframe()
    logger.info("Evaluating count data split by date")
    b = clsf.ClassifierPipeline(df, default_config)
    b.run()
    logger.info("Evaluating count data split randomly")
    c = clsf.ClassifierPipeline(df, random_split_count_config)
    c.run()
    logger.info("Evaluating rate data split by random split")
    d = clsf.ClassifierPipeline(df, random_split_rate_config)
    d.run()
    logger.info("Evaluating rate data split by date")
    e = clsf.ClassifierPipeline(df, date_split_rate_config)
    e.run()
    logger.info("End evaluation")

def label_data(df, name):
    df['split'] = name
    return df

def combine(dfs: [pd.DataFrame]):
    new_df = pd.concat(dfs)
    return new_df

def chart_test_results(df: pd.DataFrame, title: str, y_axis, x_axis, filename: str = None):
    """ """
    fig, ax = plt.subplots()
    sns.lineplot(df, ax=ax)
    xticks = ax.get_xticks()
    xtick_labels = [f'> {tick}' for tick in xticks]
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)

    ax.set_title(title)
    plt.show()

