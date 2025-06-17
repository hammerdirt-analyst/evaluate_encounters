"""
classifiers.py
Author: Roger Erismann

Purpose:
---------
General-purpose classification framework for evaluating exceedance probabilities over numerical thresholds.
Originally designed for modeling plastic shotgun-wadding encounters on Lake Geneva beaches, but adaptable to any cumulative classification task.

Key Features:
-------------
- Binary classification to estimate P(target ≥ threshold) across a configurable range
- Flexible model tuning and comparison using any scikit-learn compatible classifiers
- Threshold-based evaluation and performance summary
- Region-level (or group-based) exceedance summaries
- Automated pipeline for data splitting, model selection, full evaluation, and result export

ClassifierPipeline:
-------------------
A configurable pipeline that supports:
1. Splitting data by date or randomly
2. Tuning multiple models using user-defined thresholds
3. Selecting the best model based on aggregated performance metrics
4. Evaluating the best model over a full threshold range
5. Saving all outputs (.csv and .json) using informative filenames

Configuration Dictionary:
-------------------------
Controls all pipeline behavior. Example keys:

- "task": str — must be "classification"
- "target_column": str — name of the column to threshold (e.g., "quantity", "rate", etc.)
- "columns": list[str] — list of input feature columns
- "categorical_cols": list[str] — subset of columns treated as categorical
- "numeric_cols": list[str] — subset of columns treated as numeric
- "summary_column": str — used to group output summaries (e.g., "region")
- "split": dict — how to divide the data, e.g.:
    {
        "method": "date" or "random",
        "date_column": "date",
        "date_split": "2022-01-01"
    }
- "split_name": str — label used in output filenames
- "thresholds": list[float|int] — tuning thresholds
- "threshold_step": float — step size for evaluation thresholds
- "model_defs": dict — dictionary of model specifications, e.g.:
    {
        "ModelName": {
            "model": sklearn-compatible model instance,
            "param_grid": dict for GridSearchCV
        },
        ...
    }
- "model_classes": dict — mapping model names to their constructors (for evaluation)
- "selection_metric": dict — how to choose the best model, e.g.:
    {
        "method": "mean",
        "columns": ["1", "2", "3"],
        "maximize": True
    }
- "output_dir": str — where to save .csv/.json results

Core Functions:
---------------
- tune_models(...) → grid search on multiple models and thresholds
- evaluate_single_model(...) → performance evaluation over full range of thresholds
- summarize_region_probabilities(...) → weighted summary by a grouping variable
- tuning_results_to_dataframe(...) → transforms tuning results into comparison table

Dependencies:
-------------
- Local: logging_config.get_logger, error_utilities.handle_errors
- External: pandas, numpy, sklearn, xgboost

Supported Models:
-----------------
- Any classifier supported by scikit-learn, provided with proper config for tuning and evaluation

Usage:
------
from classifiers import csv_to_dataframe, ClassifierPipeline, default_config

df = csv_to_dataframe(<file path>)
pipeline = ClassifierPipeline(df, <a config dictionary>)
pipeline.run()
"""

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from xgboost import XGBClassifier
from logging_config import get_logger
from error_utilities import handle_errors

logger = get_logger(__name__, to_file='logs/classifiers.log')
logger.info(f"Saving logs to : 'logs/classifiers.log'")

model_classes = {
    LogisticRegression.__name__: LogisticRegression,
    RandomForestClassifier.__name__: RandomForestClassifier,
    XGBClassifier.__name__: XGBClassifier,
    MultinomialNB.__name__: MultinomialNB
}

classifiers = {
    LogisticRegression.__name__: {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "param_grid": {"clf__C": [0.01, 0.1, 1, 10]}
    },
    MultinomialNB.__name__: {
        "model": MultinomialNB(),
        "param_grid": {"clf__alpha": [0.1, 1.0, 5.0]}
    },
    RandomForestClassifier.__name__: {
        "model": RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
        "param_grid": {
            "clf__n_estimators": [100],
            "clf__max_depth": [4, 8, None]
        }
    },
    XGBClassifier.__name__: {
        "model": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0
        ),
        "param_grid": {
            "clf__n_estimators": [100],
            "clf__max_depth": [3, 6],
            "clf__scale_pos_weight": [1, 2]
        }
    }
}

def create_preprocessor(categorical_cols, numeric_cols):
    transformers = []
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    return ColumnTransformer(transformers)

@handle_errors("loading csv", "verfiy the location of the .csv file")
def csv_to_dataframe(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = "data/g70_obs_2015_2022.csv"
    logger.info(f"Reading {csv_path}")
    return pd.read_csv(csv_path)
@handle_errors("splitting data", "there are two methods 'date' and 'random', ensure you have supplied the appropriate arguments")
def split_data(df, method="date", *, date_column="date", date_split="2022-01-01", test_size=0.2, random_state=42):
    if method == "date":
        df[date_column] = pd.to_datetime(df[date_column])
        train_df = df[df[date_column] < date_split].copy()
        test_df = df[df[date_column] >= date_split].copy()
    elif method == "random":
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("Unsupported split method. Choose 'date' or 'random'.")
    return train_df, test_df
@handle_errors("summarizing a data frame", "did you supply a pands df, is there a pcs/m column")
def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Summary of data frame: {df.shape}")
    summary_data = {
        "count": len(df),
        "nlocations": df.location.nunique(),
        "quantity": df.quantity.sum(),
        "mean": df['pcs/m'].mean(),
        "std": df['pcs/m'].std(),
        "quantiles": np.quantile(df['pcs/m'], [.05, .25, .50, .75, .95]),
        "greater_than_zero": (df.quantity > 0).sum(),
        "rate": (df.quantity > 0).sum() / len(df),
    }
    return summary_data

@handle_errors("converting model evaluations", "the input for this model should be a tune_models object")
def tuning_results_to_dataframe(results: {}) -> pd.DataFrame:
    logger.info("Making table of model evaluations")
    records = []
    for model_name, details in results.items():
        row = {"model": model_name}
        for threshold, auc, _ in details["auc_scores"]:
            row[f"{threshold}"] = auc
        records.append(row)
    return pd.DataFrame(records)

@handle_errors("Tuning models with grid search", "ensure the correct parameters are given for each model")
def tune_models(df, model_defs: dict = None, tuning_thresholds: list = None, column_name: str = 'quantity', split_args: dict = None, categorical_cols: list = None, numeric_cols: list = None) -> (dict, pd.DataFrame):
    logger.info("Tuning models for classification.")
    if model_defs is None:
        model_defs = classifiers
    train_df, _ = split_data(df, **split_args) if split_args else (df, None)
    preprocessor = create_preprocessor(categorical_cols, numeric_cols)
    best_models = {}
    results_table = []

    for name, spec in model_defs.items():
        logger.info(f"Tuning model: {name}")
        pipe = Pipeline([("pre", preprocessor), ("clf", spec["model"])])
        auc_scores = []
        for t in tuning_thresholds:
            y_train_bin = (train_df[column_name] >= t).astype(int)
            grid = GridSearchCV(pipe, spec["param_grid"], cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(train_df[categorical_cols + numeric_cols], y_train_bin)
            auc_scores.append((t, grid.best_score_, grid.best_params_))
            logger.info(f"Threshold {t}, AUC: {grid.best_score_:.3f}, Params: {grid.best_params_}")
        most_common = Counter([frozenset(p.items()) for _, _, p in auc_scores]).most_common(1)[0][0]
        best_models[name] = {
            "model_class": spec["model"].__class__,
            "model": model_classes[name],
            "params": dict(most_common),
            "auc_scores": auc_scores
        }
        row = {"model": name}
        row.update({f"{t}": a for t, a, _ in auc_scores})
        results_table.append(row)

    return best_models, pd.DataFrame(results_table)

@handle_errors("evaluating cumulative classifiers", "Ensure best models and thresholds are correctly defined.")
def evaluate_single_model(df: pd.DataFrame, model_class: callable, params: {}, target_column: str = "quantity", thresholds: list = None, split_args: {} = None, categorical_cols: list = None, numeric_cols: list = None) -> {}:
    logger.info(f"Evaluating model: {model_class}")
    results = {}
    if thresholds is None:
        thresholds = list(range(1, 11)) if target_column == "quantity" else []
    train_df, test_df = split_data(df, **split_args) if split_args else (df, df)
    preprocessor = create_preprocessor(categorical_cols, numeric_cols)
    for threshold in thresholds:
        y_train_bin = (train_df[target_column] >= threshold).astype(int)
        y_test_bin = (test_df[target_column] >= threshold).astype(int)
        if y_train_bin.nunique() < 2 or y_test_bin.nunique() < 2:
            logger.warning(f"Threshold {threshold}: not enough class variation.")
            results[threshold] = {"probas": np.full(len(test_df), 0.0), "auc": np.nan, "brier": np.nan, "accuracy": np.nan}
            continue
        pipe = Pipeline([("pre", preprocessor), ("clf", model_class(**params))])
        pipe.fit(train_df[categorical_cols + numeric_cols], y_train_bin)
        probas = pipe.predict_proba(test_df[categorical_cols + numeric_cols])[:, 1]
        results[threshold] = {
            "probas": probas,
            "auc": roc_auc_score(y_test_bin, probas),
            "brier": brier_score_loss(y_test_bin, probas),
            "accuracy": accuracy_score(y_test_bin, probas >= 0.5)
        }
    return results, train_df, test_df

@handle_errors("summarizing region-level probabilities", "Verify the test data and predictions are aligned.")
def summarize_region_probabilities(results, test_df, column_name: str = "region") -> pd.DataFrame:
    logger.info("Summarizing region-level probabilities")
    summary = []
    for threshold, result in results.items():
        temp_df = test_df.copy()
        temp_df[str(threshold)] = result["probas"]
        region_means = temp_df.groupby(column_name)[str(threshold)].mean()
        region_weights = temp_df[column_name].value_counts(normalize=True)
        weighted_avg = (region_means * region_weights).sum()
        row = {
            "threshold": threshold,
            "weighted_mean_probability": weighted_avg,
            "auc": result["auc"],
            "accuracy": result["accuracy"]
        }
        row.update(region_means.to_dict())
        summary.append(row)
    return pd.DataFrame(summary)

class ClassifierPipeline:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.split_args = config.get("split", {})
        self.columns = config.get("columns", [])
        self.categorical_cols = config.get("categorical_cols", [])
        self.numeric_cols = config.get("numeric_cols", [])
        self.model_defs = config.get("model_defs")
        self.model_classes = config.get("model_classes")
        self.thresholds = config.get("thresholds", [])
        self.target_column = config.get("target_column", "quantity")
        self.output_dir = config.get("output_dir", "results")
        self.summary_column = config.get("summary_column", "region")
        self.selection_metric = config.get("selection_metric", {
            "method": "mean", "columns": self.thresholds, "maximize": True
        })
        self.threshold_step = config.get("threshold_step", [])

    def run(self):
        logger.info(f'Running classifier pipeline for : {self.model_defs.keys()}')
        max_val = self.df[self.target_column].max()
        self.full_thresholds = np.arange(
            self.threshold_step,
            max_val + self.threshold_step,
            self.threshold_step
        )
        logger.info(f'Tuning models on thresholds: {self.thresholds}, target column: {self.target_column} and split by: {self.split_args}')
        logger.info(f'Full evaluation on the range of {self.full_thresholds[0]} - {self.full_thresholds[-1]}')
        self.models, self.tune_summary = tune_models(
            self.df,
            model_defs=self.model_defs,
            tuning_thresholds=self.thresholds,
            column_name=self.target_column,
            split_args=self.split_args,
            categorical_cols=self.categorical_cols,
            numeric_cols=self.numeric_cols
        )
        self._save(self.tune_summary, "tuning_summary")

        logger.info(f'Summary results for tuning:\n{self.tune_summary.to_markdown()}')
        logger.info(f'Saving results to {self.output_dir}')
        logger.info('Selecting best model')

        self.best_model_row = self._select_best_model(self.tune_summary)
        model_name = self.best_model_row.model.values[0]
        model_params = self.models[model_name]['params']
        model_params = {k.split("__")[-1]: v for k, v in model_params.items()}
        logger.info(f"Selected model: {model_name} with params {model_params}")
        logger.info("Applying model to each threshold")

        self.predictions, self.train, self.test = evaluate_single_model(
            self.df,
            model_class=self.model_classes[model_name],
            params=model_params,
            target_column=self.target_column,
            thresholds=self.full_thresholds,
            split_args=self.split_args,
            categorical_cols=self.categorical_cols,
            numeric_cols=self.numeric_cols
        )

        self.summary = summarize_region_probabilities(self.predictions, self.test, column_name=self.summary_column).dropna()
        logger.info(f'The first rows of the regional predictions dataframe, number of rows = {len(self.summary)}\n {self.summary.to_markdown()}')
        self._save(self.summary, "summary")

        probas_df = pd.DataFrame({
            str(threshold): self.predictions[threshold]['probas']
            for threshold in self.predictions
        }, index=self.test.index)

        # Combine with self.test in one operation
        self.test = pd.concat([self.test, probas_df], axis=1)

        self._save(self.test, "test_predictions")

    def _select_best_model(self, df):
        method = self.selection_metric["method"]
        columns = self.selection_metric["columns"]
        maximize = self.selection_metric["maximize"]

        df['score'] = df[columns].astype(float).mean(axis=1)
        best = df.loc[df['score'].idxmax()] if maximize else df.loc[df['score'].idxmin()]
        tied = df[df['score'] == best['score']].copy()
        preferred = [name for name in self.model_defs if name in tied.model.values]
        return tied[tied.model == preferred[0]] if preferred else tied.iloc[[0]]

    def _sanitize(self, value):
        return re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_')

    def _save(self, df, name):
        split_label = self.config.get("split_name", "split")
        target_label = self._sanitize(self.target_column)
        base_name = f"{split_label}_{target_label}_{name}"
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        # for JSON output
        # json_path = os.path.join(self.output_dir, f"{base_name}.json")
        # df.to_json(json_path, orient="records", lines=False, indent=2)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {name} to {csv_path}.")

