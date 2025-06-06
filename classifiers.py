# cumulative_classification.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from logging_config import get_logger
from error_utilities import handle_errors

logger = get_logger(__name__)

model_defs = {
    "logistic": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "param_grid": {"clf__C": [0.01, 0.1, 1, 10]}
    },
    "naive_bayes": {
        "model": MultinomialNB(),
        "param_grid": {"clf__alpha": [0.1, 1.0, 5.0]}
    },
    "random_forest": {
        "model": RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
        "param_grid": {
            "clf__n_estimators": [100],
            "clf__max_depth": [4, 8, None]
        }
    }
}

preprocessor = ColumnTransformer([
    ("region", OneHotEncoder(handle_unknown="ignore"), ["region"])
])

@handle_errors("tuning cumulative classifiers", "Check the input data and model configurations.", logger=logger)
def tune_models(train_df):
    tuning_thresholds = [1, 2, 3]
    best_models = {}

    for name, spec in model_defs.items():
        logger.info(f"Tuning model: {name}")
        auc_scores = []

        for n in tuning_thresholds:
            y_train_bin = (train_df["quantity"] >= n).astype(int)

            pipe = Pipeline([
                ("pre", preprocessor),
                ("clf", spec["model"])
            ])

            grid = GridSearchCV(pipe, spec["param_grid"], cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(train_df[["region"]], y_train_bin)

            best_auc = grid.best_score_
            best_params = grid.best_params_
            auc_scores.append((n, best_auc, best_params))
            logger.info(f"Threshold {n}, Model {name}, AUC: {best_auc:.3f}, Params: {best_params}")

        best_params_list = [x[2] for x in auc_scores]
        common_params = max(set(best_params_list), key=best_params_list.count)
        best_models[name] = {
            "model_class": spec["model"].__class__,
            "params": common_params,
            "auc_scores": auc_scores
        }

    return best_models

@handle_errors("evaluating cumulative classifiers", "Ensure best models and thresholds are correctly defined.", logger=logger)
def evaluate_single_model(train_df, test_df, model_class, params):
    results = {}
    max_quantity = train_df["quantity"].max()

    model_instance = model_class(**params)

    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", model_instance)
    ])

    # Fit model once
    y_train_bin = (train_df["quantity"] >= 1).astype(int)
    pipe.fit(train_df[["region"]], y_train_bin)

    for n in range(1, max_quantity + 1):
        y_test_bin = (test_df["quantity"] >= n).astype(int)
        probas = pipe.predict_proba(test_df[["region"]])[:, 1]

        results[n] = {
            "probas": probas,
            "auc": roc_auc_score(y_test_bin, probas),
            "brier": brier_score_loss(y_test_bin, probas),
            "accuracy": accuracy_score(y_test_bin, probas >= 0.5)
        }

    return results
