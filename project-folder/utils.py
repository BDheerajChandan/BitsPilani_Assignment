# utils.py

import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

META_PATH = "model/meta.json"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


def load_meta(path=META_PATH):
    with open(path) as f:
        return json.load(f)


def prepare(df, meta):
    required = meta["features"] + [meta["target"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df.dropna(subset=[meta["target"]])
    X = df[meta["features"]]
    y = (df[meta["target"]].astype(str) == meta["positive"]).astype(int)
    return X, y


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    pred = model.predict(X)
    score = get_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred