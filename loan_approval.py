"""
Loan Approval Prediction
========================
Compares Logistic Regression, Decision Tree, XGBoost, and SVM on a
bank loan dataset. Tracks training time, prediction time, accuracy,
and serialised model size to support an informed model selection decision
beyond raw accuracy alone.

Dataset: loan_lead_data.csv
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb


# ── Data Loading and Preparation ──────────────────────────────────────────────

def load_and_prepare(filepath: str, test_size: float = 0.4, random_state: int = 42):
    """
    Load loan data, one-hot encode categoricals, and produce a
    60/40 train/test split.
    """
    df = pd.read_csv(filepath)
    X = pd.get_dummies(df.drop("Loan Approved", axis=1))
    y = df["Loan Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """Balance the training set with SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:")
    print(pd.Series(y_res).value_counts().to_string())
    return X_res, y_res


# ── Model Benchmarking ────────────────────────────────────────────────────────

def timed_fit(model, X_train, y_train) -> tuple:
    """Fit a model and return (fitted_model, training_time)."""
    start = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - start


def timed_predict(model, X_test) -> tuple:
    """Predict and return (predictions, prediction_time)."""
    start = time.time()
    preds = model.predict(X_test)
    return preds, time.time() - start


def serialised_size_bytes(model) -> int:
    """Return approximate serialised model size in bytes."""
    return sys.getsizeof(pickle.dumps(model))


def benchmark_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train, evaluate, and collect performance metrics for four classifiers.
    Returns a summary DataFrame.
    """
    configs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "XGBoost":             xgb.XGBClassifier(
                                   n_estimators=50,
                                   learning_rate=0.1,
                                   random_state=42,
                                   use_label_encoder=False,
                                   eval_metric="logloss",
                               ),
        "SVM":                 SVC(random_state=42),
    }

    records = []
    for name, clf in configs.items():
        clf, train_time = timed_fit(clf, X_train, y_train)
        preds, pred_time = timed_predict(clf, X_test)
        accuracy = accuracy_score(y_test, preds)
        size_kb = serialised_size_bytes(clf) / 1024

        print(f"\n{name}")
        print(f"  Accuracy        : {accuracy:.4f}")
        print(f"  Train time (s)  : {train_time:.4f}")
        print(f"  Predict time (s): {pred_time:.4f}")
        print(f"  Model size (KB) : {size_kb:.1f}")
        print(classification_report(y_test, preds))

        records.append({
            "Model":            name,
            "Accuracy":         accuracy,
            "Train Time (s)":   train_time,
            "Predict Time (s)": pred_time,
            "Model Size (KB)":  size_kb,
        })

    return pd.DataFrame(records).set_index("Model")


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_benchmark(summary: pd.DataFrame):
    """Bar charts comparing accuracy, training time, and model size."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["Accuracy", "Train Time (s)", "Model Size (KB)"]
    colors  = ["steelblue", "coral", "seagreen"]

    for ax, metric, color in zip(axes, metrics, colors):
        summary[metric].plot(kind="barh", ax=ax, color=color)
        ax.set_title(metric)
        ax.set_xlabel(metric)

    plt.suptitle("Model Comparison – Loan Approval", fontsize=13)
    plt.tight_layout()
    plt.savefig("model_benchmark.png", dpi=150)
    plt.show()
    print("Saved: model_benchmark.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "loan_lead_data.csv"

    X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)
    X_train, y_train = apply_smote(X_train, y_train)

    summary = benchmark_models(X_train, X_test, y_train, y_test)

    print("\nSummary Table:")
    print(summary.to_string())

    plot_benchmark(summary)
