"""Metrics and plots for binary classification models."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, RocCurveDisplay,
)

from src.config import FIGURES_DIR, METRICS_DIR


def calculate_metrics(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def save_evaluation_artifacts(results: dict[str, dict], best_name: str, best_model: Any,
                              X_test: pd.DataFrame, y_test: pd.Series) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"selection_metric": "roc_auc", "best_model": best_name,
                   "models": results}, handle, indent=2)
    pd.DataFrame.from_dict(results, orient="index").rename_axis("model").to_csv(
        METRICS_DIR.parent / "model_comparison.csv"
    )

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title=f"Confusion Matrix: {best_name}", xlabel="Predicted", ylabel="Actual",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["No disease", "Disease"],
           yticklabels=["No disease", "Disease"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    comparison = pd.DataFrame.from_dict(results, orient="index")
    ax = comparison["roc_auc"].plot.bar(color=["#3977a8", "#d6843c"], figsize=(6, 4))
    ax.set(title="Model Comparison by ROC-AUC", xlabel="Model", ylabel="ROC-AUC", ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # A model-specific, model-level explanation plot for the selected estimator.
    if best_name == "Random Forest":
        preprocessor = best_model.named_steps["preprocessor"]
        classifier = best_model.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        importance = pd.Series(classifier.feature_importances_, index=names).sort_values().tail(15)
        title, xlabel = "Random Forest Feature Importance", "Importance"
    else:
        preprocessor = best_model.named_steps["preprocessor"]
        classifier = best_model.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        importance = pd.Series(abs(classifier.coef_[0]), index=names).sort_values().tail(15)
        title, xlabel = "Logistic Regression Coefficient Magnitude", "Absolute coefficient"
    ax = importance.plot.barh(figsize=(8, 6), color="#3977a8")
    ax.set(title=title, xlabel=xlabel, ylabel="Transformed feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
