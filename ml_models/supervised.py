"""Train and evaluate a supervised XGBoost classifier for top-product prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "is_top_product"
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "product_id",
    "name",
    "description",
    "category",
    "short_summary",
    "extracted_tags",
    "variants",
)


def configure_logging() -> None:
    """Configure baseline logging for supervised training execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_project_root() -> Path:
    """Return the project root path from the current file location."""
    return Path(__file__).resolve().parents[1]


def get_paths(project_root: Path) -> tuple[Path, Path, Path]:
    """Return input and output artifact paths for supervised training."""
    input_path = project_root / "data" / "processed" / "ml_ready_data.csv"
    artifacts_dir = project_root / "ml_models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "xgb_model.json"
    metrics_path = artifacts_dir / "supervised_metrics.json"
    return input_path, model_path, metrics_path


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Load ML-ready CSV dataset and validate basic requirements."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ML-ready dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("ML-ready dataset is empty.")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    return dataframe


def build_feature_matrix(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric feature matrix and target vector for model training."""
    y = dataframe[TARGET_COLUMN].astype(int)

    features = dataframe.drop(columns=[column for column in [TARGET_COLUMN, *EXCLUDED_COLUMNS] if column in dataframe.columns])
    numeric_features = features.select_dtypes(include=["number", "bool"]).copy()

    if numeric_features.empty:
        raise ValueError("No numeric features available for supervised training.")

    numeric_features = numeric_features.fillna(0)

    unique_classes = y.nunique()
    if unique_classes < 2:
        raise ValueError(
            "Supervised training requires at least two classes in target 'is_top_product'. "
            f"Found {unique_classes} class."
        )

    return numeric_features, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train an XGBoost classifier for binary top-product classification."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Compute strict classification metrics and confusion matrix."""
    y_pred = model.predict(X_test)

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": int(len(y_test)),
    }

    return metrics


def run() -> None:
    """Execute supervised model training, evaluation, and artifact persistence."""
    project_root = get_project_root()
    input_path, model_path, metrics_path = get_paths(project_root)

    try:
        dataframe = load_training_data(input_path)
        X, y = build_feature_matrix(dataframe)

        min_class_count = int(y.value_counts().min())
        class_count = int(y.nunique())
        estimated_test_count = max(1, int(round(len(y) * 0.2)))
        can_stratify = min_class_count >= 2 and estimated_test_count >= class_count
        stratify_target = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
        )

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        model.save_model(str(model_path))
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("supervised_training_failed", extra={"error": str(exc)})
        raise

    LOGGER.info(
        "supervised_training_completed",
        extra={
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
        },
    )


if __name__ == "__main__":
    configure_logging()
    run()
