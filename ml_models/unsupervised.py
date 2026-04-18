"""Train unsupervised clustering model to segment products."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
NUMERIC_CLUSTER_FEATURES: tuple[str, ...] = ("price", "rating", "reviews")


def configure_logging() -> None:
    """Configure baseline logging for unsupervised training execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_project_root() -> Path:
    """Return the project root path from the current file location."""
    return Path(__file__).resolve().parents[1]


def get_paths(project_root: Path) -> tuple[Path, Path, Path, Path]:
    """Return input and output artifact paths for clustering flow."""
    input_path = project_root / "data" / "processed" / "ml_ready_data.csv"
    artifacts_dir = project_root / "ml_models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = artifacts_dir / "kmeans_scaler.joblib"
    model_path = artifacts_dir / "kmeans_model.joblib"
    metrics_path = artifacts_dir / "unsupervised_metrics.json"
    return input_path, scaler_path, model_path, metrics_path


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load ML-ready dataset for unsupervised segmentation."""
    if not csv_path.exists():
        raise FileNotFoundError(f"ML-ready dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("ML-ready dataset is empty.")

    missing_features = [feature for feature in NUMERIC_CLUSTER_FEATURES if feature not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Missing clustering features: {missing_features}")

    return dataframe


def train_clustering(dataframe: pd.DataFrame) -> tuple[StandardScaler, KMeans, Dict[str, float]]:
    """Train StandardScaler + KMeans(3) and compute silhouette score."""
    numerical = dataframe.loc[:, NUMERIC_CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    if len(numerical) < 4:
        raise ValueError("Need at least 4 samples to run 3-cluster KMeans and silhouette scoring.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numerical)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    score = float(silhouette_score(scaled, labels))
    metrics: Dict[str, float] = {
        "silhouette_score": score,
        "n_clusters": 3.0,
        "n_samples": float(len(numerical)),
    }

    return scaler, model, metrics


def run() -> None:
    """Execute unsupervised model training and persist artifacts."""
    project_root = get_project_root()
    input_path, scaler_path, model_path, metrics_path = get_paths(project_root)

    try:
        dataframe = load_data(input_path)
        scaler, model, metrics = train_clustering(dataframe)

        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("unsupervised_training_failed", extra={"error": str(exc)})
        raise

    LOGGER.info(
        "unsupervised_training_completed",
        extra={
            "scaler_path": str(scaler_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "silhouette_score": metrics["silhouette_score"],
        },
    )


if __name__ == "__main__":
    configure_logging()
    run()
