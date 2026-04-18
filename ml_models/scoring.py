"""Score products using trained models and produce top-k shortlist."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
TOP_K = 50


def configure_logging() -> None:
    """Configure baseline logging for scoring execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_project_root() -> Path:
    """Return the project root path from the current file location."""
    return Path(__file__).resolve().parents[1]


def get_paths(project_root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return input and artifact paths for scoring flow."""
    enriched_path = project_root / "data" / "processed" / "enriched_products.json"
    ml_ready_path = project_root / "data" / "processed" / "ml_ready_data.csv"

    artifacts_dir = project_root / "ml_models" / "artifacts"
    xgb_path = artifacts_dir / "xgb_model.json"
    scaler_path = artifacts_dir / "kmeans_scaler.joblib"
    kmeans_path = artifacts_dir / "kmeans_model.joblib"

    output_path = project_root / "data" / "processed" / "top_k_products.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return enriched_path, ml_ready_path, xgb_path, scaler_path, kmeans_path, output_path


def load_json_records(path: Path) -> List[dict]:
    """Load a JSON array from disk and return dictionary records."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")

    return [item for item in payload if isinstance(item, dict)]


def load_ml_ready(path: Path) -> pd.DataFrame:
    """Load ML-ready data and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"ML-ready data not found: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError("ML-ready data is empty.")
    if "product_id" not in dataframe.columns:
        raise ValueError("Column 'product_id' is required in ml_ready_data.csv.")

    return dataframe


def prepare_supervised_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build numeric feature matrix expected by XGBClassifier."""
    excluded = {
        "is_top_product",
        "product_id",
        "name",
        "description",
        "category",
        "short_summary",
        "extracted_tags",
        "variants",
    }
    features = dataframe.drop(columns=[column for column in excluded if column in dataframe.columns])
    features = features.select_dtypes(include=["number", "bool"]).fillna(0)
    return features


def apply_cluster_quality_signal(dataframe: pd.DataFrame) -> pd.Series:
    """Assign a cluster-based quality score using cluster-level means."""
    cluster_quality = (
        dataframe.groupby("cluster_id")[["rating", "reviews"]]
        .mean()
        .assign(cluster_score=lambda frame: frame["rating"] * 0.7 + frame["reviews"] * 0.3)
    )

    return dataframe["cluster_id"].map(cluster_quality["cluster_score"]).fillna(0.0)


def run() -> None:
    """Run scoring pipeline and persist top-k products JSON."""
    project_root = get_project_root()
    (
        enriched_path,
        ml_ready_path,
        xgb_path,
        scaler_path,
        kmeans_path,
        output_path,
    ) = get_paths(project_root)

    try:
        enriched_records = load_json_records(enriched_path)
        enriched_df = pd.DataFrame(enriched_records)
        ml_ready_df = load_ml_ready(ml_ready_path)

        if "reviews" not in ml_ready_df.columns and "review_count" in ml_ready_df.columns:
            ml_ready_df["reviews"] = pd.to_numeric(ml_ready_df["review_count"], errors="coerce").fillna(0)

        model = XGBClassifier()
        model.load_model(str(xgb_path))

        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)

        supervised_features = prepare_supervised_features(ml_ready_df)
        ml_ready_df["top_product_probability"] = model.predict_proba(supervised_features)[:, 1]

        clustering_features = (
            ml_ready_df.loc[:, ["price", "rating", "reviews"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        scaled = scaler.transform(clustering_features)
        ml_ready_df["cluster_id"] = kmeans.predict(scaled)

        ml_ready_df["cluster_quality_score"] = apply_cluster_quality_signal(ml_ready_df)
        ml_ready_df["rating"] = pd.to_numeric(ml_ready_df["rating"], errors="coerce").fillna(0)
        ml_ready_df["reviews"] = pd.to_numeric(ml_ready_df["reviews"], errors="coerce").fillna(0)
        ml_ready_df["in_stock_flag"] = (
            ml_ready_df.get("stock_in_stock", pd.Series([0] * len(ml_ready_df))).astype(float)
        )

        ml_ready_df["final_score"] = (
            ml_ready_df["top_product_probability"] * 0.60
            + ml_ready_df["cluster_quality_score"] * 0.15
            + ml_ready_df["rating"] * 0.15
            + ml_ready_df["reviews"] * 0.05
            + ml_ready_df["in_stock_flag"] * 0.05
        )

        enriched_lookup = enriched_df.set_index("product_id").to_dict(orient="index") if not enriched_df.empty else {}

        ranked = ml_ready_df.sort_values("final_score", ascending=False).head(TOP_K).copy()
        ranked["product_id"] = ranked["product_id"].astype(str)

        output_records: List[dict] = []
        for _, row in ranked.iterrows():
            product_id = row["product_id"]
            base_record = enriched_lookup.get(product_id, {"product_id": product_id})

            enriched_record = {
                **base_record,
                "top_product_probability": float(row["top_product_probability"]),
                "cluster_id": int(row["cluster_id"]),
                "cluster_quality_score": float(row["cluster_quality_score"]),
                "final_score": float(row["final_score"]),
            }
            output_records.append(enriched_record)

        output_path.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("scoring_failed", extra={"error": str(exc)})
        raise

    LOGGER.info(
        "scoring_completed",
        extra={
            "top_k": len(output_records),
            "output_path": str(output_path),
        },
    )


if __name__ == "__main__":
    configure_logging()
    run()
