"""Preprocess enriched product data for downstream ML tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "product_id",
    "price",
    "promotional_price",
    "rating",
    "review_count",
    "stock_status",
    "standardized_category",
)


def configure_logging() -> None:
    """Configure baseline logging for preprocessing execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_project_root() -> Path:
    """Return the project root path from the current file location."""
    return Path(__file__).resolve().parents[1]


def get_paths(project_root: Path) -> tuple[Path, Path]:
    """Build input and output paths for preprocessing."""
    input_path = project_root / "data" / "processed" / "enriched_products.json"
    output_path = project_root / "data" / "processed" / "ml_ready_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_path, output_path


def load_enriched_dataframe(input_path: Path) -> pd.DataFrame:
    """Load enriched products JSON into a pandas DataFrame."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Expected enriched_products.json to contain a JSON array.")

    dataframe = pd.DataFrame(payload)
    if dataframe.empty:
        raise ValueError("Input dataset is empty. Nothing to preprocess.")

    return dataframe


def validate_columns(dataframe: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure required columns are present in the DataFrame."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a model-ready dataset with engineered and encoded features."""
    result = dataframe.copy()

    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    rating_median = float(result["rating"].median()) if result["rating"].notna().any() else 0.0
    result["rating"] = result["rating"].fillna(rating_median)

    result["review_count"] = pd.to_numeric(result["review_count"], errors="coerce").fillna(0).astype(int)
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    price_median = float(result["price"].median()) if result["price"].notna().any() else 0.0
    result["price"] = result["price"].fillna(price_median)
    result["promotional_price"] = pd.to_numeric(result["promotional_price"], errors="coerce")
    result["promotional_price"] = result["promotional_price"].fillna(result["price"])

    result["standardized_category"] = result["standardized_category"].fillna("Unknown").astype(str)
    result["stock_status"] = result["stock_status"].fillna("unknown").astype(str)
    result["is_in_stock"] = (result["stock_status"].str.lower() == "in_stock").astype(int)

    result["discount_ratio"] = (
        (result["price"] - result["promotional_price"]) / result["price"].replace(0, pd.NA)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)

    rating_threshold = float(max(3.5, result["rating"].quantile(0.70)))
    review_threshold = float(max(1.0, result["review_count"].quantile(0.70)))
    discount_threshold = float(max(0.10, result["discount_ratio"].quantile(0.70)))

    result["is_top_product"] = (
        (
            (result["rating"] >= rating_threshold)
            & (result["review_count"] >= review_threshold)
            & (result["is_in_stock"] == 1)
        )
        |
        (
            (result["discount_ratio"] >= discount_threshold)
            & (result["is_in_stock"] == 1)
        )
    ).astype(int)

    if result["is_top_product"].nunique() < 2 and len(result) > 1:
        LOGGER.warning(
            "target_single_class_detected",
            extra={"detail": "Applying score-based fallback to synthesize class diversity."},
        )
        fallback_score = (
            result["discount_ratio"] * 0.5
            + result["rating"] * 0.3
            + result["is_in_stock"] * 0.2
        )
        minimum_positive = 2 if len(result) >= 4 else 1
        top_n = max(minimum_positive, int(round(len(result) * 0.2)))
        top_n = min(top_n, len(result) - 1)
        selected_index = fallback_score.nlargest(top_n).index
        result["is_top_product"] = 0
        result.loc[selected_index, "is_top_product"] = 1

    encoded = pd.get_dummies(
        result,
        columns=["standardized_category", "stock_status"],
        prefix=["cat", "stock"],
        dtype=int,
    )

    # Align with unsupervised requirements by exposing a `reviews` numeric feature.
    encoded["reviews"] = encoded["review_count"].astype(float)

    return encoded


def run() -> None:
    """Run preprocessing pipeline and persist ML-ready CSV."""
    project_root = get_project_root()
    input_path, output_path = get_paths(project_root)

    try:
        dataframe = load_enriched_dataframe(input_path)
        validate_columns(dataframe, REQUIRED_COLUMNS)
        processed = preprocess_features(dataframe)
        processed.to_csv(output_path, index=False)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("preprocessing_failed", extra={"error": str(exc)})
        raise

    LOGGER.info(
        "preprocessing_completed",
        extra={
            "rows": len(processed),
            "columns": len(processed.columns),
            "output_path": str(output_path),
        },
    )


if __name__ == "__main__":
    configure_logging()
    run()
