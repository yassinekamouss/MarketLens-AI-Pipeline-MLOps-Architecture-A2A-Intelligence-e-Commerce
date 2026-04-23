"""MCP server exposing safe, read-only product intelligence tools."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

LOGGER = logging.getLogger(__name__)

SERVER_NAME = "eCommerce-Intelligence-Server"
DATA_PATH_ENV = "TOP_PRODUCTS_PATH"
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "top_k_products.json"
)

mcp = FastMCP(SERVER_NAME)


def _get_data_path() -> Path:
    """Resolve processed top-k dataset path from env or project default."""
    raw_path = os.environ.get(DATA_PATH_ENV)
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_DATA_PATH


def _load_products() -> list[dict[str, Any]]:
    """Load and validate the top products JSON payload."""
    data_path = _get_data_path()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Top products dataset not found at {data_path}. "
            f"Set {DATA_PATH_ENV} to override the default path."
        )

    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format in {data_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Expected top_k_products.json to contain a JSON array.")

    validated_items: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            LOGGER.warning("Skipping non-object product record at index %s", idx)
            continue
        validated_items.append(item)

    return validated_items


def _coerce_float(value: Any) -> float | None:
    """Safely convert values to float, returning None for invalid inputs."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@mcp.tool()
def get_top_products(limit: int = 10) -> list[dict[str, Any]]:
    """Return the top N products sorted by final score in descending order."""
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer.")
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    if limit > 200:
        raise ValueError("limit must be <= 200 to keep responses bounded.")

    products = _load_products()
    ranked_products = sorted(
        products,
        key=lambda item: _coerce_float(item.get("final_score")) or 0.0,
        reverse=True,
    )
    return ranked_products[:limit]


@mcp.tool()
def get_cluster_summary(cluster_id: int) -> dict[str, Any]:
    """Return aggregate metrics for products inside a specific KMeans cluster."""
    if not isinstance(cluster_id, int):
        raise ValueError("cluster_id must be an integer.")
    if cluster_id < 0:
        raise ValueError("cluster_id must be >= 0.")

    products = _load_products()
    cluster_items = [item for item in products if item.get("cluster_id") == cluster_id]

    if not cluster_items:
        raise ValueError(f"No products found for cluster_id={cluster_id}.")

    prices = [value for item in cluster_items if (value := _coerce_float(item.get("price"))) is not None]
    ratings = [
        value for item in cluster_items if (value := _coerce_float(item.get("rating"))) is not None
    ]

    avg_price = round(sum(prices) / len(prices), 4) if prices else None
    avg_rating = round(sum(ratings) / len(ratings), 4) if ratings else None

    return {
        "cluster_id": cluster_id,
        "product_count": len(cluster_items),
        "avg_price": avg_price,
        "avg_rating": avg_rating,
        "price_samples": len(prices),
        "rating_samples": len(ratings),
    }


def main() -> None:
    """Start MCP server over stdio (default) or SSE based on MCP_TRANSPORT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in {"stdio", "sse"}:
        raise ValueError("MCP_TRANSPORT must be either 'stdio' or 'sse'.")

    LOGGER.info("Starting MCP server", extra={"server_name": SERVER_NAME, "transport": transport})
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
