"""Entry point for testing the Shopify scraping agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List

from scraping.schemas import Product
from scraping.shopify_agent import ShopifyAgent, ShopifyAgentError


def configure_logging() -> None:
    """Configure baseline logging for local execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_output_path() -> Path:
    """Return output path for raw sample products."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "raw" / "sample_products.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


async def run() -> None:
    """Scrape a sample Shopify store and persist validated data to disk."""
    store_url = os.environ.get("SHOPIFY_STORE_URL", "https://allbirds.com")
    agent = ShopifyAgent()

    try:
        products: List[Product] = await agent.scrape_store(store_url)
    except ShopifyAgentError as exc:
        logging.getLogger(__name__).error("scrape_failed", extra={"error": str(exc)})
        raise

    output_path = get_output_path()
    payload = [product.model_dump(mode="json") for product in products]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logging.getLogger(__name__).info(
        "sample_output_written",
        extra={"store": store_url, "count": len(products), "path": str(output_path)},
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run())
