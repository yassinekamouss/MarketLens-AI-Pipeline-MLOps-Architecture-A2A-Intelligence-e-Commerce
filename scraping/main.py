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
from scraping.woocommerce_agent import WooCommerceAgent, WooCommerceAgentError


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
    """Scrape a sample store dynamically based on platform and persist validated data to disk."""
    store_platform = os.environ.get("STORE_PLATFORM", "shopify").lower()
    store_url = os.environ.get("STORE_URL", "https://allbirds.com")

    if store_platform == "shopify":
        agent = ShopifyAgent()
    elif store_platform == "woocommerce":
        agent = WooCommerceAgent()
    else:
        logging.getLogger(__name__).error(
            "unsupported_platform",
            extra={"store_platform": store_platform}
        )
        raise ValueError(f"Unsupported STORE_PLATFORM: {store_platform}")

    try:
        if store_platform == "shopify":
            products: List[Product] = await agent.scrape_store(store_url)
        else: # woocommerce
            products: List[Product] = await agent.scrape_store(store_url)
    except (ShopifyAgentError, WooCommerceAgentError) as exc:
        logging.getLogger(__name__).error("scrape_failed", extra={"error": str(exc), "platform": store_platform})
        raise

    output_path = get_output_path()
    payload = [product.model_dump(mode="json") for product in products]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logging.getLogger(__name__).info(
        "sample_output_written",
        extra={"store": store_url, "platform": store_platform, "count": len(products), "path": str(output_path)},
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run())
