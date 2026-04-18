"""Asynchronous Shopify scraping agent with API-first strategy and browser fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from playwright.async_api import async_playwright
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from scraping.schemas import Product

LOGGER = logging.getLogger(__name__)
_TAG_RE = re.compile(r"<[^>]+>")


class ShopifyAgentError(Exception):
    """Raised when scraping cannot produce valid products."""


class ShopifyAgent:
    """Scrapes Shopify stores with resilient retries and strict data validation."""

    def __init__(self, timeout_seconds: int = 20, max_pages: int = 20, page_size: int = 250) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_pages = max_pages
        self.page_size = min(page_size, 250)

    async def scrape_store(self, store_url: str) -> List[Product]:
        """Scrape and validate products from a Shopify store.

        Args:
            store_url: Base store URL (for example: https://examplestore.com).

        Returns:
            A list of validated Product instances.
        """
        normalized_url = store_url.rstrip("/")
        raw_products: List[Dict[str, Any]] = []

        try:
            raw_products = await self._fetch_products_from_api(normalized_url)
            LOGGER.info(
                "shopify_api_scrape_complete",
                extra={"store": normalized_url, "raw_count": len(raw_products)},
            )
        except ShopifyAgentError as exc:
            LOGGER.warning(
                "shopify_api_scrape_failed_using_playwright_fallback",
                extra={"store": normalized_url, "error": str(exc)},
            )
            raw_products = await self._fetch_products_with_playwright(normalized_url)

        validated_products: List[Product] = []
        for item in raw_products:
            parsed = self._map_shopify_product(item)
            if parsed is None:
                continue
            validated_products.append(parsed)
            if len(validated_products) % 50 == 0:
                LOGGER.info(
                    "shopify_validation_progress",
                    extra={"store": normalized_url, "validated_count": len(validated_products)},
                )

        LOGGER.info(
            "shopify_scrape_complete",
            extra={"store": normalized_url, "validated_count": len(validated_products)},
        )

        if not validated_products:
            raise ShopifyAgentError(f"No valid products were scraped from store: {normalized_url}")

        return validated_products

    @retry(
        reraise=True,
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
    )
    async def _request_json(self, client: httpx.AsyncClient, endpoint: str) -> Dict[str, Any]:
        response = await client.get(endpoint)

        # Rate limiting and transient errors are retried by tenacity.
        if response.status_code in {429, 500, 502, 503, 504}:
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    async def _fetch_products_from_api(self, store_url: str) -> List[Dict[str, Any]]:
        products: List[Dict[str, Any]] = []
        timeout = httpx.Timeout(timeout=self.timeout_seconds)

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                for page in range(1, self.max_pages + 1):
                    endpoint = (
                        f"{store_url}/products.json?limit={self.page_size}&page={page}"
                    )
                    payload = await self._request_json(client, endpoint)
                    page_products = payload.get("products", [])

                    if not page_products:
                        break

                    products.extend(page_products)
                    LOGGER.info(
                        "shopify_api_page_scraped",
                        extra={
                            "store": store_url,
                            "page": page,
                            "page_count": len(page_products),
                            "total_count": len(products),
                        },
                    )

                    await asyncio.sleep(0.05)

            if not products:
                raise ShopifyAgentError(f"No products returned by API for store: {store_url}")
            return products

        except RetryError as exc:
            raise ShopifyAgentError(f"API retry attempts exhausted for store: {store_url}") from exc
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
            raise ShopifyAgentError(f"API scraping failed for store {store_url}: {exc}") from exc

    async def _fetch_products_with_playwright(self, store_url: str) -> List[Dict[str, Any]]:
        endpoint = f"{store_url}/products.json?limit={self.page_size}"

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(store_url, wait_until="networkidle", timeout=self.timeout_seconds * 1000)

                payload = await page.evaluate(
                    """
                    async (url) => {
                        const response = await fetch(url, { method: 'GET' });
                        if (!response.ok) {
                            throw new Error(`Fetch failed with status ${response.status}`);
                        }
                        return await response.json();
                    }
                    """,
                    endpoint,
                )

                await browser.close()

            products = payload.get("products", []) if isinstance(payload, dict) else []
            LOGGER.info(
                "shopify_playwright_fallback_complete",
                extra={"store": store_url, "raw_count": len(products)},
            )
            return products
        except Exception as exc:  # noqa: BLE001
            raise ShopifyAgentError(
                f"Playwright fallback failed for store {store_url}: {exc}"
            ) from exc

    def _map_shopify_product(self, raw_product: Dict[str, Any]) -> Optional[Product]:
        try:
            variants_raw = raw_product.get("variants", [])
            variants: List[Dict[str, Any]] = []
            variant_prices: List[float] = []
            compare_prices: List[float] = []
            in_stock = False

            for variant in variants_raw:
                raw_price = variant.get("price")
                raw_compare_at = variant.get("compare_at_price")
                price = float(raw_price) if raw_price not in (None, "") else 0.0
                compare_at_price = (
                    float(raw_compare_at) if raw_compare_at not in (None, "") else None
                )

                variant_prices.append(price)
                if compare_at_price is not None:
                    compare_prices.append(compare_at_price)

                available = bool(variant.get("available", False))
                in_stock = in_stock or available

                variants.append(
                    {
                        "variant_id": str(variant.get("id", "")),
                        "title": variant.get("title", ""),
                        "sku": variant.get("sku", ""),
                        "price": price,
                        "compare_at_price": compare_at_price,
                        "inventory_quantity": int(variant.get("inventory_quantity") or 0),
                        "available": available,
                    }
                )

            current_price = min(variant_prices) if variant_prices else 0.0
            compare_price = min(compare_prices) if compare_prices else None

            product_price = compare_price if compare_price and compare_price > current_price else current_price
            promo_price = (
                current_price
                if compare_price is not None and compare_price > current_price
                else None
            )

            description = _TAG_RE.sub(" ", raw_product.get("body_html", "")).strip()
            stock_status = "in_stock" if in_stock else "out_of_stock"

            return Product(
                product_id=str(raw_product.get("id", "")),
                name=raw_product.get("title", "").strip(),
                description=description,
                category=(raw_product.get("product_type") or "unknown").strip() or "unknown",
                price=product_price,
                promotional_price=promo_price,
                rating=None,
                review_count=0,
                stock_status=stock_status,
                variants=variants,
            )

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "shopify_product_validation_failed",
                extra={
                    "raw_product_id": str(raw_product.get("id", "unknown")),
                    "error": str(exc),
                },
            )
            return None
