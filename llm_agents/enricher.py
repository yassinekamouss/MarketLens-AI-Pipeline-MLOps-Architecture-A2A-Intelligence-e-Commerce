"""LLM agent for product data cleaning and enrichment."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from llm_agents.schemas import EnrichedProduct
from scraping.schemas import Product

LOGGER = logging.getLogger(__name__)

_ALLOWED_CATEGORIES: Tuple[str, ...] = (
    "Electronics",
    "Apparel",
    "Home",
    "Health",
    "Unknown",
)


class DataEnrichmentAgent:
    """Enrich scraped products with normalized categories and concise metadata."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialize the DeepSeek-backed structured output chain and local cache."""
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is missing. Set it in environment variables before running enrichment."
            )

        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=temperature,
            max_retries=3
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior eCommerce data analyst. "
                    "Your role is to clean and enrich raw product catalog data. "
                    "Always preserve original factual data exactly for existing fields. "
                    "Map standardized_category to one of: {allowed_categories}. "
                    "If uncertain, use Unknown. "
                    "Write short_summary in plain English with at most two sentences. "
                    "Create extracted_tags as 3 to 5 concise keywords. "
                    "Avoid marketing fluff, hallucinations, and unsupported claims.",
                ),
                (
                    "human",
                    "Enrich this product JSON and return fields matching the schema:\n{product_json}",
                ),
            ]
        )

        self._chain = prompt | llm.with_structured_output(EnrichedProduct)
        
        # Cache initialization
        project_root = Path(__file__).resolve().parents[1]
        self._cache_path = project_root / "data" / "processed" / "llm_cache.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_lock = asyncio.Lock()
        self._cache: Dict[str, dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, dict]:
        """Load cache from disk."""
        if self._cache_path.exists():
            try:
                content = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(content, dict):
                    return content
            except json.JSONDecodeError:
                LOGGER.warning("Cache file corrupted, starting fresh.")
        return {}

    async def _save_cache_entry(self, product_id: str, data: dict) -> None:
        """Save a single entry to memory and asynchronously to disk."""
        async with self._cache_lock:
            self._cache[product_id] = data
            self._cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm(self, product: Product) -> EnrichedProduct:
        """Call the LLM with retry logic."""
        return await self._chain.ainvoke(
            {
                "allowed_categories": ", ".join(_ALLOWED_CATEGORIES),
                "product_json": json.dumps(product.model_dump(mode="json"), ensure_ascii=False),
            }
        )

    async def enrich_product(self, product: Product) -> EnrichedProduct:
        """Enrich a single product, using cache if available."""
        product_id = product.product_id
        
        if product_id in self._cache:
            try:
                return EnrichedProduct.model_validate(self._cache[product_id])
            except ValidationError:
                LOGGER.warning(f"Invalid cache entry for {product_id}, forcing re-enrichment.")
                
        enriched = await self._call_llm(product)
        await self._save_cache_entry(product_id, enriched.model_dump(mode="json"))
        return enriched

    async def enrich_batch(
        self,
        products: Sequence[Product],
        max_concurrency: int = 3,
    ) -> Tuple[List[EnrichedProduct], List[Dict[str, str]]]:
        """Enrich a batch of products with bounded concurrency for API safety."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        semaphore = asyncio.Semaphore(max_concurrency)
        enriched_products: List[EnrichedProduct] = []
        errors: List[Dict[str, str]] = []

        async def _enrich_with_limit(product: Product) -> EnrichedProduct:
            async with semaphore:
                result = await self.enrich_product(product)
                await asyncio.sleep(0.5)
                return result

        tasks = [asyncio.create_task(_enrich_with_limit(product)) for product in products]

        for index, task in enumerate(tasks):
            product = products[index]
            try:
                result = await task
                enriched_products.append(result)
            except ValidationError as exc:
                LOGGER.warning(
                    "enrichment_validation_failed",
                    extra={"product_id": product.product_id, "error": str(exc)},
                )
                errors.append({"product_id": product.product_id, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "enrichment_call_failed",
                    extra={"product_id": product.product_id, "error": str(exc)},
                )
                errors.append({"product_id": product.product_id, "error": str(exc)})

        return enriched_products, errors
