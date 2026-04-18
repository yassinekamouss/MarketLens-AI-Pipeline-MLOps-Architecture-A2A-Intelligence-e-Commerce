"""LLM agent for product data cleaning and enrichment."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Sequence, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

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
        """Initialize the Gemini-backed structured output chain."""
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Set it in environment variables before running enrichment."
            )

        selected_model = model_name or os.environ.get(
            "GEMINI_MODEL_NAME", "gemini-flash-latest"
        )

        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=google_api_key,
            temperature=temperature,
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

    async def enrich_product(self, product: Product) -> EnrichedProduct:
        """Enrich a single product and validate the output against EnrichedProduct."""
        return await self._chain.ainvoke(
            {
                "allowed_categories": ", ".join(_ALLOWED_CATEGORIES),
                "product_json": json.dumps(product.model_dump(mode="json"), ensure_ascii=False),
            }
        )

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
                return await self.enrich_product(product)

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
