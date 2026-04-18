"""Pydantic schemas for LLM-enriched product data."""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from scraping.schemas import Product


class EnrichedProduct(Product):
    """Extended product schema produced by the LLM enrichment stage."""

    standardized_category: Literal[
        "Electronics",
        "Apparel",
        "Home",
        "Health",
        "Unknown",
    ] = Field(..., description="Mapped category from the controlled taxonomy.")
    short_summary: str = Field(
        ...,
        min_length=1,
        max_length=320,
        description="Concise summary of the product in at most two sentences.",
    )
    extracted_tags: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="SEO or analytical keywords extracted from product content.",
    )
