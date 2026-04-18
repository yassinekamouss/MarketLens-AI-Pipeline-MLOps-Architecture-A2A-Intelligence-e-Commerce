"""Data contract schemas for scraping outputs."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Product(BaseModel):
    """Canonical product schema used by downstream ML and analytics components."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(default="")
    category: str = Field(default="unknown")
    price: float = Field(..., ge=0.0)
    promotional_price: Optional[float] = Field(default=None, ge=0.0)
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    review_count: int = Field(default=0, ge=0)
    stock_status: str = Field(..., min_length=1)
    variants: List[Dict] = Field(default_factory=list)
