"""Run LLM enrichment on scraped sample products."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List

from pydantic import ValidationError

from llm_agents.enricher import DataEnrichmentAgent
from scraping.schemas import Product

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure structured logging for enrichment execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def load_env_file(project_root: Path) -> None:
    """Load .env variables if they are not already present in the environment."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_paths(project_root: Path) -> tuple[Path, Path]:
    """Return input and output paths for enrichment flow."""
    raw_path = project_root / "data" / "raw" / "sample_products.json"
    output_path = project_root / "data" / "processed" / "enriched_products.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path, output_path


def load_products(raw_path: Path, limit: int = 5) -> List[Product]:
    """Load and validate the first N products from raw scraped JSON."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw products file does not exist: {raw_path}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected sample_products.json to contain a JSON array.")

    validated_products: List[Product] = []
    for idx, item in enumerate(payload[:limit]):
        try:
            validated_products.append(Product.model_validate(item))
        except ValidationError as exc:
            LOGGER.warning(
                "raw_product_validation_failed",
                extra={"index": idx, "error": str(exc)},
            )

    return validated_products


async def run() -> None:
    """Execute a bounded batch enrichment and persist processed output."""
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root)

    raw_path, output_path = get_paths(project_root)

    if not os.environ.get("GOOGLE_API_KEY"):
        LOGGER.error(
            "missing_google_api_key",
            extra={"hint": "Set GOOGLE_API_KEY in environment or .env"},
        )
        return

    try:
        products = load_products(raw_path=raw_path, limit=5)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("failed_to_load_raw_products", extra={"error": str(exc)})
        return

    if not products:
        LOGGER.warning("no_valid_products_to_enrich")
        return

    try:
        agent = DataEnrichmentAgent()
        enriched_products, errors = await agent.enrich_batch(products, max_concurrency=3)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("enrichment_pipeline_failed", extra={"error": str(exc)})
        return

    output_payload = [item.model_dump(mode="json") for item in enriched_products]
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "enrichment_completed",
        extra={
            "input_count": len(products),
            "enriched_count": len(enriched_products),
            "error_count": len(errors),
            "output_path": str(output_path),
        },
    )

    if errors:
        LOGGER.warning("enrichment_partial_failures", extra={"errors": errors})


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run())
