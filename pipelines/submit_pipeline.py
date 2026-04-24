"""Submit the compiled Kubeflow pipeline to a local KFP instance."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kfp import Client

DEFAULT_HOST = "http://localhost:8080"
DEFAULT_EXPERIMENT_NAME = "Smart-eCommerce-Exp"
PIPELINE_PACKAGE = Path(__file__).resolve().with_name("pipeline.yaml")


def configure_logging() -> None:
    """Configure application logging with timestamps and levels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments for local or custom runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Kubeflow Pipelines endpoint (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Experiment name (default: {DEFAULT_EXPERIMENT_NAME}).",
    )
    parser.add_argument(
        "--pipeline-package",
        default=str(PIPELINE_PACKAGE),
        help="Path to compiled pipeline YAML package.",
    )
    return parser.parse_args()


def create_kfp_client(host: str) -> Client:
    """Create and validate a KFP client against the target host."""
    client = Client(host=host)
    client.list_experiments(page_size=1)
    return client


def get_or_create_experiment_id(client: Client, experiment_name: str) -> str:
    """Return an existing experiment id or create the experiment when absent."""
    experiment_id: Optional[str] = None

    try:
        existing_experiment = client.get_experiment(experiment_name=experiment_name)
        experiment_id = getattr(existing_experiment, "experiment_id", None)
        if experiment_id:
            logging.info(
                "Using existing experiment '%s' (id=%s).",
                experiment_name,
                experiment_id,
            )
            return experiment_id
    except Exception as exc:  # noqa: BLE001
        logging.info(
            "Experiment '%s' not found or not retrievable (%s). Creating it.",
            experiment_name,
            exc,
        )

    created_experiment = client.create_experiment(name=experiment_name)
    experiment_id = getattr(created_experiment, "experiment_id", None)
    if not experiment_id:
        raise RuntimeError(
            f"KFP returned an invalid experiment response for '{experiment_name}'."
        )

    logging.info("Created experiment '%s' (id=%s).", experiment_name, experiment_id)
    return experiment_id


def submit_pipeline_run(
    client: Client,
    host: str,
    experiment_id: str,
    pipeline_package: Path,
) -> str:
    """Submit the pipeline package and return the generated run URL."""
    run_name = f"smart-ecommerce-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    run = client.create_run_from_pipeline_package(
        pipeline_file=str(pipeline_package),
        experiment_id=experiment_id,
        run_name=run_name,
    )

    run_id = getattr(run, "run_id", None)
    if not run_id and getattr(run, "run", None) is not None:
        run_id = getattr(run.run, "id", None)

    if not run_id:
        raise RuntimeError("Pipeline run was submitted, but no run id was returned.")

    run_url = f"{host.rstrip('/')}/#/runs/details/{run_id}"
    return run_url


def main() -> int:
    """Submit the compiled pipeline to the configured Kubeflow Pipelines endpoint."""
    configure_logging()
    args = parse_args()

    pipeline_package = Path(args.pipeline_package).resolve()
    if not pipeline_package.exists():
        logging.error("Pipeline package not found: %s", pipeline_package)
        return 1

    try:
        logging.info("Connecting to Kubeflow Pipelines at %s", args.host)
        client = create_kfp_client(args.host)

        experiment_id = get_or_create_experiment_id(client, args.experiment)
        run_url = submit_pipeline_run(
            client=client,
            host=args.host,
            experiment_id=experiment_id,
            pipeline_package=pipeline_package,
        )

        logging.info("Pipeline submitted successfully.")
        logging.info("Run URL: %s", run_url)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.exception("Pipeline submission failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
