"""Kubeflow pipeline definition for eCommerce Intelligence ML orchestration."""

from pathlib import Path
import sys

import kfp
from kfp import dsl
from kfp.dsl import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.kfp_components import (  # noqa: E402
    preprocess_data_op,
    score_top_products_op,
    train_supervised_op,
    train_unsupervised_op,
)


@dsl.pipeline(name="ecommerce-intelligence-pipeline")
def ecommerce_intelligence_pipeline(
    enriched_products_uri: str = "data/processed/enriched_products.json",
) -> None:
    """Run preprocessing, parallel training, and top-k scoring."""
    enriched_input = dsl.importer(
        artifact_uri=enriched_products_uri,
        artifact_class=Dataset,
        reimport=False,
    )

    preprocess_task = preprocess_data_op(
        enriched_products=enriched_input.output,
    )

    supervised_task = train_supervised_op(
        ml_ready_data=preprocess_task.outputs["ml_ready_data"],
    )

    unsupervised_task = train_unsupervised_op(
        ml_ready_data=preprocess_task.outputs["ml_ready_data"],
    )

    score_top_products_op(
        enriched_products=enriched_input.output,
        ml_ready_data=preprocess_task.outputs["ml_ready_data"],
        supervised_model=supervised_task.outputs["supervised_model"],
        kmeans_model=unsupervised_task.outputs["kmeans_model"],
        kmeans_scaler=unsupervised_task.outputs["kmeans_scaler"],
    )


if __name__ == "__main__":
    output_path = Path(__file__).with_name("pipeline.yaml")
    kfp.compiler.Compiler().compile(
        pipeline_func=ecommerce_intelligence_pipeline,
        package_path=str(output_path),
    )
