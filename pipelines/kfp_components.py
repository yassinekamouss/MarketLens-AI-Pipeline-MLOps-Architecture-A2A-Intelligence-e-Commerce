"""Kubeflow lightweight components for the eCommerce Intelligence ML workflow."""

import json
from pathlib import Path

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
    ],
)
def preprocess_data_op(
    enriched_products: Input[Dataset],
    ml_ready_data: Output[Dataset],
) -> None:
    """Preprocess enriched products into an ML-ready tabular dataset."""
    import logging

    import pandas as pd
    from ml_models import preprocessing

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    input_path = Path(enriched_products.path)
    output_path = Path(ml_ready_data.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = preprocessing.load_enriched_dataframe(input_path)
    preprocessing.validate_columns(dataframe, preprocessing.REQUIRED_COLUMNS)
    processed = preprocessing.preprocess_features(dataframe)
    processed.to_csv(output_path, index=False)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
    ],
)
def train_supervised_op(
    ml_ready_data: Input[Dataset],
    supervised_model: Output[Model],
    supervised_metrics: Output[Dataset],
) -> None:
    """Train supervised XGBoost model and persist model + metrics artifacts."""
    import logging

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ml_models import supervised

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dataset_path = Path(ml_ready_data.path)
    model_path = Path(supervised_model.path)
    metrics_path = Path(supervised_metrics.path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty:
        raise ValueError("ML-ready dataset is empty.")

    X, y = supervised.build_feature_matrix(dataframe)

    min_class_count = int(y.value_counts().min())
    class_count = int(y.nunique())
    estimated_test_count = max(1, int(round(len(y) * 0.2)))
    can_stratify = min_class_count >= 2 and estimated_test_count >= class_count
    stratify_target = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    model = supervised.train_model(X_train, y_train)
    metrics = supervised.evaluate_model(model, X_test, y_test)

    model.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
    ],
)
def train_unsupervised_op(
    ml_ready_data: Input[Dataset],
    kmeans_model: Output[Model],
    kmeans_scaler: Output[Model],
    unsupervised_metrics: Output[Dataset],
) -> None:
    """Train unsupervised KMeans model and persist scaler/model/metrics artifacts."""
    import logging

    import joblib
    import pandas as pd
    from ml_models import unsupervised

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dataset_path = Path(ml_ready_data.path)
    model_path = Path(kmeans_model.path)
    scaler_path = Path(kmeans_scaler.path)
    metrics_path = Path(unsupervised_metrics.path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(dataset_path)
    scaler, model, metrics = unsupervised.train_clustering(dataframe)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
    ],
)
def score_top_products_op(
    enriched_products: Input[Dataset],
    ml_ready_data: Input[Dataset],
    supervised_model: Input[Model],
    kmeans_model: Input[Model],
    kmeans_scaler: Input[Model],
    top_k_products: Output[Dataset],
) -> None:
    """Score products with supervised + unsupervised signals and output top-k JSON."""
    import logging

    import joblib
    import pandas as pd
    from xgboost import XGBClassifier
    from ml_models import scoring

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    enriched_records = scoring.load_json_records(Path(enriched_products.path))
    enriched_df = pd.DataFrame(enriched_records)

    ml_ready_df = scoring.load_ml_ready(Path(ml_ready_data.path))
    if "reviews" not in ml_ready_df.columns and "review_count" in ml_ready_df.columns:
        ml_ready_df["reviews"] = pd.to_numeric(ml_ready_df["review_count"], errors="coerce").fillna(0)

    model = XGBClassifier()
    model.load_model(supervised_model.path)

    scaler = joblib.load(kmeans_scaler.path)
    kmeans = joblib.load(kmeans_model.path)

    supervised_features = scoring.prepare_supervised_features(ml_ready_df)
    ml_ready_df["top_product_probability"] = model.predict_proba(supervised_features)[:, 1]

    clustering_features = (
        ml_ready_df.loc[:, ["price", "rating", "reviews"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    scaled = scaler.transform(clustering_features)
    ml_ready_df["cluster_id"] = kmeans.predict(scaled)

    ml_ready_df["cluster_quality_score"] = scoring.apply_cluster_quality_signal(ml_ready_df)
    ml_ready_df["rating"] = pd.to_numeric(ml_ready_df["rating"], errors="coerce").fillna(0)
    ml_ready_df["reviews"] = pd.to_numeric(ml_ready_df["reviews"], errors="coerce").fillna(0)
    ml_ready_df["in_stock_flag"] = (
        ml_ready_df.get("stock_in_stock", pd.Series([0] * len(ml_ready_df))).astype(float)
    )

    ml_ready_df["final_score"] = (
        ml_ready_df["top_product_probability"] * 0.60
        + ml_ready_df["cluster_quality_score"] * 0.15
        + ml_ready_df["rating"] * 0.15
        + ml_ready_df["reviews"] * 0.05
        + ml_ready_df["in_stock_flag"] * 0.05
    )

    enriched_lookup = enriched_df.set_index("product_id").to_dict(orient="index") if not enriched_df.empty else {}

    ranked = ml_ready_df.sort_values("final_score", ascending=False).head(scoring.TOP_K).copy()
    ranked["product_id"] = ranked["product_id"].astype(str)

    output_records: list[dict] = []
    for _, row in ranked.iterrows():
        product_id = row["product_id"]
        base_record = enriched_lookup.get(product_id, {"product_id": product_id})

        enriched_record = {
            **base_record,
            "top_product_probability": float(row["top_product_probability"]),
            "cluster_id": int(row["cluster_id"]),
            "cluster_quality_score": float(row["cluster_quality_score"]),
            "final_score": float(row["final_score"]),
        }
        output_records.append(enriched_record)

    output_path = Path(top_k_products.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
