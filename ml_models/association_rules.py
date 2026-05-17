"""Extract association rules using apriori to simulate and analyze buying behaviors."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def simulate_transactions(dataframe: pd.DataFrame, num_baskets: int = 1000) -> List[List[str]]:
    """Simulate realistic transactions based on categories.

    Concentrates purchases on popular products to avoid uniform dilution.
    """
    categories = dataframe["category"].dropna().unique().tolist()
    if not categories:
        return []

    # FIX MLOPS: Pour chaque catégorie, on ne garde qu'un sous-ensemble (ex: les 8 premiers produits)
    # pour simuler des "best-sellers" et créer de vraies corrélations répétables
    cat_to_products = {}
    for cat, group in dataframe.groupby("category"):
        product_list = group["product_id"].list() if hasattr(group["product_id"], "list") else list(group["product_id"])
        cat_to_products[cat] = product_list[:8]  # Concentre les règles sur un pool restreint par catégorie

    transactions: List[List[str]] = []
    for _ in range(num_baskets):
        basket_size = random.randint(2, 4)
        basket: set[str] = set()

        base_cat = random.choice(categories)
        if cat_to_products.get(base_cat) and len(cat_to_products[base_cat]) > 0:
            basket.add(str(random.choice(cat_to_products[base_cat])))

        for _ in range(basket_size - 1):
            if random.random() < 0.7:  # 70% de chance de rester dans la même catégorie logique
                cat = base_cat
            else:
                cat = random.choice(categories)

            if cat_to_products.get(cat) and len(cat_to_products[cat]) > 0:
                basket.add(str(random.choice(cat_to_products[cat])))
        transactions.append(list(basket))

    return transactions


def run() -> None:
    """Run association rules extraction and save to JSON."""
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / "ml_ready_data.csv"
    output_path = project_root / "data" / "processed" / "association_rules.json"

    try:
        dataframe = pd.read_csv(input_path)
        if dataframe.empty:
            raise ValueError("ML-ready dataset is empty.")
        if "product_id" not in dataframe.columns or "category" not in dataframe.columns:
            raise ValueError("Missing required columns in dataset.")

        id_to_name = dataframe.set_index("product_id")["name"].to_dict() if "name" in dataframe.columns else {}

        transactions = simulate_transactions(dataframe, num_baskets=1000)

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)

        # CORRECTION DES SEUILS : On passe le min_support à 0.005 (0.5%) car le catalogue est plus grand
        frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)
        if frequent_itemsets.empty:
            LOGGER.warning("No frequent itemsets found.")
            output_path.write_text("[]", encoding="utf-8")
            return

        # Ajustement des métriques de validation
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
        rules = rules[rules["lift"] > 1.1]

        rules_list: List[Dict[str, Any]] = []
        for _, row in rules.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            
            ant_names = [str(id_to_name.get(item, id_to_name.get(int(item) if str(item).isdigit() else item, item))) for item in antecedents]
            con_names = [str(id_to_name.get(item, id_to_name.get(int(item) if str(item).isdigit() else item, item))) for item in consequents]

            rules_list.append(
                {
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "antecedents_names": ant_names,
                    "consequents_names": con_names,
                    "support": float(row["support"]),
                    "confidence": float(row["confidence"]),
                    "lift": float(row["lift"]),
                }
            )

        rules_list.sort(key=lambda x: x["lift"], reverse=True)
        output_path.write_text(json.dumps(rules_list[:50], indent=2), encoding="utf-8")
        LOGGER.info("association_rules_completed", extra={"num_rules": len(rules_list)})

    except Exception as exc:
        LOGGER.error("association_rules_failed", extra={"error": str(exc)})
        raise


if __name__ == "__main__":
    configure_logging()
    run()