"""Flask backend for Smart eCommerce Intelligence Dashboard."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request
from langchain_deepseek import ChatDeepSeek

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "top_k_products.json"
RULES_PATH = PROJECT_ROOT / "data" / "processed" / "association_rules.json"

def _load_env_file() -> None:
    """Load .env values into environment when not already present."""
    env_path = PROJECT_ROOT / ".env"
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

_load_env_file()

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_json(DATA_PATH)
    numeric_columns = ["price", "promotional_price", "rating", "review_count", "final_score", "pca_1", "pca_2"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    return df

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered_df = df.copy()
    categories = filters.get("categories", [])
    stock_status = filters.get("stock_status", [])
    
    # Robust column matching: check for standardized_category OR category
    category_col = "standardized_category" if "standardized_category" in df.columns else "category"
    if categories and category_col in df.columns:
        filtered_df = filtered_df[filtered_df[category_col].astype(str).isin(categories)]
    
    if stock_status and "stock_status" in df.columns:
        filtered_df = filtered_df[filtered_df["stock_status"].astype(str).isin(stock_status)]
        
    return filtered_df

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/data", methods=["POST"])
def get_dashboard_data():
    filters = request.json or {}
    df = load_data()
    if df.empty:
        return jsonify({"error": "No data available"}), 404
        
    # Get options for filters before filtering
    category_col = "standardized_category" if "standardized_category" in df.columns else "category"
    filter_options = {
        "categories": get_unique_values(df, category_col),
        "stock_status": get_unique_values(df, "stock_status")
    }

    filtered_df = apply_filters(df, filters)
    
    # KPIs
    kpis = {
        "total_products": len(filtered_df),
        "avg_price": float(filtered_df["price"].mean()) if not filtered_df["price"].empty and not pd.isna(filtered_df["price"].mean()) else 0.0,
        "avg_score": float(filtered_df["final_score"].mean()) if not filtered_df["final_score"].empty and not pd.isna(filtered_df["final_score"].mean()) else 0.0,
    }

    # Association rules
    rules = []
    if RULES_PATH.exists():
        try:
            all_rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
            rules = all_rules[:15]
        except Exception:
            pass

    # Clean data for frontend table and charts
    display_df = filtered_df.copy()
    display_df = display_df.fillna("N/A")
    # convert nested dicts/lists to string
    for col in display_df.columns:
        if display_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            display_df = display_df.drop(columns=[col])
            
    records = display_df.to_dict(orient="records")

    return jsonify({
        "kpis": kpis,
        "filter_options": filter_options,
        "data": records,
        "association_rules": rules
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")
    filters = data.get("filters", {})
    history = data.get("history", [])
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
        
    df = load_data()
    filtered_df = apply_filters(df, filters)
    
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        return jsonify({"reply": "Error: DEEPSEEK_API_KEY is not set in environment."})

    try:
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.2,
            max_retries=3
        )
        
        # Build context
        safe_df = filtered_df.copy()
        for col in safe_df.columns:
            safe_df[col] = safe_df[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
        
        data_summary = json.dumps(safe_df.head(100).to_dict(orient="records"), ensure_ascii=False)
        history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]])
        
        prompt = (
            "You are a helpful eCommerce BI assistant. "
            "Answer the user's question based ONLY on the following dataset of top products: "
            f"{data_summary}\n\n"
            "Keep answers concise, factual, and reference only visible data. "
            "If the answer is not in the data, say you cannot find it in the filtered dataset.\n\n"
            f"Conversation context:\n{history_context}\n\n"
            f"User question: {user_message}"
        )
        
        response = llm.invoke(prompt)
        reply = response.content if hasattr(response, "content") else str(response)
        
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"AI service error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
