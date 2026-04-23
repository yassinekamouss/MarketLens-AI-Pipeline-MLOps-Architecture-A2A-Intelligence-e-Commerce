"""Streamlit BI dashboard for top product intelligence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.chat_component import render_chat_interface

st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    "<style> .stApp { background-color: #0E1117; color: white; } </style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0E1117;
            color: #f4f6f8;
        }
        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #f4f6f8;
            letter-spacing: 0.02em;
            margin-bottom: 0.25rem;
        }
        .dashboard-subtitle {
            color: #c7d4df;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_top_products_df() -> pd.DataFrame:
    """Load top-k products from processed dataset into a DataFrame."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "top_k_products.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_json(data_path)

    numeric_columns = ["price", "promotional_price", "rating", "review_count", "final_score"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")

    return df


def format_metric(value: float | None, precision: int = 2, prefix: str = "") -> str:
    """Format metric values with safe fallback for missing values."""
    if value is None or pd.isna(value):
        return f"{prefix}{0:.{precision}f}"
    return f"{prefix}{value:,.{precision}f}"


def safe_mean(df: pd.DataFrame, column: str) -> float:
    """Return safe mean for a column, defaulting to 0.0 when unavailable."""
    if column not in df.columns:
        return 0.0
    series = pd.to_numeric(df[column], errors="coerce")
    if series.dropna().empty:
        return 0.0
    return float(series.mean())


def build_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a UI-safe table by dropping nested columns and filling nulls."""
    display_df = df.copy()

    unnamed_columns = [column for column in display_df.columns if str(column).startswith("Unnamed")]

    nested_columns: list[str] = []
    for column in display_df.columns:
        sample_series = display_df[column].dropna()
        if sample_series.empty:
            continue
        if sample_series.map(lambda value: isinstance(value, (dict, list))).any():
            nested_columns.append(column)

    columns_to_drop = sorted(set(unnamed_columns + nested_columns))
    if columns_to_drop:
        display_df = display_df.drop(columns=columns_to_drop)

    numeric_columns = display_df.select_dtypes(include=["number"]).columns
    text_columns = display_df.select_dtypes(exclude=["number"]).columns

    if len(numeric_columns) > 0:
        display_df[numeric_columns] = display_df[numeric_columns].fillna(0)
    if len(text_columns) > 0:
        display_df[text_columns] = display_df[text_columns].fillna("N/A")

    return display_df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters and return filtered DataFrame."""
    st.sidebar.header("Filters")

    filtered_df = df.copy()

    if "standardized_category" in df.columns:
        categories = sorted(df["standardized_category"].dropna().astype(str).unique().tolist())
        selected_categories = st.sidebar.multiselect(
            "Category",
            options=categories,
        )
        if selected_categories:
            filtered_df = filtered_df[
                filtered_df["standardized_category"].astype(str).isin(selected_categories)
            ]

    if "stock_status" in df.columns:
        stock_values = sorted(df["stock_status"].dropna().astype(str).unique().tolist())
        selected_stock = st.sidebar.multiselect(
            "Stock Status",
            options=stock_values,
        )
        if selected_stock:
            filtered_df = filtered_df[filtered_df["stock_status"].astype(str).isin(selected_stock)]

    return filtered_df


def render_kpis(df: pd.DataFrame) -> None:
    """Render first row KPI metrics."""
    avg_price = safe_mean(df, "price")
    avg_score = safe_mean(df, "final_score")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", f"{len(df):,}")
    col2.metric("Average Price", format_metric(avg_price, prefix="$"))
    col3.metric("Average Final Score", format_metric(avg_score, precision=3))


def render_scatter_plot(df: pd.DataFrame) -> None:
    """Render second row interactive scatter chart (Price vs Final Score)."""
    st.subheader("Price vs Product Performance")

    filtered_df = df.copy()
    required_columns = ["price", "final_score", "standardized_category"]
    missing_columns = [column for column in required_columns if column not in filtered_df.columns]
    if missing_columns:
        st.info(
            "Scatter plot requires columns: "
            + ", ".join(required_columns)
            + ". Missing: "
            + ", ".join(missing_columns)
        )
        return

    filtered_df["price"] = pd.to_numeric(filtered_df["price"], errors="coerce")
    filtered_df["final_score"] = pd.to_numeric(filtered_df["final_score"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["price", "final_score"])
    if filtered_df.empty:
        st.info("No valid points are available for scatter plotting after null filtering.")
        return

    hover_columns = [column for column in ["name", "short_summary"] if column in filtered_df.columns]

    fig = px.scatter(
        filtered_df,
        x="price",
        y="final_score",
        color="standardized_category",
        hover_data=hover_columns,
        template="plotly_dark",
        labels={"final_score": "Final Score"},
        title="Interactive Product Positioning by Price and Score",
    )

    fig.update_traces(
        marker=dict(size=15, opacity=0.8, line=dict(width=1, color="DarkSlateGrey"))
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.7)",
    )

    st.plotly_chart(fig, use_container_width=True)


st.markdown('<div class="dashboard-title">Smart eCommerce Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Top-K products analytics, filtering, and explainable insights.</div>',
    unsafe_allow_html=True,
)

try:
    data_df = load_top_products_df()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Failed to load processed dataset: {exc}")
    st.stop()

filtered_data_df = apply_filters(data_df)

if filtered_data_df.empty:
    st.warning("No products match the selected filters. Adjust filters in the sidebar.")
    st.stop()

render_kpis(filtered_data_df)
st.divider()
render_scatter_plot(filtered_data_df)
st.divider()
st.subheader("Top Products Data")
display_data_df = build_display_dataframe(filtered_data_df)
st.dataframe(display_data_df, use_container_width=True)
st.divider()
render_chat_interface(filtered_data_df)
