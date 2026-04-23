"""Streamlit chat component backed by Gemini for BI interactions."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


def _load_env_file(project_root: Path) -> None:
    """Load .env values into environment when not already present."""
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


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model with API key from environment."""
    project_root = Path(__file__).resolve().parents[1]
    _load_env_file(project_root)

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing from environment variables.")

    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=google_api_key,
        temperature=0.2,
    )


def _build_data_summary(filtered_df: pd.DataFrame, max_rows: int = 100) -> str:
    """Build a JSON summary string from the currently filtered dashboard data."""
    safe_df = filtered_df.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(
            lambda value: value if not isinstance(value, (dict, list)) else str(value)
        )

    rows = safe_df.head(max_rows).to_dict(orient="records")
    return json.dumps(rows, ensure_ascii=False)


def _build_history_context(messages: list[dict[str, str]], max_turns: int = 6) -> str:
    """Build compact conversation context from the most recent chat turns."""
    recent_messages = messages[-max_turns:]
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in recent_messages
    )


def _extract_text_from_response(response: object) -> str:
    """Extract plain assistant text from AIMessage-like or dict-like responses."""
    content: object

    if hasattr(response, "content"):
        content = getattr(response, "content")
    elif isinstance(response, dict):
        content = response.get("text") or response.get("content") or ""
    else:
        content = response

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        nested = content.get("text") or content.get("content") or ""
        return str(nested).strip()

    if isinstance(content, list):
        extracted_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                extracted_parts.append(item)
                continue
            if isinstance(item, dict):
                if "text" in item and item["text"]:
                    extracted_parts.append(str(item["text"]))
                elif "content" in item and item["content"]:
                    extracted_parts.append(str(item["content"]))
        return "\n".join(part.strip() for part in extracted_parts if part).strip()

    return str(content).strip()


def _generate_assistant_response(user_message: str, filtered_df: pd.DataFrame) -> str:
    """Generate assistant response using Gemini grounded on filtered dashboard data."""
    llm = _get_llm()
    data_summary = _build_data_summary(filtered_df)
    history_context = _build_history_context(st.session_state.messages)

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
    response_text = _extract_text_from_response(response)
    return response_text or "I could not extract a readable response from the AI output."


def render_chat_interface(filtered_df: pd.DataFrame) -> None:
    """Render Streamlit chat UI with persistent history and Gemini responses."""
    st.subheader("AI Insights Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask about top products, prices, or clusters...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing filtered dataset..."):
            try:
                assistant_reply = _generate_assistant_response(user_prompt, filtered_df)
            except Exception as exc:  # noqa: BLE001
                assistant_reply = (
                    "I could not complete the request because the AI service is unavailable. "
                    f"Details: {exc}"
                )
        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
