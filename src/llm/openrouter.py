from __future__ import annotations

import os
from typing import Optional, Dict

from langchain_openai import ChatOpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_openrouter(
    model: str,
    temperature: float = 0.2,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    app_title: Optional[str] = None,
    referer: Optional[str] = None,
):
    """Return a LangChain ChatOpenAI configured for OpenRouter."""
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    resolved_base_url = (
        base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or DEFAULT_OPENROUTER_BASE_URL
    )

    headers: Dict[str, str] = {}
    if referer or os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = referer or os.getenv("OPENROUTER_REFERER", "")
    if app_title or os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = app_title or os.getenv("OPENROUTER_APP_TITLE", "")

    return ChatOpenAI(
        model=model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        temperature=temperature,
        default_headers=headers or None,
    )