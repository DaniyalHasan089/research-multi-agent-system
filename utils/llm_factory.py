"""
LLM Factory — returns the correct LangChain chat model based on the
LLM_PROVIDER environment variable. All three LLM agents import from
here so that switching providers requires only a single .env change.

Supported providers:
  - gemini  → Google Gemini Flash (default)
  - openai  → OpenAI GPT-4o-mini (lightweight) or GPT-4o (heavy tasks)
"""

import os
from dotenv import load_dotenv

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def get_llm(task: str = "default"):
    """
    Return the appropriate LangChain chat model for the given task.

    Args:
        task: Hint about the calling agent. Used to select model size.
              - "light"  → fast/cheap model (Query Planner)
              - "heavy"  → more capable model (Report Writer, Analyzer)
              - "default" → same as "light"

    Returns:
        A LangChain BaseChatModel instance.
    """
    if _PROVIDER == "gemini":
        return _get_gemini(task)
    elif _PROVIDER == "openai":
        return _get_openai(task)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{_PROVIDER}'. "
            "Set LLM_PROVIDER to 'gemini' or 'openai' in your .env file."
        )


def _get_gemini(task: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in the environment.")

    # Use the same flash model for all tasks — it's fast and cost-effective
    model_name = "gemini-3.1-flash-lite"

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
    )


def _get_openai(task: str):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    # Light tasks → gpt-4o-mini (fast, cheap)
    # Heavy tasks → gpt-4o (better reasoning for report writing)
    model_name = "gpt-4o" if task == "heavy" else "gpt-4o-mini"

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        temperature=0.2,
    )


def extract_text(content) -> str:
    """
    Safely extract a plain string from an LLM response content value.

    Newer versions of langchain-google-genai may return content as a list
    of dicts (e.g. [{'type': 'text', 'text': '...'}]) instead of a plain
    string. This helper normalises both formats.

    Args:
        content: response.content from any LangChain chat model.

    Returns:
        A plain stripped string.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content).strip()
