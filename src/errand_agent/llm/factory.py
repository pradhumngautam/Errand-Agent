"""Pick the live model provider from the environment.

LLM_PROVIDER = claude_cli | openrouter | gemini. When unset: OpenRouter if OPENROUTER_API_KEY
is set, else Gemini if GEMINI_API_KEY is set, else the local Claude Code CLI (no key needed).
"""

from __future__ import annotations

import os

from errand_agent.llm.base import LLMClient


def make_live_llm(cache_dir: str | None = None) -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        provider = ("openrouter" if os.environ.get("OPENROUTER_API_KEY")
                    else "gemini" if os.environ.get("GEMINI_API_KEY") else "claude_cli")
    if provider == "claude_cli":
        from errand_agent.llm.claude_cli import ClaudeCLIClient
        return ClaudeCLIClient()
    if provider == "openrouter":
        from errand_agent.llm.openrouter import OpenRouterClient
        return OpenRouterClient()
    from errand_agent.llm.gemini import GeminiClient
    return GeminiClient(cache_path=f"{cache_dir}/.gemini_model" if cache_dir else None)
