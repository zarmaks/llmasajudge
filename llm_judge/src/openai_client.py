from __future__ import annotations

"""Tiny wrapper around the **Mistral AI** Python SDK so that the rest of the
codebase is agnostic to the underlying provider.  We keep the exported class
name `OpenAIClient` so the rest of the code (Judge, CLI, tests) remains
unchanged.  Internally it calls Mistral's chat‑completions API."""

import os
from typing import Any

try:
    from mistralai import Mistral
except ImportError as e:  # pragma: no cover
    raise ImportError("mistralai package not installed — add it to requirements.txt") from e


# -----------------------------------------------------------------------------
# Helper to fetch the API key
# -----------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY environment variable not set. See .env.example")
    return key


# -----------------------------------------------------------------------------
# LLM wrapper (same public API as before)
# -----------------------------------------------------------------------------

class OpenAIClient:  # name kept for backward compatibility
    """Very thin abstraction over the Mistral chat‑completions endpoint."""

    def __init__(
        self, model: str = "mistral-small-latest", temperature: float = 0.0
    ) -> None:
        self._client = Mistral(api_key=_get_api_key())
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------
    def chat(
        self, *, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> str:
        """Return the assistant message text only (first choice)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Type ignores for Mistral client - external library
        response = self._client.chat.complete(  # type: ignore
            model=self.model,
            messages=messages,  # type: ignore
            temperature=self.temperature,
            **kwargs,
        )
        return response.choices[0].message.content.strip()  # type: ignore
