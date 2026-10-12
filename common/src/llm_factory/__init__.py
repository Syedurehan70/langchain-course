"""One place that decides which chat and embedding model each project uses.

Models are configured per *role* in .env, as "provider:model":

    LLM_DEFAULT=groq:openai/gpt-oss-20b
    LLM_AGENT=groq:openai/gpt-oss-20b
    LLM_STRUCTURER=ollama:qwen2.5:7b
    EMBEDDINGS=ollama:nomic-embed-text

A role with no entry falls back to LLM_DEFAULT. Code asks for a role
("give me the agent model"); .env decides which provider and model fill it.

The factory never changes behaviour behind your back. Provider limits we
have hit ourselves are listed in KNOWN_LIMITS and exposed through
supports(); the calling code reads them and decides what to do.

This module does not load .env. Call load_dotenv() in your script first.
"""

import os

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

__all__ = ["get_llm", "get_embeddings", "model_spec", "supports", "KNOWN_LIMITS"]

FALLBACK_LLM = "groq:openai/gpt-oss-20b"
FALLBACK_EMBEDDINGS = "ollama:nomic-embed-text"

# Limits observed in this course, per provider. Anything not listed is
# assumed to work. Add an entry only after seeing the failure yourself.
KNOWN_LIMITS: dict[str, set[str]] = {
    # Groq rejects a request that combines tool calling with JSON-mode
    # structured output ("json mode cannot be combined with tool/function
    # calling"). Seen in 02_search_engine with openai/gpt-oss-20b.
    "groq": {"tools_with_structured_output"},
}


def model_spec(role: str = "default") -> tuple[str, str]:
    """Return (provider, model) configured for a role, e.g. ("groq", "openai/gpt-oss-20b")."""
    spec = (
        os.environ.get(f"LLM_{role.upper()}")
        or os.environ.get("LLM_DEFAULT")
        or FALLBACK_LLM
    )
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(
            f"LLM_{role.upper()}={spec!r} is not 'provider:model' "
            "(e.g. 'groq:openai/gpt-oss-20b' or 'ollama:qwen2.5:7b')"
        )
    return provider, model


def get_llm(role: str = "default", **kwargs) -> BaseChatModel:
    """Build the chat model configured for a role.

    Extra keyword arguments (temperature, max_tokens, ...) go straight to
    the provider's chat model class.
    """
    provider, model = model_spec(role)
    return init_chat_model(model, model_provider=provider, **kwargs)


def get_embeddings(**kwargs) -> Embeddings:
    """Build the embedding model configured in EMBEDDINGS."""
    spec = os.environ.get("EMBEDDINGS") or FALLBACK_EMBEDDINGS
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"EMBEDDINGS={spec!r} is not 'provider:model'")
    return init_embeddings(model, provider=provider, **kwargs)


def supports(role: str, feature: str) -> bool:
    """False if the role's provider has a known limit for this feature.

    Features currently tracked: "tools_with_structured_output".
    """
    provider, _ = model_spec(role)
    return feature not in KNOWN_LIMITS.get(provider, set())
