from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

from .schema import MemoryRecord

EMBEDDING_DIM = 64
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def embed_text(text: str, *, dims: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * max(8, dims)
    tokens = _tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        hashed = _hash_token(token)
        index = hashed % len(vector)
        sign = -1.0 if (hashed >> 1) & 1 else 1.0
        weight = 1.0 + (len(token) / 10.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def embed_record(record: MemoryRecord, *, dims: int = EMBEDDING_DIM) -> list[float]:
    if record.embedding is not None:
        return list(record.embedding)
    payload = f"{record.content} {record.payload}" if record.payload is not None else record.content
    return embed_text(payload, dims=dims)


class HashEmbedder:
    """Deterministic dependency-free embedder for local/dev hybrid retrieval."""

    def __init__(self, dims: int = EMBEDDING_DIM) -> None:
        self.dims = max(8, dims)

    def embed(self, text: str) -> list[float]:
        return embed_text(text, dims=self.dims)


def _dependency_available(module_name: str) -> bool:
    """Check whether an optional dependency is importable."""
    import importlib

    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


class OpenAIEmbedder:
    """Embedder using the OpenAI SDK format — works with any compatible provider.

    Supports OpenAI, Gemini, Cohere, Mistral, Together AI, and any provider
    that exposes an OpenAI-compatible ``/embeddings`` endpoint via ``base_url``.

    Examples::

        # OpenAI (default)
        embedder = OpenAIEmbedder(model="text-embedding-3-small")

        # Google Gemini
        embedder = OpenAIEmbedder(
            model="gemini-embedding-001",
            api_key="GEMINI_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # Cohere
        embedder = OpenAIEmbedder(
            model="embed-english-v3.0",
            api_key="COHERE_KEY",
            base_url="https://api.cohere.ai/compatibility/v1",
        )

        # Mistral
        embedder = OpenAIEmbedder(
            model="mistral-embed",
            api_key="MISTRAL_KEY",
            base_url="https://api.mistral.ai/v1",
        )

        # Together AI
        embedder = OpenAIEmbedder(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            api_key="TOGETHER_KEY",
            base_url="https://api.together.xyz/v1",
        )
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        if not _dependency_available("openai"):
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbedder. "
                "Install it with: pip install openai"
            )
        import openai

        self.model = model
        self.dimensions = dimensions
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {"input": text, "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        return list(response.data[0].embedding)


class LiteLLMEmbedder:
    """Embedder backed by LiteLLM — unified interface for 100+ providers.

    Supports OpenAI, Gemini, Cohere, Mistral, Bedrock, Azure, Voyage,
    and any provider supported by litellm.

    Examples::

        # OpenAI
        embedder = LiteLLMEmbedder(model="text-embedding-3-small")

        # Cohere
        embedder = LiteLLMEmbedder(model="cohere/embed-english-v3.0")

        # Gemini
        embedder = LiteLLMEmbedder(model="gemini/gemini-embedding-001")

        # Bedrock
        embedder = LiteLLMEmbedder(model="bedrock/amazon.titan-embed-text-v1")
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        if not _dependency_available("litellm"):
            raise ImportError(
                "The 'litellm' package is required for LiteLLMEmbedder. "
                "Install it with: pip install litellm"
            )
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        import litellm

        kwargs: dict[str, Any] = {"model": self.model, "input": [text]}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = litellm.embedding(**kwargs)
        return list(response.data[0]["embedding"])


class SentenceTransformerEmbedder:
    """Embedder backed by the ``sentence-transformers`` library (local models)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _dependency_available("sentence_transformers"):
            raise ImportError(
                "The 'sentence-transformers' package is required for "
                "SentenceTransformerEmbedder. Install it with: "
                "pip install sentence-transformers"
            )
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()


def create_embedder(provider: str = "hash", **kwargs: Any) -> Embedder:
    """Factory that returns the appropriate :class:`Embedder` implementation.

    Parameters
    ----------
    provider:
        One of ``"hash"`` (default), ``"openai"`` (also works with Gemini,
        Cohere, Mistral, Together AI via ``base_url``), ``"litellm"`` (100+
        providers), or ``"sentence_transformers"``.
    **kwargs:
        Forwarded to the chosen embedder constructor.
    """
    if provider == "hash":
        return HashEmbedder(**kwargs)
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    if provider == "litellm":
        return LiteLLMEmbedder(**kwargs)
    if provider in ("sentence_transformers", "sentence-transformers", "st"):
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(size):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
