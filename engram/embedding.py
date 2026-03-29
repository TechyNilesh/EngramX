from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

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
