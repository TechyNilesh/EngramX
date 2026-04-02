"""
Engram Embeddings Example
=========================

This example demonstrates Engram's pluggable embedding system:

1. Using the default HashEmbedder (zero dependencies)
2. Creating embeddings and computing cosine similarity
3. Configuring the OpenAI embedder (and compatible providers)
4. Using base_url for Gemini, Cohere, Mistral, Together AI
5. LiteLLM embedder (100+ providers via a single package)
6. SentenceTransformer embedder (local models)
7. Passing a custom embedder to MemoryClient
8. Running a working search to see hybrid retrieval scores

The HashEmbedder sections are fully runnable with no extra dependencies.
Other providers are shown as commented examples -- uncomment and supply
your API keys to try them.

Run this file directly:

    python examples/05_embeddings.py
"""

import asyncio

from engramx import MemoryClient
from engramx.embedding import (
    HashEmbedder,
    cosine_similarity,
    create_embedder,
    embed_text,
)


# ---------------------------------------------------------------------------
# 1. HashEmbedder basics -- deterministic, zero dependencies
# ---------------------------------------------------------------------------

def hash_embedder_demo() -> None:
    """Show how the built-in HashEmbedder works."""

    embedder = HashEmbedder()  # default 64 dimensions
    print("=== HashEmbedder (default, 64-dim) ===")

    vec = embedder.embed("The quick brown fox jumps over the lazy dog")
    print(f"Vector length : {len(vec)}")
    print(f"First 8 values: {[round(v, 4) for v in vec[:8]]}")

    # You can also use the module-level helper directly.
    vec2 = embed_text("The quick brown fox jumps over the lazy dog")
    assert vec == vec2, "embed_text and HashEmbedder.embed produce the same output"
    print("embed_text() matches HashEmbedder.embed() -- deterministic!\n")

    # Custom dimensionality (minimum 8).
    big = HashEmbedder(dims=128)
    print(f"128-dim vector length: {len(big.embed('hello'))}")

    # The factory function works too.
    factory_embedder = create_embedder("hash", dims=32)
    print(f"Factory-created 32-dim vector: {len(factory_embedder.embed('hello'))}\n")


# ---------------------------------------------------------------------------
# 2. Cosine similarity -- comparing embeddings
# ---------------------------------------------------------------------------

def similarity_demo() -> None:
    """Embed several sentences and compare them pairwise."""

    print("=== Cosine Similarity ===")
    embedder = HashEmbedder()

    sentences = [
        "The cat sat on the mat",
        "A kitten rested on the rug",
        "Python is a programming language",
        "The cat chased the mouse",
    ]

    vectors = {s: embedder.embed(s) for s in sentences}

    # Compare every pair.
    for i, a in enumerate(sentences):
        for b in sentences[i + 1 :]:
            sim = cosine_similarity(vectors[a], vectors[b])
            print(f"  sim({a!r:.40}, {b!r:.40}) = {sim:.4f}")

    # Identical text always returns 1.0.
    same = cosine_similarity(vectors[sentences[0]], vectors[sentences[0]])
    print(f"\n  Self-similarity: {same:.4f}  (always 1.0 for identical text)\n")


# ---------------------------------------------------------------------------
# 3 & 4. OpenAI embedder + compatible providers (commented)
# ---------------------------------------------------------------------------

def openai_embedder_examples() -> None:
    """Commented examples for OpenAI and OpenAI-compatible providers.

    Uncomment the provider you want to use and set your API key.
    Requires: pip install openai
    """

    # --- OpenAI (default) ---------------------------------------------------
    # from engramx.embedding import OpenAIEmbedder
    #
    # embedder = OpenAIEmbedder(model="text-embedding-3-small")
    # vec = embedder.embed("Hello, world!")
    # print(f"OpenAI embedding dim: {len(vec)}")
    #
    # # You can also request a specific output dimensionality.
    # embedder_256 = OpenAIEmbedder(
    #     model="text-embedding-3-small",
    #     dimensions=256,
    # )
    # vec_256 = embedder_256.embed("Hello, world!")
    # print(f"OpenAI 256-dim embedding: {len(vec_256)}")

    # --- Google Gemini (via base_url) ----------------------------------------
    # embedder = OpenAIEmbedder(
    #     model="gemini-embedding-001",
    #     api_key="YOUR_GEMINI_API_KEY",
    #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    # )

    # --- Cohere (via base_url) -----------------------------------------------
    # embedder = OpenAIEmbedder(
    #     model="embed-english-v3.0",
    #     api_key="YOUR_COHERE_API_KEY",
    #     base_url="https://api.cohere.ai/compatibility/v1",
    # )

    # --- Mistral (via base_url) ----------------------------------------------
    # embedder = OpenAIEmbedder(
    #     model="mistral-embed",
    #     api_key="YOUR_MISTRAL_API_KEY",
    #     base_url="https://api.mistral.ai/v1",
    # )

    # --- Together AI (via base_url) ------------------------------------------
    # embedder = OpenAIEmbedder(
    #     model="togethercomputer/m2-bert-80M-8k-retrieval",
    #     api_key="YOUR_TOGETHER_API_KEY",
    #     base_url="https://api.together.xyz/v1",
    # )

    # --- Using the factory ---------------------------------------------------
    # from engramx.embedding import create_embedder
    #
    # embedder = create_embedder(
    #     "openai",
    #     model="text-embedding-3-small",
    #     base_url="https://api.mistral.ai/v1",
    #     api_key="YOUR_KEY",
    # )

    print("=== OpenAI / Compatible Providers ===")
    print("  (Skipped -- uncomment and add your API key to try.)\n")


# ---------------------------------------------------------------------------
# 5. LiteLLM embedder (commented)
# ---------------------------------------------------------------------------

def litellm_embedder_examples() -> None:
    """Commented examples for the LiteLLM embedder.

    LiteLLM provides a unified interface for 100+ providers.
    Requires: pip install litellm
    """

    # from engramx.embedding import LiteLLMEmbedder
    #
    # # OpenAI via LiteLLM
    # embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    #
    # # Cohere via LiteLLM (note the "cohere/" prefix)
    # embedder = LiteLLMEmbedder(model="cohere/embed-english-v3.0")
    #
    # # Gemini via LiteLLM
    # embedder = LiteLLMEmbedder(model="gemini/gemini-embedding-001")
    #
    # # Bedrock via LiteLLM
    # embedder = LiteLLMEmbedder(model="bedrock/amazon.titan-embed-text-v1")
    #
    # # With explicit API key and reduced dimensions
    # embedder = LiteLLMEmbedder(
    #     model="text-embedding-3-small",
    #     api_key="YOUR_OPENAI_KEY",
    #     dimensions=256,
    # )
    #
    # # Factory shorthand
    # from engramx.embedding import create_embedder
    # embedder = create_embedder("litellm", model="cohere/embed-english-v3.0")

    print("=== LiteLLM Embedder ===")
    print("  (Skipped -- pip install litellm, then uncomment to try.)\n")


# ---------------------------------------------------------------------------
# 6. SentenceTransformer embedder (commented)
# ---------------------------------------------------------------------------

def sentence_transformer_examples() -> None:
    """Commented examples for local sentence-transformers models.

    Runs entirely on your machine -- no API keys needed, but the first
    run downloads the model weights.
    Requires: pip install sentence-transformers
    """

    # from engramx.embedding import SentenceTransformerEmbedder
    #
    # # Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality)
    # embedder = SentenceTransformerEmbedder()
    # vec = embedder.embed("Local embeddings are fast!")
    # print(f"SentenceTransformer dim: {len(vec)}")
    #
    # # Use a different model
    # embedder = SentenceTransformerEmbedder(model_name="all-mpnet-base-v2")
    #
    # # Factory shorthand (accepts "sentence_transformers", "sentence-transformers", or "st")
    # from engramx.embedding import create_embedder
    # embedder = create_embedder("st", model_name="all-MiniLM-L6-v2")

    print("=== SentenceTransformer Embedder ===")
    print("  (Skipped -- pip install sentence-transformers, then uncomment to try.)\n")


# ---------------------------------------------------------------------------
# 7 & 8. Passing an embedder to MemoryClient + hybrid search demo
# ---------------------------------------------------------------------------

async def memory_client_search_demo() -> None:
    """Store memories and search them, showing how the embedder feeds into
    hybrid retrieval scoring."""

    print("=== MemoryClient with HashEmbedder -- Hybrid Search ===")

    # Pass a custom embedder when constructing the client.
    embedder = HashEmbedder(dims=64)
    client = MemoryClient(driver="memory", embedder=embedder)

    # Add a handful of memories on different topics.
    memories = [
        ("episodic", "user", "Deployed the new authentication service to production."),
        ("episodic", "user", "Had lunch with the backend team and discussed caching."),
        ("semantic", "global", "Redis supports sorted sets, streams, and pub/sub."),
        ("semantic", "global", "OAuth 2.0 uses access tokens and refresh tokens."),
        ("procedural", "agent", "Before deploying, always run the integration tests."),
    ]
    for mem_type, scope, content in memories:
        await client.add(type=mem_type, scope=scope, content=content)

    # Search for something related to deployment.
    query = "deployment and authentication"
    results = await client.search(query, top_k=5)

    print(f"\n  Query: {query!r}\n")
    print(f"  {'Score':>7}  {'Lexical':>7}  {'Importance':>10}  {'Decay':>6}  Content")
    print(f"  {'-----':>7}  {'-------':>7}  {'----------':>10}  {'-----':>6}  -------")
    for item in results:
        print(
            f"  {item.score:7.4f}  {item.lexical_score:7.4f}  "
            f"{item.importance_component:10.4f}  {item.decay_component:6.4f}  "
            f"{item.record.content[:60]}"
        )

    # Show the raw embedding similarity between the query and the top result.
    if results:
        top = results[0]
        query_vec = embedder.embed(query)
        record_vec = embedder.embed(top.record.content)
        raw_sim = cosine_similarity(query_vec, record_vec)
        print(f"\n  Raw cosine similarity (query vs top result): {raw_sim:.4f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fully runnable sections (HashEmbedder, no extra deps).
    hash_embedder_demo()
    similarity_demo()

    # Commented provider sections (print a skip message).
    openai_embedder_examples()
    litellm_embedder_examples()
    sentence_transformer_examples()

    # Async search demo using HashEmbedder + MemoryClient.
    asyncio.run(memory_client_search_demo())
