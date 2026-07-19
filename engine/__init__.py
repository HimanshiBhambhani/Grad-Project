"""
engine/vectorstore.py — FAISS vector index for semantic search over cleaned reviews.
Uses OpenAI text-embedding-3-small to embed review text, then builds a local FAISS index.
Supports saving/loading the index to disk for persistence.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

INDEX_PATH = config.OUTPUT_DIR / "faiss_index.bin"
META_PATH = config.OUTPUT_DIR / "faiss_meta.pkl"
EMBED_BATCH_SIZE = 100  # OpenAI embedding batch size


def _get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings from OpenAI API in batches."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        # Truncate very long texts
        batch = [t[:8000] for t in batch]

        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )
        embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(embeddings)

        if (i + EMBED_BATCH_SIZE) % 500 == 0:
            logger.info("Embedded %d / %d texts...", i + EMBED_BATCH_SIZE, len(texts))

    return np.array(all_embeddings, dtype="float32")


def build_index(df: pd.DataFrame) -> tuple:
    """
    Build a FAISS index from the cleaned DataFrame.

    Args:
        df: DataFrame with at least 'Raw Content' column.

    Returns:
        (faiss_index, metadata_list) tuple.
    """
    import faiss

    texts = df["Raw Content"].tolist()
    logger.info("Building FAISS index for %d documents...", len(texts))

    # Get embeddings
    embeddings = _get_embeddings(texts)
    dimension = embeddings.shape[1]

    # Build FAISS index (L2 distance)
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine-like after normalization)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Store metadata for retrieval
    metadata = df.to_dict("records")

    # Save to disk
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("FAISS index built: %d vectors, %d dimensions. Saved to %s", index.ntotal, dimension, INDEX_PATH)
    return index, metadata


def load_index() -> tuple:
    """Load a previously saved FAISS index and metadata."""
    import faiss

    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    logger.info("Loaded FAISS index: %d vectors.", index.ntotal)
    return index, metadata


def search(query: str, index, metadata: list[dict], top_k: int = 15) -> list[dict]:
    """
    Semantic search: find the most relevant reviews for a query.

    Returns list of metadata dicts with added 'score' field.
    """
    import faiss

    query_embedding = _get_embeddings([query])
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

    return results
