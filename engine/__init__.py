"""
engine/__init__.py — FAISS vector index for semantic search over cleaned reviews.
Uses a local sentence-transformers model (all-MiniLM-L6-v2) for fast, API-free embeddings.
Optionally falls back to Gemini API if configured. Builds a local FAISS index for cosine search.
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

# Local embedding model — no API key needed, no rate limits
LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality

_local_model = None  # cached singleton


def _get_local_model():
    """Lazy-load the sentence-transformers model (cached)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model '%s'...", LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        logger.info("Local embedding model loaded.")
    return _local_model


def _get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings using local sentence-transformers model.

    Uses all-MiniLM-L6-v2 (384-dim) — runs entirely on CPU, no API calls.
    Fast: embeds ~1800 texts in under 30 seconds on M-series Mac.
    """
    model = _get_local_model()

    # Truncate very long texts (model max is 256 tokens, but handles gracefully)
    truncated = [t[:2000] for t in texts]

    logger.info("Embedding %d texts with local model...", len(truncated))
    embeddings = model.encode(
        truncated,
        show_progress_bar=True,
        batch_size=128,
        normalize_embeddings=False,  # We normalize later for FAISS
    )

    return np.array(embeddings, dtype="float32")


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

    query_embedding = _get_embeddings([query])  # uses local model, no API
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

    return results
