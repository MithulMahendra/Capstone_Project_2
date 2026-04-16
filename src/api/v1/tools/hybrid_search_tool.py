"""
Hybrid search: vector + FTS merged via Reciprocal Rank Fusion (RRF).
score(chunk) = sum(1 / (rank + 60)) across both result lists.
Best for short or ambiguous queries.
"""

from src.api.v1.tools.vector_search_tool import vector_search
from src.api.v1.tools.fts_search_tool import fts_search

_RRF_K = 60


def hybrid_search(query: str, k: int = 10) -> list[dict]:
    """
    Hybrid search combining vector and FTS results via RRF.
    Best for short or ambiguous queries.
    """
    print(f"[hybrid_search] query='{query}' k={k}")

    vector_docs = vector_search(query=query, k=k)
    fts_docs = fts_search(query=query, k=k)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc["chunk_id"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key] = doc

    for rank, doc in enumerate(fts_docs):
        key = doc["chunk_id"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # If the chunk already exists from vector search, keep the richer version if needed
        if key not in chunk_map:
            chunk_map[key] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []
    for key, score in ranked[:k]:
        chunk = chunk_map[key].copy()
        chunk["score"] = round(score, 4)
        final_chunks.append(chunk)

    return final_chunks
