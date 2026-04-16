from dotenv import load_dotenv

from src.core.database import get_connection
from src.core.helper import get_embedding_model

load_dotenv(override=True)

_SQL = """
    SELECT
        c.id::text          AS chunk_id,
        c.content,
        c.chunk_type,
        c.source_page,
        c.document_name,
        c.metadata,
        d.filename,
        d.title,
        d.source_path,
        d.created_at::text  AS doc_created_at,
        1 - (c.embedding <=> %(embedding)s::vector) AS score
    FROM chunks c
    JOIN documents d
        ON d.id = c.document_id
    ORDER BY c.embedding <=> %(embedding)s::vector
    LIMIT %(k)s;
"""


def vector_search(query: str, k: int = 10) -> list[dict]:
    """
    Semantic vector search against the chunks table.
    Best for natural-language / conceptual questions.
    """
    print(f"[vector_search] query='{query}' k={k}")

    embedder = get_embedding_model()
    query_vec = embedder.embed_query(query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"embedding": query_vec, "k": k})
            rows = cur.fetchall()

    return [_format(row) for row in rows]


def _format(row: dict) -> dict:
    base_metadata = row.get("metadata") or {}

    return {
        "chunk_id": row["chunk_id"],
        "content": row["content"],
        "chunk_type": row.get("chunk_type", "text"),
        "source_page": row.get("source_page"),
        "document_name": row.get("document_name") or row.get("filename") or "N/A",
        "score": round(float(row.get("score", 0.0)), 4),
        "metadata": {
            "title": row.get("title") or row.get("filename"),
            "source": row.get("source_path"),
            "created_at": row.get("doc_created_at"),
            **base_metadata,
        },
    }
