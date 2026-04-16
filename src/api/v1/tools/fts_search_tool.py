from src.core.database import get_connection

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
        ts_rank(
            to_tsvector('english', c.content),
            websearch_to_tsquery('english', %(query)s)
        ) AS fts_rank
    FROM chunks c
    JOIN documents d
        ON d.id = c.document_id
    WHERE to_tsvector('english', c.content)
          @@ websearch_to_tsquery('english', %(query)s)
    ORDER BY fts_rank DESC
    LIMIT %(k)s;
"""


def fts_search(query: str, k: int = 10) -> list[dict]:
    """
    Full-text keyword search against the chunks table.
    Best for codes, IDs, abbreviations, exact keywords.
    """
    print(f"[fts_search] query='{query}' k={k}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SQL, {"query": query, "k": k})
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
        "score": round(float(row.get("fts_rank", 0.0)), 4),
        "metadata": {
            "title": row.get("title") or row.get("filename"),
            "source": row.get("source_path"),
            "created_at": row.get("doc_created_at"),
            **base_metadata,
        },
    }
