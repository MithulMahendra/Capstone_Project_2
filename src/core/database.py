import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv(override=True)

PG_CONN = os.getenv("PG_CONNECTION_STRING")


def get_connection():
    return psycopg.connect(PG_CONN, row_factory=dict_row)


def init_db():
    """Create the documents and chunks tables if they don't exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Required extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    filename    TEXT NOT NULL,
                    title       TEXT,
                    source_path TEXT,
                    file_hash   TEXT UNIQUE,
                    page_count  INT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id    UUID REFERENCES documents(id) ON DELETE CASCADE,
                    content        TEXT NOT NULL,
                    embedding      vector(1536),
                    chunk_type     VARCHAR(50) NOT NULL,
                    source_page    INT,
                    document_name  VARCHAR(500),
                    metadata       JSONB DEFAULT '{}',
                    created_at     TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Vector index for similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            # Full-text search index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_fts_idx
                ON chunks USING GIN (to_tsvector('english', content));
            """)

            # Useful secondary indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                ON chunks(document_id);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx
                ON chunks(chunk_type);
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_source_page_idx
                ON chunks(source_page);
            """)

            conn.commit()

    print("Database initialised successfully.")
