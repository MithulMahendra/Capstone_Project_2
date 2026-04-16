import os
import hashlib
from io import BytesIO
from pathlib import Path

from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.core.database import get_connection
from src.core.helper import get_embedding_model
from src.ingestion.chunking import parse_and_chunk

load_dotenv(override=True)


def _file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _doc_already_ingested(conn, file_hash: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
        return cur.fetchone() is not None


def _save_image(pil_image, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    pil_image.save(output_path, format="PNG")
    return str(output_path)


def _describe_image_with_gemini(pil_image, captions: list[str]) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    caption_hint = " | ".join(captions).strip()

    prompt = f"""
        You are analyzing an image extracted from a PDF for a RAG / semantic search pipeline.

        Task:
        1. If this is a corporate logo, watermark, repeated brand image, decorative header/footer graphic,
        icon-only mark, separator, background ornament, or any purely decorative/non-informational image,
        return exactly:
        SKIP_IMAGE

        2. Otherwise, if the image contains meaningful content (such as charts, graphs, infographics,
        annotated diagrams, screenshots, floating text boxes, tables-as-images, or process visuals),
        describe it clearly and briefly.

        Existing PDF caption (may help, if relevant):
        {caption_hint if caption_hint else "None"}

        Output format:
        Image description: <concise factual description>

        Rules:
        - Be concise, factual, and retrieval-friendly.
        - Do not hallucinate.
        - Mention important visible content such as labels, axes, trends, steps, components,
        or text only if clearly present.
        - Ignore branding and repeated decorative elements.
    """

    client = genai.Client()

    response = client.models.generate_content(
        model=os.getenv("GEMINI_VLM_MODEL", "gemini-3.1-flash-lite-preview"),
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    text = (response.text or "").strip()

    if text in {"SKIP_IMAGE", "Image description: SKIP_IMAGE"}:
        return "SKIP_IMAGE"

    return text


def ingest_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    path = Path(file_path)
    file_hash = _file_hash(file_path)
    embedder = get_embedding_model()

    with get_connection() as conn:
        if _doc_already_ingested(conn, file_hash):
            print(f"[ingest] '{path.name}' already ingested — skipping.")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id FROM documents WHERE file_hash = %s",
                    (file_hash,),
                )
                row = cur.fetchone()
                return row["id"] if row else None

        # Insert document row
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, title, source_path, file_hash)
                VALUES (%s, %s, %s, %s)
                RETURNING id::text AS id;
                """,
                (path.name, path.stem, str(path), file_hash),
            )
            document_id = cur.fetchone()["id"]

        print(f"[ingest] Created document id={document_id} for '{path.name}'")

        # Parse & chunk with Docling
        print("[ingest] Parsing PDF with Docling …")
        chunks = parse_and_chunk(file_path)
        print(f"[ingest] {len(chunks)} raw chunks extracted")

        # Create image output directory
        img_dir = Path("img") / path.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        # Convert image chunks into text using Gemini VLM
        final_chunks = []
        image_count = 0
        skipped_images = 0

        for chunk in chunks:
            if chunk.get("type") == "image" and chunk.get("image") is not None:
                image_count += 1

                # Save every image regardless of relevance
                page_part = (
                    f"page_{chunk.get('page')}"
                    if chunk.get("page") is not None
                    else "page_unknown"
                )
                image_filename = f"{path.stem}_{page_part}_img_{image_count}.png"
                saved_path = _save_image(chunk["image"], img_dir, image_filename)
                print(f"[ingest] Saved image {image_count} -> {saved_path}")

                print(f"[ingest] Describing image {image_count} with Gemini …")

                vlm_desc = _describe_image_with_gemini(
                    pil_image=chunk["image"],
                    captions=chunk.get("captions", []),
                ).strip()

                if vlm_desc == "SKIP_IMAGE":
                    skipped_images += 1
                    print(f"[ingest] Skipping decorative/repeated image {image_count}")
                    continue

                caption_text = " ".join(chunk.get("captions", [])).strip()

                parts = []
                if caption_text:
                    parts.append(f"Image caption: {caption_text}")
                parts.append(
                    vlm_desc
                    if vlm_desc.startswith("Image description:")
                    else f"Image description: {vlm_desc}"
                )

                chunk["text"] = "\n".join(parts).strip()
                chunk["type"] = "image_caption"   # important
                del chunk["image"]

                final_chunks.append(chunk)

            else:
                if chunk.get("text", "").strip():
                    final_chunks.append(chunk)

        chunks = final_chunks

        print(f"[ingest] {skipped_images} image chunks skipped")
        print(f"[ingest] {len(chunks)} final chunks after filtering")

        pages = [c["page"] for c in chunks if c.get("page") is not None]
        page_count = max(pages) if pages else None

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET page_count = %s WHERE id = %s",
                (page_count, document_id),
            )

        # Embed all chunk texts
        valid_chunks = []

        for chunk in chunks:
            text = str(chunk.get("text", "") or "").strip()
            if not text:
                continue

            # normalize text back into chunk
            chunk["text"] = text
            valid_chunks.append(chunk)

        print(f"[ingest] {len(valid_chunks)} chunks to embed")

        if not valid_chunks:
            print("[ingest] No valid text chunks found to embed.")
            conn.commit()
            return document_id

        texts = [c["text"] for c in valid_chunks]


        # Embed chunks
        print("[ingest] Embedding chunks …")
        embeddings = embedder.embed_documents(texts)

        # Safety check
        if len(embeddings) != len(valid_chunks):
            print(
                f"[warning] Embedding count mismatch. "
                f"Expected {len(valid_chunks)}, got {len(embeddings)}."
            )
            print("[warning] Retrying one-by-one embedding to identify problematic chunks...")

            recovered_chunks = []
            recovered_embeddings = []

            for i, chunk in enumerate(valid_chunks):
                try:
                    emb = embedder.embed_documents([chunk["text"]])[0]
                    recovered_chunks.append(chunk)
                    recovered_embeddings.append(emb)
                except Exception as e:
                    print(f"[warning] Failed to embed chunk {i} (page={chunk.get('page')}): {e}")
                    print(f"[warning] Skipping problematic chunk text preview: {chunk['text'][:300]}")

            valid_chunks = recovered_chunks
            embeddings = recovered_embeddings

        print(f"[ingest] {len(embeddings)} embeddings ready for DB insert")

        # Optional final guard
        if len(embeddings) != len(valid_chunks):
            raise RuntimeError(
                f"Embedding mismatch after retry: "
                f"{len(valid_chunks)} chunks vs {len(embeddings)} embeddings"
            )

        # Insert chunks
        with conn.cursor() as cur:
            for idx, (chunk, embedding) in enumerate(zip(valid_chunks, embeddings)):
                metadata = {
                    "headings": chunk.get("headings", []),
                    "captions": chunk.get("captions", []),
                    "contains_table": chunk.get("type") == "table",
                }

                cur.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        content,
                        embedding,
                        chunk_type,
                        source_page,
                        document_name,
                        metadata
                    )
                    VALUES (%s, %s, %s::vector, %s, %s, %s, %s);
                    """,
                    (
                        document_id,
                        chunk["text"],
                        embedding,
                        chunk.get("type", "text"),
                        chunk.get("page"),
                        path.name,
                        Jsonb(metadata),
                    ),
                )
        conn.commit()

    print(f"[ingest] Ingestion complete — {len(chunks)} chunks stored.")
    return document_id


if __name__ == "__main__":
    ingest_pdf("data/KB_Credit_Card_Spend_Summarizer.pdf")
