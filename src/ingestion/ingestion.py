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
