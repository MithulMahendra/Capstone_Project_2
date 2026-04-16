import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
import re
from typing import Any, Dict, List, Optional
import json
from src.api.v1.graph.state import GraphState

load_dotenv(override=True)

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )


def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDINGS_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=1536,
    )

def get_sql_database() -> SQLDatabase:
    url = os.getenv("SQL_DATABASE_URL")
    if not url:
        raise ValueError("SQL_DATABASE_URL is not found")
    return SQLDatabase.from_uri(url)


def _safe_text(content: Any) -> str:
    """
    Convert model output content into plain string safely.
    """
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def _current_query(state: GraphState) -> str:
    """
    Use rephrased query if present, otherwise original query.
    """
    return state.get("rephrased_query") or state["query"]


def _clean_sql(sql: str) -> str:
    """
    Clean markdown/sql wrappers from model-generated SQL.
    """
    sql = sql.strip().strip("```").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


def _is_safe_select_query(sql: str) -> bool:
    """
    Allow only SELECT / WITH...SELECT style read-only queries.
    Block obviously dangerous statements.
    """
    normalized = re.sub(r"\s+", " ", sql.strip().lower())

    dangerous = [
        " insert ", " update ", " delete ", " drop ", " alter ",
        " truncate ", " create ", " grant ", " revoke ", " merge "
    ]
    if any(tok in f" {normalized} " for tok in dangerous):
        return False

    return normalized.startswith("select") or normalized.startswith("with")


def _looks_like_has_data(sql_result: str) -> bool:
    """
    Heuristic to determine whether SQL result contains usable data.
    """
    if not sql_result:
        return False

    text = str(sql_result).strip()
    if not text:
        return False

    if "SQL execution error" in text:
        return False

    empty_markers = {
        "[]",
        "{}",
        "()",
        "",
        "No rows returned",
        "None",
        "null",
    }
    if text in empty_markers:
        return False

    return len(text) > 2


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON object from model output.
    More robust against markdown fences, extra prose, and partial wrappers.
    """
    if not text:
        return None

    text = text.strip()

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Remove markdown fences
    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Regex fallback
    answer_found_match = re.search(
        r'"answer_found"\s*:\s*(true|false)',
        cleaned,
        flags=re.IGNORECASE
    )
    answer_match = re.search(
        r'"answer"\s*:\s*"([^"]*)"',
        cleaned,
        flags=re.DOTALL
    )

    if answer_found_match or answer_match:
        return {
            "answer_found": (
                answer_found_match.group(1).lower() == "true"
                if answer_found_match else False
            ),
            "answer": answer_match.group(1).strip() if answer_match else "",
        }

    return None


def _chunk_to_searchable_text(chunk: Dict[str, Any]) -> str:
    """
    Build a richer text representation of a chunk for reranking/summarization.
    Especially useful for table chunks and structured data.
    """
    metadata = chunk.get("metadata", {}) or {}
    headings = metadata.get("headings", []) or []
    captions = metadata.get("captions", []) or []

    parts: List[str] = []

    document_name = str(chunk.get("document_name", "")).strip()
    if document_name:
        parts.append(f"Document: {document_name}")

    chunk_type = str(chunk.get("chunk_type", "")).strip()
    if chunk_type:
        parts.append(f"Chunk Type: {chunk_type}")

    source_page = chunk.get("source_page")
    if source_page not in (None, "", "N/A"):
        parts.append(f"Source Page: {source_page}")

    if headings:
        heading_text = " > ".join(str(h).strip() for h in headings if str(h).strip())
        if heading_text:
            parts.append(f"Headings: {heading_text}")

    if captions:
        caption_text = " | ".join(str(c).strip() for c in captions if str(c).strip())
        if caption_text:
            parts.append(f"Captions: {caption_text}")

    content = str(chunk.get("content", "")).strip()
    if content:
        parts.append("Content:")
        parts.append(content)

    # Common structured/table fields if your ingestion pipeline stores them
    possible_table_fields = [
        "table_text",
        "table_markdown",
        "table_rows",
        "rows",
        "cells",
        "structured_content",
    ]

    for key in possible_table_fields:
        value = chunk.get(key)
        if value:
            parts.append(f"{key}:")
            if isinstance(value, str):
                parts.append(value)
            else:
                try:
                    parts.append(json.dumps(value, ensure_ascii=False))
                except Exception:
                    parts.append(str(value))

    return "\n".join(parts).strip()

