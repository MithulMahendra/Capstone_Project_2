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
